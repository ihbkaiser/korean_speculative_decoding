#!/usr/bin/env python3
"""Correctness suite and staged benchmark for NP_BOUNDARY_GUARD.

The command is intended to be launched inside a dedicated tmux session under
CUDA_VISIBLE_DEVICES=3. It refuses to load model weights unless the process
sees exactly one A100 at framework-local cuda:0.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import e2_model_pairs as e2
from src.morphology_guard import MorphologyGuard

OUT = ROOT / "runs/morphology_aware_boundary_guard"
VARIANTS = [
    "TARGET_ONLY", "FIXED_K_SD", "NP_BOUNDARY_GUARD", "ALL_MORPH_BOUNDARY_GUARD",
    "RANDOM_MATCHED_GUARD", "ENTROPY_MATCHED_GUARD",
]
SD_VARIANTS = VARIANTS[1:]
WORKLOADS = ["WIKIPEDIA", "FLORES"]
REPEATS = 3
BOOTSTRAP_SEED = 3091
PARITY_EVENT_FIELDS = [
    "round_index", "proposal_slot", "output_token_position", "draft_proposed_token_id",
    "target_verification_token_id", "accepted", "rejected", "sd_valid",
    "is_first_rejection", "invalidated_after_first_rejection", "accepted_prefix_length",
    "first_rejection_output_position", "guard_triggered", "guard_cut_slot",
    "guard_applied", "guard_unverified", "reference_target_token_id",
]


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False, engine="pyarrow")
    tmp.replace(path)


AUDIT_KEY_COLUMNS = [
    "stage", "repetition", "workload", "variant", "pair", "prompt_id", "round_index", "proposal_slot",
]


def persist_audit_chunk(records: list[dict[str, Any]], stage: str, workload: str, pair: str, variant: str, repetition: int) -> None:
    """Durably checkpoint morphology decisions before prompt progress is committed."""
    if not records:
        return
    frame = pd.DataFrame(records)
    prompt_ids = sorted(frame.prompt_id.astype(int).unique().tolist())
    shard_dir = OUT / "audit_checkpoints" / stage
    shard = shard_dir / (
        f"{workload.lower()}_{pair.lower()}_{variant.lower()}_rep{repetition}_"
        f"{prompt_ids[0]}-{prompt_ids[-1]}.parquet"
    )
    if shard.exists():
        frame = pd.concat([pd.read_parquet(shard), frame], ignore_index=True)
    frame = frame.drop_duplicates(AUDIT_KEY_COLUMNS, keep="last")
    atomic_parquet(frame, shard)


def consolidate_audit_chunks(stages: list[str], destination: Path) -> None:
    """Build the requested audit artifact from durable per-chunk shards."""
    frames: list[pd.DataFrame] = []
    if destination.exists():
        reader = pd.read_csv if destination.suffix == ".csv" else pd.read_parquet
        frames.append(reader(destination))
    for stage in stages:
        for shard in sorted((OUT / "audit_checkpoints" / stage).glob("*.parquet")):
            frames.append(pd.read_parquet(shard))
    if not frames:
        return
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(AUDIT_KEY_COLUMNS, keep="last")
    if destination.suffix == ".csv":
        atomic_csv(frame, destination)
    else:
        atomic_parquet(frame, destination)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    headers = [str(x) for x in frame.columns]
    body = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(headers[i]), *(len(row[i]) for row in body)) for i in range(len(headers))]
    head = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    rows = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in body]
    return "\n".join([head, sep, *rows])


class PhysicalGpu3Monitor:
    def __init__(self, path: Path, interval: float = 10.0):
        self.path = path
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write("timestamp_utc,utilization_gpu_pct,memory_used_mib,memory_free_mib\n")
                f.flush()
            while not self.stop_event.is_set():
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--id=3", "--query-gpu=utilization.gpu,memory.used,memory.free", "--format=csv,noheader,nounits"],
                        capture_output=True, check=True, text=True, timeout=5,
                    ).stdout.strip().replace(" ", "")
                    f.write(f"{datetime.now(timezone.utc).isoformat()},{result}\n")
                    f.flush()
                except Exception as exc:
                    f.write(f"{datetime.now(timezone.utc).isoformat()},error,{type(exc).__name__},\n")
                    f.flush()
                self.stop_event.wait(self.interval)

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="physical-gpu3-monitor", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10)


def gpu3_preflight() -> tuple[Any, dict[str, Any]]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("Refusing model work: set CUDA_VISIBLE_DEVICES=3")
    inventory = subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, text=True).stdout.strip()
    physical3 = subprocess.run(
        ["nvidia-smi", "--id=3", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    nvidia_smi_text = subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True).stdout
    import torch
    import transformers

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible device on cuda:0; got {torch.cuda.device_count()}")
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    if "A100" not in name or props.total_memory < 78 * 1024**3:
        raise RuntimeError(f"Expected physical GPU 3 A100 80GB, got {name} ({props.total_memory} bytes)")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(3090)
    np.random.seed(3090)
    torch.manual_seed(3090)
    torch.cuda.manual_seed_all(3090)
    metadata = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "nvidia_smi_L_output": inventory,
        "nvidia_smi_physical_gpu3": physical3,
        "physical_gpu_index": 3,
        "GPU_name": name,
        "driver_version": physical3.split(",")[1].strip() if "," in physical3 else None,
        "nvidia_smi_cuda_version": nvidia_smi_text.split("CUDA Version:")[-1].split("|")[0].strip() if "CUDA Version:" in nvidia_smi_text else None,
        "torch_cuda_runtime_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "attention_backend": "sdpa",
        "dtype": "torch.float16",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "framework_local_device": "cuda:0",
        "visible_device_count": torch.cuda.device_count(),
        "device_total_memory_bytes": int(props.total_memory),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    if torch.cuda.current_device() != 0:
        raise RuntimeError("Framework-local device must be cuda:0")
    return torch, metadata


def load_models(torch: Any) -> tuple[dict[str, Any], Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    specs = e2.MODELS
    tokenizer = AutoTokenizer.from_pretrained(
        specs["4B"]["id"], revision=specs["4B"]["revision"], use_fast=True, local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError("Fast tokenizer with offsets is required")
    models: dict[str, Any] = {}
    for size in ["0.6B", "1.7B", "4B"]:
        spec = specs[size]
        model = AutoModelForCausalLM.from_pretrained(
            spec["id"], revision=spec["revision"], torch_dtype=torch.float16,
            low_cpu_mem_usage=True, attn_implementation="sdpa", local_files_only=True,
        )
        model.to("cuda:0")
        model.eval()
        models[size] = model
        torch.cuda.synchronize(0)
        print(f"Loaded {size} @ pinned revision {spec['revision']} on cuda:0", flush=True)
    return models, tokenizer


def load_inputs() -> tuple[pd.DataFrame, dict[tuple[str, str, int], list[int]], dict[str, Any]]:
    config = json.loads((OUT / "method_config.json").read_text(encoding="utf-8"))
    prompt_frame = pd.read_csv(OUT / "prompt_manifest.csv")
    prompt_frame["prompt_ids"] = prompt_frame.prompt_ids_json.map(json.loads)
    refs = pd.read_parquet(OUT / "source_target_references.parquet")
    reference_by_key = {
        (str(r.workload), str(r.pair), int(r.prompt_id)): [int(x) for x in r.target_token_ids]
        for r in refs.itertuples(index=False)
    }
    if len(reference_by_key) != 3 * (1000 + 1012):
        raise AssertionError("Target-reference artifact is missing pair/workload records")
    return prompt_frame, reference_by_key, config


def make_controller(variant: str, tokenizer: Any, kiwi: Any, pair: str, config: dict[str, Any]) -> MorphologyGuard | None:
    if variant in {"TARGET_ONLY", "FIXED_K_SD"}:
        return None
    threshold_info = config["entropy_thresholds_wikipedia_only"][pair]
    threshold = threshold_info.get("threshold")
    distribution = {int(k): float(v) for k, v in config["wikipedia_pilot_slot_distributions"].get(pair, {}).items()}
    entropy_slot_probabilities = {
        int(k): float(v) for k, v in threshold_info.get("slot_thinning_probabilities", {}).items()
    }
    return MorphologyGuard(
        variant=variant, tokenizer=tokenizer, kiwi=kiwi, pair=pair,
        slot_distribution=distribution, entropy_threshold=threshold,
        entropy_slot_probabilities=entropy_slot_probabilities, seed=int(config["random_guard_seed"]),
    )


def output_metrics(output: list[int], reference: list[int]) -> dict[str, Any]:
    return {
        "output_tokens": len(output),
        "reference_tokens": len(reference),
        "exact_output_match": output == reference,
        "exact_stopping_position_match": len(output) == len(reference),
        "eos_terminated": bool(output and output[-1] == 151643),
        "output_token_ids_json": json.dumps(output, separators=(",", ":")),
    }


def _sd_summary(stats: dict[str, Any], output_tokens: int) -> dict[str, Any]:
    rounds = int(stats.get("rounds", 0))
    verified = int(stats.get("target_proposal_positions_verified", 0))
    return {
        "rounds": rounds,
        "draft_forward_calls": int(stats.get("draft_forward_calls", 0)),
        "target_forward_calls": int(stats.get("target_forward_calls", 0)),
        "draft_tokens_proposed": int(stats.get("draft_tokens_proposed", 0)),
        "target_proposal_positions_verified": verified,
        "accepted_draft_tokens": int(stats.get("accepted_draft_tokens", 0)),
        "rejected_draft_tokens": int(stats.get("rejected_draft_tokens", 0)),
        "mean_accepted_tokens_per_verification_call": float(stats.get("accepted_draft_tokens", 0) / max(1, rounds)),
        "rejection_rate": float(stats.get("rejected_draft_tokens", 0) / max(1, verified)),
        "target_forward_calls_per_output_token": float(stats.get("target_forward_calls", 0) / max(1, output_tokens)),
        "draft_tokens_proposed_per_output_token": float(stats.get("draft_tokens_proposed", 0) / max(1, output_tokens)),
        "target_proposal_positions_verified_per_output_token": float(verified / max(1, output_tokens)),
        "guard_triggered_rounds": int(stats.get("guard_triggered_rounds", 0)),
        "guard_applied_rounds": int(stats.get("guard_applied_rounds", 0)),
        "guard_activation_rate": float(stats.get("guard_triggered_rounds", 0) / max(1, rounds)),
        "guard_position_histogram_json": json.dumps({str(k): stats.get("guard_positions", []).count(k) for k in sorted(set(stats.get("guard_positions", [])))}, separators=(",", ":")),
        "candidate_positions_checked": int(stats.get("candidate_positions_checked", 0)),
        "invalid_candidate_positions": int(stats.get("invalid_candidate_positions", 0)),
        "invalid_or_ambiguous_candidate_fraction": float(stats.get("invalid_candidate_positions", 0) / max(1, stats.get("candidate_positions_checked", 0))),
        "morphology_detection_seconds": float(stats.get("morphology_seconds", 0.0)),
        "controller_seconds": float(stats.get("controller_seconds", 0.0)),
        "draft_resynchronization_seconds": float(stats.get("draft_resynchronization_seconds", 0.0)),
        "guard_audit": stats.get("guard_audit", []),
    }


def run_prompt(
    torch: Any,
    models: dict[str, Any],
    tokenizer: Any,
    kiwi: Any,
    pair: str,
    workload: str,
    variant: str,
    prompt: dict[str, Any],
    reference: list[int],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch.cuda.synchronize(0)
    start = time.perf_counter()
    controller = make_controller(variant, tokenizer, kiwi, pair, config)
    stats: dict[str, Any] = {}
    if variant == "TARGET_ONLY":
        model = models[e2.PAIRS[pair]["target"]]
        output = e2.greedy_generate_single(model, prompt["prompt_ids"], 128, 151643)
        stats_summary: dict[str, Any] = {}
    else:
        draft = models[e2.PAIRS[pair]["draft"]]
        target = models[e2.PAIRS[pair]["target"]]
        output, events = e2.speculative_greedy_fast(
            draft, target, prompt["prompt_ids"], 128, 151643, 4, int(prompt["prompt_id"]),
            guard_controller=controller, runtime_stats=stats,
            batch_target_verification=bool(config.get("batch_target_verification", False)),
        )
        stats_summary = _sd_summary(stats, len(output))
        stats_summary["events"] = events
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - start
    result = {
        "workload": workload, "pair": pair, "variant": variant,
        "prompt_id": int(prompt["prompt_id"]), "prompt_hash": str(prompt["prompt_hash"]),
        "prompt_wall_seconds": elapsed,
        **output_metrics(output, reference),
        **{k: v for k, v in stats_summary.items() if k not in {"guard_audit", "events"}},
    }
    audit = stats_summary.get("guard_audit", [])
    if variant != "TARGET_ONLY":
        valid_events = [e for e in stats_summary.get("events", []) if e.get("sd_valid")]
        result["valid_event_count"] = len(valid_events)
        result["verified_rejection_count"] = sum(bool(e.get("rejected")) for e in valid_events)
    else:
        audit = []
    for row in audit:
        row["workload"] = workload
        row["variant"] = variant
    return result, audit


def warm_up(torch: Any, models: dict[str, Any], tokenizer: Any, kiwi: Any, pair: str, workload: str, variant: str, prompt: dict[str, Any], reference: list[int], config: dict[str, Any]) -> None:
    # A short, fixed warm-up avoids benchmarking lazy CUDA kernels on prompt one.
    torch.cuda.synchronize(0)
    if variant == "TARGET_ONLY":
        e2.greedy_generate_single(models[e2.PAIRS[pair]["target"]], prompt["prompt_ids"], 4, 151643)
    else:
        controller = make_controller(variant, tokenizer, kiwi, pair, config)
        e2.speculative_greedy_fast(
            models[e2.PAIRS[pair]["draft"]], models[e2.PAIRS[pair]["target"]],
            prompt["prompt_ids"], 4, 151643, 4, int(prompt["prompt_id"]), guard_controller=controller,
            batch_target_verification=bool(config.get("batch_target_verification", False)),
        )
    torch.cuda.synchronize(0)


def _bootstrap_rate(prompt_rows: pd.DataFrame, reps: int = 1000) -> tuple[float, float]:
    if prompt_rows.empty:
        return float("nan"), float("nan")
    token_counts = prompt_rows.output_tokens.to_numpy(dtype=float)
    seconds = prompt_rows.prompt_wall_seconds.to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(prompt_rows)
    samples = np.empty(reps, dtype=float)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        samples[i] = token_counts[idx].sum() / max(1e-12, seconds[idx].sum())
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def result_row(workload: str, pair: str, variant: str, repetition: int, prompt_rows: pd.DataFrame, wall_seconds: float, peak_allocated: float, peak_reserved: float) -> dict[str, Any]:
    tokens = int(prompt_rows.output_tokens.sum())
    rate = tokens / max(1e-12, wall_seconds)
    ci_low, ci_high = _bootstrap_rate(prompt_rows, 1000)
    sd_rows = prompt_rows.loc[prompt_rows.variant.ne("TARGET_ONLY")]
    def mean_column(col: str) -> float:
        return float(sd_rows[col].mean()) if col in sd_rows and len(sd_rows) else float("nan")
    return {
        "workload": workload, "model_pair": pair, "variant": variant, "repetition": int(repetition),
        "prompts": len(prompt_rows), "generated_target_tokens": tokens,
        "wall_clock_seconds": float(wall_seconds), "target_tokens_per_second": rate,
        "throughput_bootstrap_ci_low": ci_low, "throughput_bootstrap_ci_high": ci_high,
        "mean_prompt_latency_seconds": float(prompt_rows.prompt_wall_seconds.mean()),
        "median_prompt_latency_seconds": float(prompt_rows.prompt_wall_seconds.median()),
        "seconds_per_generated_target_token": float(wall_seconds / max(1, tokens)),
        "speedup_vs_target_only": float("nan"), "speedup_vs_fixed_k_sd": float("nan"),
        "mean_accepted_tokens_per_verification_call": mean_column("mean_accepted_tokens_per_verification_call"),
        "rejection_rate": mean_column("rejection_rate"),
        "target_forward_calls_per_output_token": mean_column("target_forward_calls_per_output_token"),
        "draft_tokens_proposed_per_output_token": mean_column("draft_tokens_proposed_per_output_token"),
        "target_proposal_positions_verified_per_output_token": mean_column("target_proposal_positions_verified_per_output_token"),
        "guard_activation_rate": mean_column("guard_activation_rate"),
        "guard_applied_rounds_mean": mean_column("guard_applied_rounds"),
        "invalid_or_ambiguous_candidate_fraction": mean_column("invalid_or_ambiguous_candidate_fraction"),
        "morphology_detection_seconds": float(sd_rows.get("morphology_detection_seconds", pd.Series(dtype=float)).sum()),
        "controller_seconds": float(sd_rows.get("controller_seconds", pd.Series(dtype=float)).sum()),
        "draft_resynchronization_seconds": float(sd_rows.get("draft_resynchronization_seconds", pd.Series(dtype=float)).sum()),
        "peak_allocated_vram_gb": float(peak_allocated), "peak_reserved_vram_gb": float(peak_reserved),
        "all_outputs_exact": bool(prompt_rows.exact_output_match.astype(bool).all()),
    }


def checkpoint_path(stage: str, workload: str, pair: str, variant: str, repetition: int) -> Path:
    return OUT / stage / "checkpoints" / f"{workload.lower()}_{pair.lower()}_{variant.lower()}_rep{repetition}.csv"


def run_combo(
    torch: Any, models: dict[str, Any], tokenizer: Any, kiwi: Any, config: dict[str, Any],
    prompts: pd.DataFrame, references: dict[tuple[str, str, int], list[int]],
    stage: str, workload: str, pair: str, variant: str, repetition: int,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    selected = prompts.loc[
        prompts.workload.eq(workload)
        & (prompts.correctness_selected if stage == "correctness" else prompts.pilot_selected if stage == "pilot" else True)
    ].sort_values("source_index")
    outdir = OUT / ("correctness" if stage == "correctness" else "pilot" if stage == "pilot" else "full_benchmark")
    path = checkpoint_path(stage, workload, pair, variant, repetition)
    state_path = path.with_suffix(".state.json")
    summary_path = path.with_suffix(".summary.json")
    if path.exists():
        done = pd.read_csv(path)
    else:
        done = pd.DataFrame()
    completed = set(done.prompt_id.astype(int)) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []
    audits: list[dict[str, Any]] = []
    if len(selected) == 0:
        raise RuntimeError(f"No prompts selected for {stage}/{workload}")
    warm_prompt_row = next(selected.itertuples(index=False))
    warm_prompt = warm_prompt_row._asdict()
    warm_prompt["prompt_ids"] = json.loads(warm_prompt["prompt_ids_json"])
    warm_key = (workload, pair, int(warm_prompt["prompt_id"]))
    warm_up(torch, models, tokenizer, kiwi, pair, workload, variant, warm_prompt, references[warm_key], config)
    expected_ids = set(selected.prompt_id.astype(int))
    if completed == expected_ids and summary_path.exists():
        return done, json.loads(summary_path.read_text(encoding="utf-8")), []
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    checkpoint_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    saved_elapsed = float(checkpoint_state.get("elapsed_seconds", 0.0))
    state_completed = int(checkpoint_state.get("completed_prompts", 0))
    if state_completed > len(rows):
        raise RuntimeError(f"Checkpoint state is ahead of prompt rows for {stage}/{workload}/{pair}/{variant}")
    if len(rows) > state_completed:
        # The prompt CSV is atomically replaced before the state JSON. Recover a
        # crash in that narrow window from the newly committed prompt timings.
        recovered = rows[state_completed:]
        saved_elapsed += sum(float(row.get("prompt_wall_seconds", 0.0)) for row in recovered)
    segment_start = time.perf_counter()
    for ordinal, record in enumerate(selected.to_dict("records"), start=1):
        pid = int(record["prompt_id"])
        if pid in completed:
            continue
        record["prompt_ids"] = json.loads(record["prompt_ids_json"])
        reference = references[(workload, pair, pid)]
        prompt_result, prompt_audit = run_prompt(
            torch, models, tokenizer, kiwi, pair, workload, variant, record, reference, config,
        )
        for audit_row in prompt_audit:
            audit_row["stage"] = stage
            audit_row["repetition"] = repetition
        rows.append(prompt_result)
        audits.extend(prompt_audit)
        if ordinal % 10 == 0 or ordinal == len(selected):
            # Persist decision evidence first. If interrupted before the prompt
            # checkpoint, the same chunk is replayed and its stable keys dedupe.
            persist_audit_chunk(audits, stage, workload, pair, variant, repetition)
            audits.clear()
            atomic_csv(pd.DataFrame(rows), path)
            saved_elapsed += time.perf_counter() - segment_start
            atomic_json(state_path, {"elapsed_seconds": saved_elapsed, "completed_prompts": len(rows), "updated_at_utc": datetime.now(timezone.utc).isoformat()})
            segment_start = time.perf_counter()
            print(f"{stage} {workload}/{pair}/{variant}/rep{repetition}: {ordinal}/{len(selected)} prompts", flush=True)
        if stage == "correctness" and not prompt_result["exact_output_match"]:
            atomic_csv(pd.DataFrame(rows), path)
            raise RuntimeError(f"Greedy exactness failed for {workload}/{pair}/{variant}/prompt_id={pid}")
    torch.cuda.synchronize(0)
    wall_seconds = saved_elapsed + (time.perf_counter() - segment_start)
    frame = pd.DataFrame(rows).sort_values("prompt_id")
    if set(frame.prompt_id.astype(int)) != expected_ids:
        raise AssertionError(f"Incomplete checkpoint {path}: {len(frame)} of {len(expected_ids)} prompts")
    if stage == "correctness" and not frame.exact_output_match.astype(bool).all():
        raise RuntimeError(f"Exactness failed in {stage}/{workload}/{pair}/{variant}")
    peak_allocated = float(torch.cuda.max_memory_allocated(0) / 1024**3)
    peak_reserved = float(torch.cuda.max_memory_reserved(0) / 1024**3)
    row = result_row(workload, pair, variant, repetition, frame, wall_seconds, peak_allocated, peak_reserved)
    row["correctness_wall_clock_seconds"] = wall_seconds if stage == "correctness" else float("nan")
    atomic_json(summary_path, row)
    return frame, row, audits


def write_sampling_audit() -> None:
    path = OUT / "correctness/sampling_logic_audit.md"
    text = """# Sampling logic audit

Repository source audit (`src/e2_model_pairs.py`, `src/speculative_decoding.py`) found only deterministic greedy speculative decoding in the E2/FLORES path. The online acceptance rule is token-ID equality between each draft proposal and the target argmax; a rejected proposal is replaced with that target argmax. There is no multinomial proposal sampler, acceptance-probability ratio, residual distribution, or RNG-driven speculative-sampling path in this codebase.

`NP_BOUNDARY_GUARD` leaves this greedy argmax and correction behavior unchanged. After a fully accepted safe prefix, it emits the existing target argmax for the first unverified position and advances the target KV cache exactly as for a normal correction token. Therefore this run verifies greedy token-ID equivalence only. It does **not** implement, alter, or make any claim about exact stochastic speculative sampling. Sampling exactness remains outside the repository's implemented decoder and is not evaluated here.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _event_signature(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: event.get(key) for key in PARITY_EVENT_FIELDS} for event in events]


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_batched_target_parity(
    torch: Any,
    models: dict[str, Any],
    tokenizer: Any,
    kiwi: Any,
    config: dict[str, Any],
    prompts: pd.DataFrame,
    references: dict[tuple[str, str, int], list[int]],
) -> None:
    """Compare one-call causal block verification with the sequential reference path.

    The fixed 8 prompt subset comes from the already frozen correctness sample,
    in source order. All five SD variants, both workloads, and all three pairs
    are checked before enabling the batched path for the official suite.
    """
    out_path = OUT / "correctness/batched_target_verification_parity.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path)
        failures = existing.loc[existing.result.ne("PASS")]
        if len(failures):
            raise RuntimeError("A prior batched-verification parity run contains failures")
    else:
        existing = pd.DataFrame()
    completed = {
        (str(row.workload), str(row.model_pair), str(row.variant), int(row.prompt_id))
        for row in existing.itertuples(index=False)
        if str(row.result) == "PASS"
    }
    prompts = prompts.copy()
    prompts["prompt_ids"] = prompts.prompt_ids_json.map(json.loads)
    for workload in WORKLOADS:
        for pair in e2.PAIRS:
            subset = prompts.loc[
                prompts.workload.eq(workload) & prompts.correctness_selected.astype(bool)
            ].sort_values("source_index").head(8)
            if len(subset) != 8:
                raise RuntimeError(f"Need 8 frozen correctness prompts for {workload}/{pair}; found {len(subset)}")
            for variant in SD_VARIANTS:
                for record in subset.to_dict("records"):
                    pid = int(record["prompt_id"])
                    key = (workload, pair, variant, pid)
                    if key in completed:
                        continue
                    reference = references[(workload, pair, pid)]
                    signatures: dict[str, Any] = {}
                    outputs: dict[str, list[int]] = {}
                    durations: dict[str, float] = {}
                    for mode, batched in [("sequential", False), ("batched", True)]:
                        controller = make_controller(variant, tokenizer, kiwi, pair, config)
                        stats: dict[str, Any] = {}
                        torch.cuda.synchronize(0)
                        started = time.perf_counter()
                        output, events = e2.speculative_greedy_fast(
                            models[e2.PAIRS[pair]["draft"]], models[e2.PAIRS[pair]["target"]],
                            record["prompt_ids"], 128, 151643, 4, pid,
                            guard_controller=controller, runtime_stats=stats,
                            batch_target_verification=batched,
                        )
                        torch.cuda.synchronize(0)
                        durations[mode] = time.perf_counter() - started
                        outputs[mode] = output
                        signatures[mode] = _event_signature(events)
                    output_equal = outputs["sequential"] == outputs["batched"]
                    events_equal = signatures["sequential"] == signatures["batched"]
                    target_equal = outputs["sequential"] == reference and outputs["batched"] == reference
                    passed = output_equal and events_equal and target_equal
                    row = {
                        "workload": workload, "model_pair": pair, "variant": variant, "prompt_id": pid,
                        "result": "PASS" if passed else "FAIL",
                        "sequential_equals_batched": output_equal,
                        "event_decisions_equal": events_equal,
                        "both_equal_target_reference": target_equal,
                        "sequential_output_sha256": _sha_json(outputs["sequential"]),
                        "batched_output_sha256": _sha_json(outputs["batched"]),
                        "sequential_events_sha256": _sha_json(signatures["sequential"]),
                        "batched_events_sha256": _sha_json(signatures["batched"]),
                        "sequential_seconds": durations["sequential"], "batched_seconds": durations["batched"],
                        "output_tokens": len(outputs["sequential"]),
                    }
                    _store_prompt_rows(
                        [row], out_path,
                        ["workload", "model_pair", "variant", "prompt_id"],
                    )
                    if not passed:
                        raise RuntimeError(f"Batched target verification parity failed: {key}")
                print(f"batched-target parity {workload}/{pair}/{variant}: 8/8 PASS", flush=True)
    final = pd.read_csv(out_path)
    expected = len(WORKLOADS) * len(e2.PAIRS) * len(SD_VARIANTS) * 8
    if len(final) != expected or not final.result.eq("PASS").all():
        raise RuntimeError(f"Incomplete batched target verification parity: {len(final)}/{expected} cases")
    config["batch_target_verification"] = True
    config["batched_verification_parity"] = {
        "artifact": str(out_path.relative_to(ROOT)), "cases": int(len(final)),
        "prompts_per_workload_pair_variant": 8, "all_variants": SD_VARIANTS,
        "outputs_match_sequential_and_target_reference": bool(final.both_equal_target_reference.astype(bool).all()),
        "proposal_and_accept_reject_events_match_sequential": bool(final.event_decisions_equal.astype(bool).all()),
        "cache_rollback": "DynamicCache.crop removes every uncommitted proposal after the first rejection; cache length is asserted before correction.",
    }
    config["stage"] = "Batched target verification parity passed; official correctness suite ready"
    atomic_json(OUT / "method_config.json", config)


def _store_audits(records: list[dict[str, Any]], path: Path) -> None:
    if records:
        frame = pd.DataFrame(records)
        if path.exists():
            prior = pd.read_csv(path)
            frame = pd.concat([prior, frame], ignore_index=True)
        atomic_csv(frame, path)


def _store_prompt_rows(records: list[dict[str, Any]], path: Path, key_columns: list[str]) -> None:
    if not records:
        return
    frame = pd.DataFrame(records)
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    if len(frame):
        frame = frame.drop_duplicates(key_columns, keep="last")
    atomic_csv(frame, path)


def _store_parquet_rows(records: list[dict[str, Any]], path: Path, key_columns: list[str]) -> None:
    if not records:
        return
    frame = pd.DataFrame(records)
    if path.exists():
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
    if len(frame):
        frame = frame.drop_duplicates(key_columns, keep="last")
    atomic_parquet(frame, path)


def aggregate_results(rows: list[dict[str, Any]], destination: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if destination.exists():
        frames.append(pd.read_csv(destination))
    if rows:
        frames.append(pd.DataFrame(rows))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(frame):
        frame = frame.drop_duplicates(["workload", "model_pair", "variant", "repetition"], keep="last")
        fixed = frame.loc[frame.variant.eq("FIXED_K_SD"), ["workload", "model_pair", "repetition", "wall_clock_seconds"]].rename(columns={"wall_clock_seconds": "fixed_wall"})
        target = frame.loc[frame.variant.eq("TARGET_ONLY"), ["workload", "model_pair", "repetition", "wall_clock_seconds"]].rename(columns={"wall_clock_seconds": "target_wall"})
        frame = frame.drop(columns=["speedup_vs_target_only", "speedup_vs_fixed_k_sd"], errors="ignore")
        frame = frame.merge(fixed, on=["workload", "model_pair", "repetition"], how="left").merge(target, on=["workload", "model_pair", "repetition"], how="left")
        frame["speedup_vs_target_only"] = frame.target_wall / frame.wall_clock_seconds
        frame["speedup_vs_fixed_k_sd"] = frame.fixed_wall / frame.wall_clock_seconds
        frame = frame.drop(columns=["fixed_wall", "target_wall"])
        atomic_csv(frame, destination)
    return frame


def write_pilot_report(frame: pd.DataFrame, full_gate: bool) -> None:
    lines = [
        "# Pilot benchmark summary", "",
        "Pilot uses the frozen deterministic 200-prompt subset per workload, three synchronized repeats, and includes controller/morphology overhead.",
        "", markdown_table(frame.groupby(["workload", "model_pair", "variant"], dropna=False).agg(
            repeats=("repetition", "nunique"), mean_tokens_s=("target_tokens_per_second", "mean"),
            median_seconds_token=("seconds_per_generated_target_token", "median"),
            mean_speedup_fixed=("speedup_vs_fixed_k_sd", "mean"),
            exact=("all_outputs_exact", "all"),
        ).reset_index()) if len(frame) else "No pilot data.",
        "",
        f"Stage 4 full benchmark gate: **{'PASS' if full_gate else 'STOP (negative pilot: NP is >2% slower in all six pair×workload cells)'}**.",
    ]
    (OUT / "pilot/pilot_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "correctness", "pilot", "full", "verify-batched-target"], default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("Launch as CUDA_VISIBLE_DEVICES=3; no GPU has been inspected/loaded yet")
    torch, gpu_meta = gpu3_preflight()
    config_path = OUT / "method_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["gpu_metadata"] = gpu_meta
    config["gpu_requirement"]["inventory_and_versions_recorded_before_first_GPU_job"] = True
    config["stage"] = "GPU-3 preflight passed; model load starting"
    atomic_json(config_path, config)
    print(f"GPU preflight passed: physical GPU 3 -> {gpu_meta['GPU_name']} / cuda:0", flush=True)
    models, tokenizer = load_models(torch)
    config["gpu_metadata"]["all_model_weights_allocated_gb"] = float(torch.cuda.memory_allocated(0) / 1024**3)
    config["gpu_metadata"]["all_model_weights_reserved_gb"] = float(torch.cuda.memory_reserved(0) / 1024**3)
    config["stage"] = "All pinned model weights resident on physical GPU 3"
    atomic_json(config_path, config)
    monitor = PhysicalGpu3Monitor(OUT / "gpu3_utilization.csv")
    monitor.start()
    try:
        prompts, references, config = load_inputs()
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        write_sampling_audit()
        if args.mode == "verify-batched-target":
            verify_batched_target_parity(torch, models, tokenizer, kiwi, config, prompts, references)
            return
        correctness_rows: list[dict[str, Any]] = []
        pilot_rows: list[dict[str, Any]] = []
        full_rows: list[dict[str, Any]] = []

        if args.mode in {"all", "correctness"}:
            config["stage"] = "Correctness suite in progress"
            atomic_json(config_path, config)
            audit_path = OUT / "correctness/guard_decision_audit.csv"
            for workload in WORKLOADS:
                for pair in e2.PAIRS:
                    for variant in VARIANTS:
                        _, row, audits = run_combo(torch, models, tokenizer, kiwi, config, prompts, references, "correctness", workload, pair, variant, 0)
                        correctness_rows.append(row)
                        _store_prompt_rows(
                            audits, audit_path,
                            ["stage", "repetition", "workload", "variant", "pair", "prompt_id", "round_index", "proposal_slot"],
                        )
                        row_frame = pd.read_csv(checkpoint_path("correctness", workload, pair, variant, 0))
                        ref_lookup = {(workload, pair, int(pid)): references[(workload, pair, int(pid))] for pid in row_frame.prompt_id.astype(int)}
                        row_frame["target_reference_tokens"] = [len(ref_lookup[(workload, pair, int(pid))]) for pid in row_frame.prompt_id]
                        _store_prompt_rows(
                            row_frame.to_dict("records"), OUT / "correctness/greedy_equality.csv",
                            ["workload", "pair", "variant", "prompt_id"],
                        )
            consolidate_audit_chunks(["correctness"], OUT / "correctness/guard_decision_audit.csv")
            config["stage"] = "Correctness suite passed"
            atomic_json(config_path, config)
            if args.mode == "correctness":
                return

        if args.mode in {"all", "pilot"}:
            config["stage"] = "Pilot benchmark in progress"
            atomic_json(config_path, config)
            result_path = OUT / "pilot/benchmark_results.csv"
            audit_path = OUT / "correctness/guard_decision_audit.csv"
            for workload in WORKLOADS:
                for pair in e2.PAIRS:
                    for variant in VARIANTS:
                        for repetition in range(1, REPEATS + 1):
                            _, row, audits = run_combo(torch, models, tokenizer, kiwi, config, prompts, references, "pilot", workload, pair, variant, repetition)
                            pilot_rows.append(row)
                            _store_prompt_rows(
                                audits, audit_path,
                                ["stage", "repetition", "workload", "variant", "pair", "prompt_id", "round_index", "proposal_slot"],
                            )
                            aggregate_results(pilot_rows, result_path)
            pilot_frame = aggregate_results(pilot_rows, result_path)
            consolidate_audit_chunks(["correctness", "pilot"], OUT / "correctness/guard_decision_audit.csv")
            np_rows = pilot_frame.loc[pilot_frame.variant.eq("NP_BOUNDARY_GUARD")]
            ratios = np_rows.merge(
                pilot_frame.loc[pilot_frame.variant.eq("FIXED_K_SD"), ["workload", "model_pair", "repetition", "wall_clock_seconds"]],
                on=["workload", "model_pair", "repetition"], suffixes=("", "_fixed"), validate="one_to_one",
            )
            cell_min = ratios.groupby(["workload", "model_pair"]).apply(
                lambda g: float((g.wall_clock_seconds / g.wall_clock_seconds_fixed).mean()), include_groups=False,
            )
            # Full stage is barred only if every pair/workload cell is >2% slower.
            full_gate = not bool((cell_min >= 1.02).all()) and bool(np_rows.all_outputs_exact.astype(bool).all())
            write_pilot_report(pilot_frame, full_gate)
            write_cpu_analysis_report("pilot")
            if not full_gate:
                config["stage"] = "PILOT_NEGATIVE; full benchmark not launched by prespecified gate"
                config["pilot_gate"] = {"np_to_fixed_mean_wall_ratio_by_cell": {f"{w}/{p}": float(v) for (w, p), v in cell_min.items()}, "full_benchmark_launched": False}
                atomic_json(config_path, config)
                write_final_summary(pilot_frame, pd.DataFrame(), "NEGATIVE", "Pilot did not meet the full-run gate or exactness failed.")
                write_cpu_analysis_report("pilot")
                return
            config["stage"] = "PILOT_GATE_PASSED; full benchmark starting"
            config["pilot_gate"] = {"np_to_fixed_mean_wall_ratio_by_cell": {f"{w}/{p}": float(v) for (w, p), v in cell_min.items()}, "full_benchmark_launched": True}
            atomic_json(config_path, config)

        if args.mode in {"all", "full"}:
            config["stage"] = "Full benchmark in progress"
            atomic_json(config_path, config)
            result_path = OUT / "full_benchmark/benchmark_results.csv"
            for workload in WORKLOADS:
                for pair in e2.PAIRS:
                    for variant in VARIANTS:
                        for repetition in range(1, REPEATS + 1):
                            _, row, audits = run_combo(torch, models, tokenizer, kiwi, config, prompts, references, "full", workload, pair, variant, repetition)
                            full_rows.append(row)
                            _store_parquet_rows(
                                audits, OUT / "full_benchmark/guard_events.parquet",
                                ["repetition", "workload", "variant", "pair", "prompt_id", "round_index", "proposal_slot"],
                            )
                            aggregate_results(full_rows, result_path)
            full_frame = aggregate_results(full_rows, result_path)
            consolidate_audit_chunks(["full"], OUT / "full_benchmark/guard_events.parquet")
            prompt_checkpoints = [pd.read_csv(p) for p in (OUT / "full_benchmark/checkpoints").glob("*.csv")]
            if prompt_checkpoints:
                atomic_parquet(pd.concat(prompt_checkpoints, ignore_index=True), OUT / "full_benchmark/per_prompt_metrics.parquet")
            write_final_summary(pilot_frame if len(pilot_rows) else pd.read_csv(OUT / "pilot/benchmark_results.csv"), full_frame, "INCONCLUSIVE", "Full benchmark completed; apply the frozen decision rule in the CPU report generator.")
            write_cpu_analysis_report("full")
    finally:
        monitor.stop()


def write_final_summary(pilot: pd.DataFrame, full: pd.DataFrame, result: str, sentence: str) -> None:
    primary = full if len(full) else pilot
    lines = [
        f"# Morphology-aware boundary guard\n\nMethod result: {result} — {sentence}", "",
        "## Completed stages", "",
        f"Pilot benchmark rows: {len(pilot):,}; full benchmark rows: {len(full):,}.",
        "", "## Results", "",
        markdown_table(primary.groupby(["workload", "model_pair", "variant"], dropna=False).agg(
            repeats=("repetition", "nunique"), mean_tokens_s=("target_tokens_per_second", "mean"),
            mean_speedup_fixed=("speedup_vs_fixed_k_sd", "mean"), exact=("all_outputs_exact", "all"),
        ).reset_index()) if len(primary) else "No benchmark results.",
        "", "GPU allocation and software versions are recorded in `method_config.json`; utilization samples are in `gpu3_utilization.csv`.",
    ]
    (OUT / "method_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cpu_analysis_report(stage: str) -> None:
    try:
        from scripts.summarize_morphology_guard import generate_report
        generate_report(stage)
    except Exception as exc:
        # The benchmark checkpoints/results remain complete and resumable; the
        # report can be regenerated independently with the same CPU-only CLI.
        print(f"CPU report generation failed for stage={stage}: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
