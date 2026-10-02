#!/usr/bin/env python3
"""Collect additional greedy speculative-decoding traces for Korean prompts.

Uses the pinned E2 model revisions and decoder, one model pair at a time. It
records proposal/target entropy at each actual verification decision and
checks independent target-greedy parity on the first 32 prompts per workload.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e2_model_pairs import MODELS, PAIRS, greedy_generate_single, speculative_greedy_fast


OUT = ROOT / "runs/proposal_boundary_data_expansion"
MODEL_PARITY_PROMPTS = 32
CHECKPOINT_EVERY = 10
MAX_NEW_TOKENS = 128
SPECULATIVE_K = 4
SAMPLE_SEED = 20261001


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_pair(torch_module: Any, pair_name: str) -> dict[str, Any]:
    pair = PAIRS[pair_name]
    loaded = {}
    for size in (pair["draft"], pair["target"]):
        spec = MODELS[size]
        model = AutoModelForCausalLM.from_pretrained(
            spec["id"], revision=spec["revision"], local_files_only=True,
            dtype=torch_module.float16, low_cpu_mem_usage=True, attn_implementation="sdpa",
        )
        model.to("cuda:0")
        model.eval()
        loaded[size] = model
        print(f"Loaded {size} {spec['id']}@{spec['revision']} on cuda:0", flush=True)
    return loaded


def load_prompt_groups() -> dict[str, list[dict[str, Any]]]:
    groups = {"WIKIPEDIA": [], "FLORES": []}
    path = OUT / "prompts.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Run scripts/prepare_boundary_expansion_data.py first: {path}")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                groups[str(row["workload"])].append(row)
    for workload, rows in groups.items():
        if len(rows) != (1000 if workload == "WIKIPEDIA" else 997):
            raise RuntimeError(f"Unexpected {workload} prompt count: {len(rows)}")
        if len({int(r["prompt_id"]) for r in rows}) != len(rows):
            raise RuntimeError(f"Duplicate prompt IDs in {workload}")
    return groups


def read_completed(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                done.add(int(row["prompt_id"]))
    return done


def select_prompts(workload: str, all_prompts: list[dict[str, Any]], max_prompts: int | None) -> list[dict[str, Any]]:
    """Keep the parity pilot and hash-sample the rest across the full source order."""
    if max_prompts is None or max_prompts >= len(all_prompts):
        return all_prompts
    if max_prompts <= 0:
        raise ValueError("--max-prompts must be positive")
    pilot_count = min(MODEL_PARITY_PROMPTS, max_prompts, len(all_prompts))
    selected = list(all_prompts[:pilot_count])
    remaining_count = max_prompts - pilot_count
    if remaining_count:
        candidates = all_prompts[pilot_count:]
        candidates = sorted(
            candidates,
            key=lambda row: hashlib.sha256(
                f"{SAMPLE_SEED}|{workload}|{row['prompt_hash']}".encode("utf-8")
            ).digest(),
        )
        selected.extend(candidates[:remaining_count])
    return sorted(selected, key=lambda row: int(row["source_ordinal"]))


def collect_pair(pair_name: str, groups: dict[str, list[dict[str, Any]]], max_prompts: int | None) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("This instance exposes one RTX A4000 as physical GPU 0; launch with CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible GPU as cuda:0; found {torch.cuda.device_count()}")
    torch.cuda.set_device(0)
    if "A4000" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Expected this instance's RTX A4000; found {torch.cuda.get_device_name(0)}")
    torch.manual_seed(3090)
    torch.cuda.manual_seed_all(3090)
    torch.backends.cuda.matmul.allow_tf32 = True

    models = load_pair(torch, pair_name)
    pair = PAIRS[pair_name]
    draft = models[pair["draft"]]
    target = models[pair["target"]]
    eos_token_id = 151643
    for workload, all_prompts in groups.items():
        prompts = select_prompts(workload, all_prompts, max_prompts)
        trace_path = OUT / "traces" / f"{workload.lower()}_{pair_name.lower()}.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        completed = read_completed(trace_path)
        started = time.perf_counter()
        completed_this_run = 0
        with trace_path.open("a", encoding="utf-8") as handle:
            for index, prompt in enumerate(prompts, start=1):
                prompt_id = int(prompt["prompt_id"])
                if prompt_id in completed:
                    continue
                prompt_ids = [int(x) for x in prompt["input_ids"]]
                parity_checked = index <= MODEL_PARITY_PROMPTS
                reference_ids = None
                if parity_checked:
                    reference_ids = greedy_generate_single(target, prompt_ids, MAX_NEW_TOKENS, eos_token_id)
                runtime_stats: dict[str, Any] = {}
                sd_ids, events = speculative_greedy_fast(
                    draft, target, prompt_ids, MAX_NEW_TOKENS, eos_token_id, SPECULATIVE_K,
                    prompt_id, runtime_stats=runtime_stats, record_entropies=True,
                )
                exact_match = None if reference_ids is None else [int(x) for x in sd_ids] == reference_ids
                if exact_match is False:
                    failed = {
                        "prompt_id": prompt_id, "workload": workload, "pair": pair_name,
                        "prompt_hash": prompt["prompt_hash"], "reference_token_ids": reference_ids,
                        "sd_token_ids": [int(x) for x in sd_ids], "exact_match": False,
                    }
                    atomic_json(trace_path.with_suffix(".parity_failure.json"), failed)
                    raise RuntimeError(f"Target greedy parity failed for {workload}/{pair_name}/prompt_id={prompt_id}")
                row = {
                    "workload": workload, "pair": pair_name, "prompt_id": prompt_id,
                    "source_split": prompt["source_split"], "source_ordinal": int(prompt["source_ordinal"]),
                    "source_index": prompt.get("source_index"), "prompt_hash": prompt["prompt_hash"],
                    "input_ids": prompt_ids, "sd_token_ids": [int(x) for x in sd_ids],
                    "reference_token_ids": reference_ids,
                    "target_parity_checked": parity_checked, "target_parity_exact": exact_match,
                    "events": events, "runtime_stats": runtime_stats,
                }
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                completed.add(prompt_id)
                completed_this_run += 1
                if completed_this_run % CHECKPOINT_EVERY == 0:
                    print(f"{pair_name}/{workload}: committed {len(completed)}/{len(prompts)} prompts", flush=True)
        rows = []
        with trace_path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        expected = {int(r["prompt_id"]) for r in prompts}
        if {int(r["prompt_id"]) for r in rows} != expected:
            raise RuntimeError(f"Incomplete trace coverage for {pair_name}/{workload}")
        metadata = {
            "workload": workload, "pair": pair_name, "prompt_count": len(rows),
            "target_parity_checked": sum(bool(r["target_parity_checked"]) for r in rows),
            "target_parity_failures": sum(r["target_parity_exact"] is False for r in rows),
            "speculative_outputs_match_independent_target_greedy_on_audited_prompts": all(
                r["target_parity_exact"] is True for r in rows if r["target_parity_checked"]
            ),
            "valid_proposal_events": sum(sum(bool(e["sd_valid"]) for e in r["events"]) for r in rows),
            "first_rejection_events": sum(sum(bool(e["is_first_rejection"]) for e in r["events"]) for r in rows),
            "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            "elapsed_seconds_this_run": time.perf_counter() - started,
            "model_pair": {"draft": MODELS[pair["draft"]], "target": MODELS[pair["target"]]},
            "max_new_tokens": MAX_NEW_TOKENS, "speculative_k": SPECULATIVE_K,
            "requested_prompt_cap_per_workload": max_prompts,
            "sample_seed": SAMPLE_SEED,
            "dtype": "float16", "attention_backend": "sdpa", "physical_gpu": 0,
            "gpu_name": torch.cuda.get_device_name(0),
            "peak_allocated_vram_gb": torch.cuda.max_memory_allocated(0) / 1024**3,
            "prompt_manifest_sha256": hashlib.sha256((OUT / "prompt_manifest.csv").read_bytes()).hexdigest(),
        }
        atomic_json(trace_path.with_suffix(".metadata.json"), metadata)
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    del models
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=["P1", "P2", "all"], default="all")
    parser.add_argument("--max-prompts", type=int, help="per-workload sample size; keeps the first 32 parity prompts and hash-samples the remainder")
    args = parser.parse_args()
    groups = load_prompt_groups()
    pairs = ("P1", "P2") if args.pair == "all" else (args.pair,)
    for pair_name in pairs:
        collect_pair(pair_name, groups, args.max_prompts)
    selection = {
        "sample_seed": SAMPLE_SEED,
        "sampling": (
            "all source rows in source order" if args.max_prompts is None
            else "retain first 32 parity prompts; SHA-256 sample from remaining source rows; restore source order"
        ),
        "max_prompts_per_workload": args.max_prompts,
        "selected_prompt_ids": {
            workload: [int(row["prompt_id"]) for row in select_prompts(workload, rows, args.max_prompts)]
            for workload, rows in groups.items()
        },
    }
    atomic_json(OUT / "selected_prompt_sample.json", selection)


if __name__ == "__main__":
    main()
