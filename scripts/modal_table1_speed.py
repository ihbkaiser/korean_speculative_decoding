#!/usr/bin/env python3
"""Bounded real-model B200 throughput tuning against the existing Table 1 path."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
# Freeze only sources, not live logs/artifacts. Previous whole-repo uploads
# failed when Tee-Object changed a log during Modal's upload.
if modal.is_local():
    sys.path.insert(0, str(ROOT))
    from src.table1_smoke import snapshot_sources
    SOURCE = Path(tempfile.mkdtemp(prefix="table1-speed-source-"))
    snapshot_sources(ROOT, SOURCE)
else:
    SOURCE = ROOT

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5,<3", "transformers>=4.51,<6", "mistral-common>=1.8.6",
        "datasets>=3.3,<5", "huggingface-hub>=0.30,<2", "kiwipiepy>=0.21,<1",
        "pyarrow>=19", "pandas>=2.2", "PyYAML>=6", "tabulate>=0.9", "pytest>=8",
    )
    # Opt-in allocator experiment; default and production launchers are unchanged.
    .env({"PYTORCH_ALLOC_CONF": os.environ.get("TABLE1_TRIAL_ALLOC_CONF", "")})
    .add_local_dir(str(SOURCE), remote_path="/root/speedrepo")
)
cache = modal.Volume.from_name("korean-speculative-decoding-hf-cache")
artifacts = modal.Volume.from_name("korean-speculative-decoding-table1-artifacts")
app = modal.App("table1-b200-throughput", image=image)
VOLUMES = {"/root/cache": cache, "/root/artifacts": artifacts}
REPO = Path("/root/speedrepo")
WORKSPACE = Path("/root/artifacts/workspace")


def _setup():
    sys.path.insert(0, str(REPO))
    os.environ.update(HF_HOME="/root/cache/huggingface", TOKENIZERS_PARALLELISM="false")


@app.function(volumes=VOLUMES, cpu=2, memory=4096, timeout=300)
def inspect_remote():
    _setup()
    metadata = WORKSPACE / "metadata/model_revisions.json"
    result = {
        "prompt_pool": str(WORKSPACE / "data/prompts_40k.parquet"),
        "prompt_pool_exists": (WORKSPACE / "data/prompts_40k.parquet").exists(),
        "model_metadata_exists": metadata.exists(),
        "model_directories": [str(p) for p in Path("/root/cache/models").glob("*/*") if p.is_dir()],
    }
    if metadata.exists():
        value = json.loads(metadata.read_text())
        result["pairs"] = {key: {side: item.get(side, {}) for side in ("draft", "target")}
                           for key, item in value.get("pairs", {}).items()}
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from scripts.table1_pipeline import load_config
    cfg = load_config(REPO / "configs/table1_pipeline.yaml")
    result["m1_tokenizers"] = {}
    for side, path in _models("M1", cfg).items():
        info = {"files": [p.name for p in Path(path).iterdir() if p.suffix == ".json"]}
        for mode, cls in (("auto", AutoTokenizer), ("explicit_fast", PreTrainedTokenizerFast)):
            try:
                tok = cls.from_pretrained(path, use_fast=True, local_files_only=True,
                                           **({"fix_mistral_regex": True} if mode == "explicit_fast" else {}))
                info[mode] = {"class": type(tok).__name__, "is_fast": getattr(tok, "is_fast", False),
                              "probe_ids": tok.encode("한국어 형태소 분석", add_special_tokens=False)}
            except Exception as exc:
                info[mode] = {"error": str(exc)}
        result["m1_tokenizers"][side] = info
    print("TABLE1_INSPECT_JSON=" + json.dumps(result), flush=True)
    return result


def _models(pair_id, config):
    pair = config["pairs"][pair_id]
    paths = {}
    for side in ("draft", "target"):
        revision = Path(config["model_paths"][pair_id][side]).name
        candidate = Path("/root/cache/models") / pair[side].replace("/", "--") / revision
        if not (candidate / "config.json").exists():
            raise FileNotFoundError(f"Pinned model snapshot missing: {candidate}")
        paths[side] = str(candidate)
    return paths


@app.function(volumes=VOLUMES, gpu="B200", cpu=16, memory=131072, timeout=1200)
def benchmark_remote(pair_id="Q1", prompts=512, baseline_prompts=0,
                     max_new_tokens=128, batch_sizes=(128, 256),
                     dtype="bfloat16", attention="sdpa", label="sweep", diagnostic_prompts=0):
    _setup()
    import gc
    import hashlib
    import importlib.metadata
    import torch
    from transformers import AutoTokenizer
    from tqdm.auto import tqdm
    from scripts.table1_pipeline import load_config
    from src.batched_speculative import speculative_greedy_microbatch
    from src.data import encode_prompt
    from src.models import load_models, load_table1_tokenizer
    from src.speculative_decoding import greedy_generate, speculative_greedy_cached
    from src.table1_runner import load_pair_records, generate_reference_ids
    from src.table1_smoke import validate_trial_request

    validate_trial_request("benchmark", prompts, max_new_tokens, batch_sizes)

    cfg = load_config(REPO / "configs/table1_fast_b200.yaml")
    pair = cfg["pairs"][pair_id]
    paths = _models(pair_id, cfg)
    records = load_pair_records(WORKSPACE / cfg["paths"]["prompts"], pair)[:prompts]
    validate_trial_request("benchmark", len(records), max_new_tokens, batch_sizes)
    tokenizer = load_table1_tokenizer(paths["target"], local_files_only=True,
                                      backend=pair.get("tokenizer_backend", "auto"),
                                      fix_mistral_regex=pair.get("fix_mistral_regex", False))
    prompt_ids = [encode_prompt(tokenizer, row["text"], 128) for row in records]
    eos = tokenizer.eos_token_id
    report = {
        "pair": pair_id, "prompts": len(records), "max_new_tokens": max_new_tokens,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"), "model_paths": paths,
        "prompt_ids_sha256": hashlib.sha256("\n".join(row["doc_id"] for row in records).encode()).hexdigest(),
        "dtype": dtype, "attention_backend": attention, "use_remove_padding": False,
        "gpu_total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF", ""),
        "scalar_fallback": False, "full_run_started": False,
        "measurements": [],
    }
    destination = Path("/root/artifacts/benchmarks") / f"table1_speed_{pair_id.lower()}_{label}.json"

    def save():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        artifacts.commit()

    def emit(row):
        report["measurements"].append(row)
        print("TABLE1_SPEED_METRIC=" + json.dumps(row), flush=True)
        save()

    # A bounded sample of the ACTUAL current production path: FP32/eager,
    # independent batch references and full singleton reruns on disagreement.
    if baseline_prompts:
        old_draft, old_target = load_models(paths["draft"], paths["target"],
                                            dtype="float32", attention_backend="eager")
        cfg_run = {**cfg["inference"], "max_new_tokens": max_new_tokens}
        n = min(baseline_prompts, len(records))
        speculative_greedy_cached(old_draft, old_target, prompt_ids[0], 4, eos, 4,
                                 batch_target_verification=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        refs = generate_reference_ids(records[:n], tokenizer=tokenizer, target_model=old_target,
                                      pair_spec=pair, config=cfg_run)
        reference_seconds = time.perf_counter() - started
        fallbacks, generated = 0, 0
        for record, ids in tqdm(list(zip(records[:n], prompt_ids[:n])),
                                desc="Baseline SD", unit="prompt", mininterval=1):
            out, _ = speculative_greedy_cached(old_draft, old_target, ids, max_new_tokens, eos, 4,
                                               batch_target_verification=True)
            if out != refs[record["doc_id"]]:
                scalar = greedy_generate(old_target, ids, max_new_tokens, eos)
                out, _ = speculative_greedy_cached(old_draft, old_target, ids, max_new_tokens, eos, 4)
                if out != scalar:
                    raise AssertionError("Baseline scalar repair failed")
                fallbacks += 1
            generated += len(out)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        emit({"mode": "current_fp32_eager_reference_fallback", "batch_size": 1,
              "prompts": n, "seconds": elapsed, "reference_seconds": reference_seconds,
              "prompts_per_second": n / elapsed, "tokens_per_second": generated / elapsed,
              "fallbacks": fallbacks, "estimated_20000_seconds": 20000 * elapsed / n,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3})
        del old_draft, old_target
        gc.collect()
        torch.cuda.empty_cache()

    draft, target = load_models(paths["draft"], paths["target"], dtype=dtype, attention_backend=attention)
    # Independent target check is diagnostic, on the same numeric path.
    diagnostic_n = min(diagnostic_prompts, len(prompt_ids))
    scalar_refs = [greedy_generate(target, ids, max_new_tokens, eos) for ids in prompt_ids[:diagnostic_n]]
    for size in batch_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        phase = "warmup"
        try:
            print(f"TABLE1_SPEED_PHASE={pair_id} B={size} warmup", flush=True)
            warmup_started = time.perf_counter()
            speculative_greedy_microbatch(draft, target, prompt_ids[:min(size, len(prompt_ids))], 4, eos, 4)
            torch.cuda.synchronize()
            warmup_seconds = time.perf_counter() - warmup_started
            warmup_peak = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()
            phase = "measurement"
            print(f"TABLE1_SPEED_PHASE={pair_id} B={size} measurement", flush=True)
            started = time.perf_counter()
            outputs, block_times, event_count, diagnostic_mismatches = [], [], 0, []
            with tqdm(total=len(prompt_ids), desc=f"{pair_id} B={size}", unit="prompt",
                      mininterval=1) as bar:
                for offset in range(0, len(prompt_ids), size):
                    tick = time.perf_counter()
                    out, events = speculative_greedy_microbatch(draft, target, prompt_ids[offset:offset + size],
                                                               max_new_tokens, eos, 4)
                    torch.cuda.synchronize()
                    block_times.append(time.perf_counter() - tick)
                    outputs.extend(out)
                    event_count += sum(map(len, events))
                    bar.update(len(out))
            elapsed = time.perf_counter() - started
            for index in range(diagnostic_n):
                if outputs[index] != scalar_refs[index]:
                    diagnostic_mismatches.append(index)
            generated = sum(map(len, outputs))
            emit({"mode": "microbatched", "batch_size": size, "prompts": len(prompt_ids),
                  "seconds": elapsed, "block_seconds": block_times,
                  "prompts_per_second": len(prompt_ids) / elapsed,
                  "tokens_per_second": generated / elapsed, "generated_tokens": generated,
                  "proposal_events": event_count, "estimated_20000_seconds": 20000 * elapsed / len(prompt_ids),
                  "scalar_diagnostic_prompts": diagnostic_n, "scalar_mismatch_indices": diagnostic_mismatches,
                  "actual_batch_sizes": [len(prompt_ids[offset:offset + size]) for offset in range(0, len(prompt_ids), size)],
                  "warmup_seconds": warmup_seconds, "warmup_peak_allocated_gib": warmup_peak,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3})
        except torch.cuda.OutOfMemoryError as exc:
            emit({"mode": "microbatched", "batch_size": size, "status": "oom", "phase": phase,
                  "error": str(exc), "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3})
            gc.collect()
            torch.cuda.empty_cache()
    successful = [row for row in report["measurements"] if row["mode"] == "microbatched" and "seconds" in row]
    if not successful:
        raise RuntimeError("No successful microbatch measurement")
    report["best"] = max(successful, key=lambda row: row["prompts_per_second"])
    save()
    print("TABLE1_SPEED_SUMMARY=" + json.dumps(report), flush=True)
    return report


@app.function(volumes=VOLUMES, gpu="B200", cpu=16, memory=131072, timeout=1200)
def pilot_remote(pair_id="Q1", prompts=256, batch_size=256, max_new_tokens=128,
                 dtype="bfloat16", attention="sdpa", label="pilot"):
    """Bounded production-path trial including checkpoint and parquet writes."""
    _setup()
    import importlib.metadata
    import pandas as pd
    import torch
    from scripts.table1_pipeline import load_config
    from src.table1_runner import load_pair_records, run_sd_shard
    from src.table1_morphology import align_pair_shard, build_table1_outputs
    from src.table1_smoke import validate_fast_shard

    if not 1 <= prompts <= 512 or not 1 <= max_new_tokens <= 128:
        raise ValueError("Pilot is limited to 512 prompts and 128 output tokens; no full run")
    cfg = load_config(REPO / "configs/table1_fast_b200.yaml")
    original = load_config(REPO / "configs/table1_pipeline.yaml")
    records = load_pair_records(WORKSPACE / cfg["paths"]["prompts"], cfg["pairs"][pair_id])[:prompts]
    paths = _models(pair_id, original)
    root = Path("/root/artifacts/benchmarks") / f"pilot_{pair_id.lower()}_{label}"
    if root.exists():
        raise FileExistsError(f"Use a fresh pilot label; existing artifacts: {root}")
    (root / "data").mkdir(parents=True)
    pd.DataFrame(records).to_parquet(root / "data/prompts.parquet", index=False)
    pair = {**cfg["pairs"][pair_id], "prompt_count": len(records), "dtype": dtype,
            "attention_backend": attention, "sd_batch_size": batch_size}
    config = {**cfg["inference"], "max_new_tokens": max_new_tokens,
              "pairs": {pair_id: pair},
              "paths": {**cfg["paths"], "prompts": "data/prompts.parquet", "runs": "runs"}}
    metadata = {side: {"id": pair[side], "local_path": paths[side]} for side in ("draft", "target")}
    started = time.perf_counter()
    result = run_sd_shard(root=root, pair_id=pair_id, pair_spec=pair, model_meta=metadata,
                          config=config, progress_log_every=batch_size)
    sd_total_seconds = time.perf_counter() - started
    shard = Path(result["path"])
    references = pd.read_parquet(shard / "references.parquet")
    generated = sum(map(len, references.target_continuation_token_ids))
    integrity = validate_fast_shard(shard, expected_prompts=len(records), max_new_tokens=max_new_tokens)
    artifacts.commit()
    # Alignment is measured separately; its time includes tokenizer/Kiwi setup.
    started = time.perf_counter()
    alignment = align_pair_shard(root=root, pair_id=pair_id, config=config,
                                 target_model_path=paths["target"])
    alignment_seconds = time.perf_counter() - started
    table_config = {**cfg, "pairs": {pair_id: pair},
                    "paths": {**config["paths"], "results": "results_smoke"}}
    started = time.perf_counter()
    table_result = build_table1_outputs(root=root, config=table_config)
    if table_result["status"] != "COMPLETE":
        raise AssertionError(f"Smoke table build incomplete: {table_result}")
    aggregation_seconds = time.perf_counter() - started
    try:
        flash_version = importlib.metadata.version("flash-attn")
    except importlib.metadata.PackageNotFoundError:
        flash_version = None
    report = {
        "pair": pair_id, "prompts": len(records), "batch_size": batch_size,
        "max_new_tokens": max_new_tokens, "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "compute_capability": list(torch.cuda.get_device_capability()), "flash_attn": flash_version,
        "use_remove_padding": False,
        "dtype": dtype, "attention_backend": attention,
        "model_paths": paths, "sd_including_model_setup_seconds": sd_total_seconds,
        "sd_generation_checkpoint_parquet_seconds": result["elapsed_seconds"],
        "sd_prompts_per_second": len(records) / result["elapsed_seconds"],
        "estimated_20000_sd_seconds": 20000 * result["elapsed_seconds"] / len(records),
        "alignment_including_setup_seconds": alignment_seconds,
        "aggregation_seconds": aggregation_seconds,
        "table_status": "COMPLETE_SMOKE_ONLY", "table_results_dir": table_result["results_dir"],
        "integrity": integrity, "status": "passed", "scalar_fallback_prompt_count": 0,
        "alignment_rows": alignment["aligned_rows"], "generated_tokens": generated,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "progress_bytes": (shard / "progress.jsonl").stat().st_size,
        "scalar_parity_validated": False, "full_run_started": False,
        "checkpoint_path": str(shard),
    }
    (root / "pilot_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    artifacts.commit()
    print("TABLE1_PILOT_SUMMARY=" + json.dumps(report), flush=True)
    return report


@app.function(volumes=VOLUMES, cpu=8, memory=32768, timeout=600)
def finish_smoke_remote(pair_id, label, finish_label=""):
    """Recheck CPU stages using completed B200 SD; never regenerate tokens."""
    _setup()
    import importlib.metadata
    from scripts.table1_pipeline import load_config
    from src.table1_morphology import align_pair_shard, build_table1_outputs
    from src.table1_smoke import validate_fast_shard

    source = Path("/root/artifacts/benchmarks") / f"pilot_{pair_id.lower()}_{label}"
    relative_shard = Path("runs") / pair_id / "shards/shard-00000-of-00001"
    source_shard = source / relative_shard
    metadata = json.loads((source_shard / "run_metadata.json").read_text())
    count, max_tokens = metadata["prompt_count"], metadata["max_new_tokens"]
    if not 1 <= count <= 512 or not 1 <= max_tokens <= 128:
        raise ValueError("Only bounded smoke checkpoints can be postprocessed")
    integrity = validate_fast_shard(source_shard, expected_prompts=count, max_new_tokens=max_tokens)
    root = source.parent / f"finished_{pair_id.lower()}_{finish_label or label}"
    (root / relative_shard).mkdir(parents=True, exist_ok=False)
    (root / "data").mkdir()
    shutil.copy2(source / "data/prompts.parquet", root / "data/prompts.parquet")
    for name in ("references.parquet", "sd_events.parquet", "run_metadata.json", "COMPLETE"):
        shutil.copy2(source_shard / name, root / relative_shard / name)
    cfg = load_config(REPO / "configs/table1_fast_b200.yaml")
    pair = {**cfg["pairs"][pair_id], "prompt_count": count}
    config = {**cfg, "pairs": {pair_id: pair},
              "paths": {**cfg["paths"], "prompts": "data/prompts.parquet",
                        "runs": "runs", "results": "results_smoke"}}
    started = time.perf_counter()
    alignment = align_pair_shard(root=root, config=config, pair_id=pair_id,
                                 target_model_path=metadata["target_model"])
    alignment_seconds = time.perf_counter() - started
    started = time.perf_counter()
    table = build_table1_outputs(root=root, config=config)
    aggregation_seconds = time.perf_counter() - started
    if table["status"] != "COMPLETE":
        raise AssertionError(f"Smoke table incomplete: {table}")
    sd_seconds = metadata["elapsed_seconds"]
    report = {
        "pair": pair_id, "status": "passed", "prompts": count,
        "batch_size": metadata["sd_batch_size"], "max_new_tokens": max_tokens,
        "gpu": "NVIDIA B200 (completed source SD job)",
        "transformers": importlib.metadata.version("transformers"),
        "dtype": metadata["dtype"], "attention_backend": metadata["attention_backend"],
        "use_remove_padding": False, "scalar_fallback_prompt_count": 0,
        "scalar_parity_validated": False, "full_run_started": False,
        "sd_generation_checkpoint_parquet_seconds": sd_seconds,
        "sd_prompts_per_second": count / sd_seconds,
        "estimated_20000_sd_seconds": 20000 * sd_seconds / count,
        "alignment_including_setup_seconds": alignment_seconds,
        "aggregation_seconds": aggregation_seconds, "alignment_rows": alignment["aligned_rows"],
        "generated_tokens": integrity["generated_tokens"], "integrity": integrity,
        "table_status": "COMPLETE_SMOKE_ONLY", "table_results_dir": table["results_dir"],
        "checkpoint_path": str(source_shard), "postprocessing_root": str(root),
        "generation_rerun": False,
    }
    (root / "pilot_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    artifacts.commit()
    print("TABLE1_SMOKE_FINISH_SUMMARY=" + json.dumps(report), flush=True)
    return report


@app.function(volumes=VOLUMES, gpu="B200", cpu=16, memory=131072, timeout=1200)
def launcher_smoke_remote(prompts=256, batch_size=256, max_new_tokens=128, label="launcher_smoke"):
    """Run the real shell launcher and resume on one B200, with bounded inputs."""
    _setup()
    import hashlib
    import importlib.metadata
    import re
    import subprocess
    import pandas as pd
    import torch
    import yaml
    from scripts.table1_pipeline import load_config
    from src.table1_runner import load_pair_records
    from src.table1_smoke import make_launcher_smoke_config, validate_fast_shard

    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Smoke label must be a simple directory name")
    cfg = load_config(REPO / "configs/table1_fast_b200.yaml")
    model_paths = {pair: _models(pair, cfg) for pair in cfg["pairs"]}
    bounded = make_launcher_smoke_config(cfg, prompts, batch_size, max_new_tokens, model_paths)
    root = Path("/root/artifacts/benchmarks") / f"launcher_{label}"
    if root.exists():
        raise FileExistsError(f"Use a new label; existing smoke artifacts: {root}")
    staged_repo, output = root / "repo", root / "output with spaces"
    shutil.copytree(REPO, staged_repo, ignore=shutil.ignore_patterns("__pycache__"))
    (staged_repo / "data").mkdir(exist_ok=True)
    (staged_repo / "metadata").mkdir(exist_ok=True)
    records = load_pair_records(WORKSPACE / cfg["paths"]["prompts"], cfg["pairs"]["Q1"])[:prompts]
    if len(records) != prompts:
        raise AssertionError("Insufficient frozen smoke prompts")
    pd.DataFrame(records).to_parquet(staged_repo / bounded["paths"]["prompts"], index=False)
    shutil.copy2(WORKSPACE / cfg["paths"]["dataset_metadata"],
                 staged_repo / bounded["paths"]["dataset_metadata"])
    # Only the isolated fixture changes prompt counts and Modal snapshot paths.
    (staged_repo / "configs/table1_fast_b200.yaml").write_text(yaml.safe_dump(bounded))
    env = {**os.environ, "TABLE1_CONFIG": str(staged_repo / "configs/table1_pipeline.yaml"),
           "PYTHON_BIN": sys.executable, "NUM_SHARDS": "1", "SD_BATCH_SIZE": str(batch_size),
           "TABLE1_LOG_TAG": "launcher-smoke", "DEVICE": "cuda", "ALIGN_DEVICE": "cpu"}
    command = ["bash", "run_table1.sh", "all", "--output-dir", str(output)]

    def launch(name):
        print("TABLE1_LAUNCHER_PHASE=" + name, flush=True)
        started = time.perf_counter()
        try:
            with (root / f"{name}.log").open("w") as log:
                with subprocess.Popen(command, cwd=staged_repo, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    code = process.wait()
            if code:
                raise RuntimeError(f"Launcher {name} exited with code {code}; see persistent logs")
        finally:
            artifacts.commit()
        return time.perf_counter() - started

    elapsed = launch("initial")
    logs = (root / "initial.log").read_text()
    if "Target references" in logs or "scalar parity fallback" in logs:
        raise AssertionError("Launcher entered a forbidden reference/fallback path")
    pairs, hashes = {}, {}
    for pair in cfg["pairs"]:
        shard = output / "runs" / pair / "shards/shard-00000-of-00001"
        integrity = validate_fast_shard(shard, expected_prompts=prompts, max_new_tokens=max_new_tokens)
        metadata = json.loads((shard / "run_metadata.json").read_text())
        manifest = json.loads((output / f"metadata/table1_main_{pair}.json").read_text())
        assert manifest["require_microbatched"] is True
        assert manifest["resolved_config"]["pairs"][pair]["decoder"] == "microbatched"
        assert (shard / "ALIGNMENT_COMPLETE").exists()
        assert f"{pair} SD (B={batch_size})" in logs
        hashes[pair] = hashlib.sha256((shard / "progress.jsonl").read_bytes()).hexdigest()
        pairs[pair] = {"status": "passed", "integrity": integrity,
                       "sd_seconds": metadata["elapsed_seconds"],
                       "sd_prompts_per_second": prompts / metadata["elapsed_seconds"]}
    table = json.loads((output / "results/table1_build_metadata.json").read_text())
    assert table["status"] == "COMPLETE"
    resume_seconds = launch("resume")
    for pair in pairs:
        progress = output / "runs" / pair / "shards/shard-00000-of-00001/progress.jsonl"
        assert hashlib.sha256(progress.read_bytes()).hexdigest() == hashes[pair], "resume rewrote prompts"
    assert not (staged_repo / "runs").exists(), "output escaped custom output directory"
    report = {"status": "passed", "stage": "launcher-smoke", "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "transformers": importlib.metadata.version("transformers"),
              "compute_capability": list(torch.cuda.get_device_capability()),
              "prompts_per_pair": prompts, "batch_size": batch_size, "max_new_tokens": max_new_tokens,
              "dtype": "bfloat16", "attention_backend": "sdpa", "pairs": pairs,
              "initial_launcher_seconds": elapsed, "resume_seconds": resume_seconds,
              "resume_progress_unchanged": True, "table_status": table["status"],
              "strict_environment_override_ignored": True, "scalar_fallback_prompt_count": 0,
              "scalar_parity_validated": False, "full_run_started": False,
              "source_sha256": {name: hashlib.sha256((staged_repo / name).read_bytes()).hexdigest()
                                for name in ("run_table1.sh", "scripts/run_company_table1_fast.sh",
                                             "scripts/table1_pipeline.py", "src/table1_runner.py")},
              "artifact_root": str(root), "output_dir": str(output)}
    (root / "launcher_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    artifacts.commit()
    print("TABLE1_LAUNCHER_SMOKE_SUMMARY=" + json.dumps(report), flush=True)
    return report


@app.local_entrypoint()
def main(stage: str = "inspect", pair: str = "Q1", prompts: int = 256,
         baseline_prompts: int = 0, max_new_tokens: int = 128,
         batch_sizes: str = "256", dtype: str = "bfloat16",
         attention: str = "sdpa", label: str = "sweep",
         output: str = "profile_output/modal_b200_speed.json", diagnostic_prompts: int = 0,
         finish_label: str = ""):
    sys.path.insert(0, str(ROOT))
    from src.table1_smoke import validate_trial_request

    sizes = tuple(int(b) for b in batch_sizes.split(","))
    validate_trial_request(stage, prompts, max_new_tokens, sizes)
    if not sizes or min(sizes) < 1 or baseline_prompts < 0 or not 0 <= diagnostic_prompts <= 8:
        raise ValueError("Batch sizes must be positive and baseline-prompts non-negative")
    if stage == "inspect":
        result = inspect_remote.remote()
    elif stage == "launcher-smoke":
        if len(sizes) != 1:
            raise ValueError("launcher-smoke requires exactly one batch size")
        result = launcher_smoke_remote.remote(prompts, sizes[0], max_new_tokens, label)
    elif stage in {"smoke-all", "smoke-finish"}:
        if len(sizes) != 1 or prompts < sizes[0]:
            raise ValueError("smoke-all requires one batch size and at least one full batch of prompts")
        pending = [(p, finish_smoke_remote.spawn(p, label, finish_label) if stage == "smoke-finish" else
                    pilot_remote.spawn(p, prompts, sizes[0], max_new_tokens, dtype, attention, label))
                   for p in ("Q1", "Q2", "Q3", "M1", "G1")]
        result = {"stage": stage, "prompts_per_pair": prompts, "batch_size": sizes[0],
                  "scalar_fallback": False, "full_run_started": False, "pairs": {}}
        for p, call in pending:
            try:
                result["pairs"][p] = call.get()
            except Exception as exc:
                result["pairs"][p] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        result["status"] = "passed" if all(v["status"] == "passed" for v in result["pairs"].values()) else "failed"
        print("TABLE1_SMOKE_ALL_SUMMARY=" + json.dumps(result), flush=True)
    elif stage == "pilot":
        if len(sizes) != 1:
            raise ValueError("pilot requires exactly one batch size")
        result = pilot_remote.remote(pair, prompts, sizes[0], max_new_tokens, dtype, attention, label)
    else:
        result = benchmark_remote.remote(pair, prompts, baseline_prompts, max_new_tokens,
                                         sizes, dtype, attention, label, diagnostic_prompts)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {destination}")
    if result.get("stage") in {"smoke-all", "smoke-finish"} and result["status"] != "passed":
        raise RuntimeError("One or more smoke pairs failed; inspect the saved summary, no fallback was used")
