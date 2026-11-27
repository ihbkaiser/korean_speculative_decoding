#!/usr/bin/env python3
"""Measure synchronized wall and CUDA kernel time for cached Qwen forwards."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from huggingface_hub.utils import get_token

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.rebuttal.streamed_model_loader import _load_streamed_model
from src.naacl_experiment import load_config
from src.rebuttal.runtime import write_gpu_metadata


def _prompt() -> tuple[list[int], list[int], int]:
    directory = ROOT / "results/naacl_experiments/continuations/qwen4/all"
    files = sorted(directory.glob("shard-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No stored Q2 continuations under {directory}")
    frame = pd.concat(
        [pd.read_parquet(path, columns=["pool_index", "prompt_ids", "continuation_ids"])
         for path in files], ignore_index=True)
    frame = frame[(frame.pool_index % 2 == 1) & frame.prompt_ids.map(len).ge(128)]
    if frame.empty:
        raise RuntimeError("No eligible 128-token Q2 test prompt")
    row = frame.sort_values("pool_index", kind="stable").iloc[0]
    return list(map(int, row.prompt_ids[:128])), list(map(int, row.continuation_ids)), int(row.pool_index)


def _load(config: dict, key: str, api: HfApi, token: str):
    model, path, mode = _load_streamed_model(config, key, api, token)
    model.config.use_cache = True
    model.eval()
    return model, path, mode


def _sync() -> None:
    torch.cuda.synchronize()


def _self_device_time_us(event) -> float:
    if not str(getattr(event, "device_type", "")).endswith("CUDA"):
        return 0.0
    value = getattr(event, "self_device_time_total", None)
    if value is None:
        value = event.self_cuda_time_total
    return float(value)


@torch.inference_mode()
def _cache(model, input_ids):
    return model(input_ids=input_ids, use_cache=True).past_key_values


@torch.inference_mode()
def _call(model, base_past, ids):
    # The DynamicCache is mutated by forward. Crop its appended suffix between
    # calls so the timed region contains only the cached model forward, not a
    # deep copy of every prompt key/value tensor.
    extra_tokens = int(base_past.get_seq_length()) - 128
    if extra_tokens > 0:
        base_past.crop(-extra_tokens)
    return model(input_ids=ids, past_key_values=base_past, use_cache=True)


def _measure(model, ids: torch.Tensor, base_past, repeats: int, warmup: int):
    wall: list[float] = []
    for index in range(warmup + repeats):
        _sync()
        start = time.perf_counter()
        _call(model, base_past, ids)
        _sync()
        elapsed_ms = (time.perf_counter() - start) * 1000
        if index >= warmup:
            wall.append(elapsed_ms)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        result = _call(model, base_past, ids)
        _sync()
    del result
    events = prof.key_averages()
    gpu_ms = sum(_self_device_time_us(event) for event in events) / 1000.0
    wall_ms = float(np.median(wall))
    return wall_ms, gpu_ms, wall_ms - gpu_ms, wall


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--models", nargs="+", choices=["qwen06", "qwen4", "qwen8"],
                        default=["qwen06", "qwen4", "qwen8"])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rebuttal/wallclock_opt")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup nonnegative")
    config = load_config(args.config)
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    prompt, continuation, pool_index = _prompt()
    cpu = subprocess.run(["lscpu"], check=True, capture_output=True, text=True).stdout
    (output / "lscpu.txt").write_text(cpu, encoding="utf-8")
    write_gpu_metadata(
        output / "run_metadata.json", job="forward_overhead_profile", require_idle=True,
        details={"models": args.models, "pool_index": pool_index, "prompt_length": len(prompt),
                 "operations": ["cached_single_token", "verify_5", "verify_17"],
                 "repeats": args.repeats, "warmup": args.warmup,
                 "model_revisions": {key: config["models"][key]["revision"] for key in args.models},
                 "cpu_lscpu": cpu})

    rows: list[dict] = []
    api = HfApi(token=get_token())
    token = get_token()
    device = torch.device("cuda")
    for key in args.models:
        print(f"PROFILE_MODEL_START model={key}", flush=True)
        model, snapshot, storage_mode = _load(config, key, api, token)
        ids_prompt = torch.tensor([prompt], dtype=torch.long, device=device)
        past = _cache(model, ids_prompt)
        layer_count = int(getattr(model.config, "num_hidden_layers", 0))
        if not layer_count:
            layer_count = len(getattr(model, "model").layers)
        inputs = {
            "single_token": torch.tensor([[continuation[0]]], dtype=torch.long, device=device),
            "verify_5": torch.tensor([continuation[:5]], dtype=torch.long, device=device),
            "verify_17": torch.tensor([continuation[:17]], dtype=torch.long, device=device),
        }
        for operation, ids in inputs.items():
            wall_ms, gpu_ms, host_ms, _ = _measure(model, ids, past, args.repeats, args.warmup)
            rows.append({"row_type": "model", "model": key, "operation": operation,
                         "n_tokens": int(ids.shape[1]), "context_length": len(prompt),
                         "layer_count": layer_count, "wall_ms": wall_ms,
                         "wall_ms_per_layer": wall_ms / layer_count,
                         "gpu_kernel_ms": gpu_ms, "host_overhead_ms": host_ms,
                         "host_overhead_share": host_ms / wall_ms if wall_ms else float("nan"),
                         "repeats": args.repeats, "storage_mode": storage_mode,
                         "snapshot": str(snapshot)})
            print(f"PROFILE_ROW model={key} op={operation} wall_ms={wall_ms:.3f} gpu_ms={gpu_ms:.3f} host_ms={host_ms:.3f}", flush=True)
        del past, model
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        _sync()

    by_model = {row["model"]: row for row in rows if row["operation"] == "single_token"}
    verify = {(row["model"], row["n_tokens"]): row for row in rows if row["operation"].startswith("verify_")}
    for draft, target, pair in (("qwen06", "qwen4", "Q2"), ("qwen06", "qwen8", "Q5")):
        if draft not in by_model or target not in by_model:
            continue
        for n in (5, 17):
            dr = by_model[draft]
            tr = verify[(target, n)]
            rows.append({"row_type": "pair_ratio", "pair": pair, "draft_model": draft,
                         "target_model": target, "operation": f"draft_1_over_target_verify_{n}",
                         "n_tokens": n, "wall_ms_ratio": dr["wall_ms"] / tr["wall_ms"],
                         "gpu_kernel_ms_ratio": dr["gpu_kernel_ms"] / tr["gpu_kernel_ms"],
                         "draft_wall_ms": dr["wall_ms"], "target_wall_ms": tr["wall_ms"],
                         "draft_gpu_kernel_ms": dr["gpu_kernel_ms"],
                         "target_gpu_kernel_ms": tr["gpu_kernel_ms"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "overhead.csv", index=False)
    print(f"OVERHEAD_PROFILE_COMPLETE rows={len(frame)} output={output / 'overhead.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
