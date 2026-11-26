#!/usr/bin/env python3
"""Measure the compiled fixed-shape StaticCache forward path used by T7."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from huggingface_hub.utils import get_token
from transformers import StaticCache

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.rebuttal.profile_forward_overhead import _prompt
from scripts.rebuttal.score_cross_family_judge import _load_streamed_model
from src.naacl_experiment import atomic_json, load_config
from src.rebuttal.runtime import write_gpu_metadata
from src.rebuttal.static_compiled_sd import _call, _set_logical_length, clear_compiled_models, _compiled


def _sync():
    torch.cuda.synchronize()


def _self_device_time_us(event) -> float:
    if not str(getattr(event, "device_type", "")).endswith("CUDA"):
        return 0.0
    value = getattr(event, "self_device_time_total", None)
    if value is None:
        value = event.self_cuda_time_total
    return float(value)


def _measure(model, cache, ids, positions, valid_end, capacity, repeats, warmup):
    wall = []
    for index in range(warmup + repeats):
        _set_logical_length(cache, 128)
        _sync()
        start = time.perf_counter()
        _call(model, cache, ids, positions, 128, valid_end, capacity, logits_to_keep=ids.shape[1])
        _sync()
        elapsed = (time.perf_counter() - start) * 1000
        if index >= warmup:
            wall.append(elapsed)
    _set_logical_length(cache, 128)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        _call(model, cache, ids, positions, 128, valid_end, capacity, logits_to_keep=ids.shape[1])
        _sync()
    gpu_ms = sum(_self_device_time_us(event) for event in prof.key_averages()) / 1000.0
    wall_ms = float(np.median(wall))
    return wall_ms, gpu_ms, wall_ms - gpu_ms


@torch.inference_mode()
def _profile_model(key: str, config, api, token, mode: str, repeats: int, warmup: int) -> list[dict]:
    model, snapshot, storage_mode = _load_streamed_model(config, key, api, token)
    model.config.use_cache = True
    model.eval()
    device = next(model.parameters()).device
    prompt, continuation, pool_index = _prompt()
    capacity = len(prompt) + 128 + 17
    cache = StaticCache(model.config, max_cache_len=capacity)
    prompt_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    prompt_pos = torch.arange(len(prompt), device=device, dtype=torch.long)[None, :]
    mask = torch.zeros((1, capacity), dtype=torch.long, device=device)
    mask[:, :len(prompt)] = 1
    prefill = model(input_ids=prompt_ids, attention_mask=mask, position_ids=prompt_pos,
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
    step_model = _compiled(model, True, mode)
    one_ids = torch.tensor([[continuation[0]]], dtype=torch.long, device=device)
    one_pos = torch.tensor([[len(prompt)]], dtype=torch.long, device=device)
    _set_logical_length(cache, len(prompt))
    _call(step_model, cache, one_ids, one_pos, len(prompt), len(prompt) + 1, capacity, 1)
    ids17 = continuation[:17]
    pad_id = int(getattr(model.config, "pad_token_id", None) or 0)
    inputs = {}
    for n in (5, 17):
        values = ids17[:n] + [pad_id] * (17 - n)
        ids = torch.tensor([values], dtype=torch.long, device=device)
        positions = torch.arange(len(prompt), len(prompt) + 17, dtype=torch.long, device=device)[None, :]
        inputs[n] = (ids, positions, len(prompt) + n)
        _set_logical_length(cache, len(prompt))
        _call(step_model, cache, ids, positions, len(prompt), len(prompt) + n,
              capacity, logits_to_keep=17)

    rows = []
    for operation, n, ids, positions, valid_end in (
        ("cached_single_token", 1, one_ids, one_pos, len(prompt) + 1),
        ("verify_5_fixed17", 5, *inputs[5]),
        ("verify_17_fixed17", 17, *inputs[17]),
    ):
        wall_ms, gpu_ms, host_ms = _measure(step_model, cache, ids, positions, valid_end,
                                            capacity, repeats, warmup)
        rows.append({"row_type": "model", "model": key, "operation": operation,
                     "meaningful_tokens": n, "input_shape": list(ids.shape),
                     "context_length": len(prompt), "layer_count": int(model.config.num_hidden_layers),
                     "wall_ms": wall_ms, "gpu_kernel_ms": gpu_ms, "host_overhead_ms": host_ms,
                     "host_overhead_share": host_ms / wall_ms if wall_ms else float("nan"),
                     "wall_ms_per_layer": wall_ms / int(model.config.num_hidden_layers),
                     "compile_mode": mode, "storage_mode": storage_mode,
                     "model_revision": config["models"][key]["revision"],
                     "pool_index": pool_index})
        print(f"OPT_LATENCY model={key} op={operation} wall_ms={wall_ms:.3f} gpu_ms={gpu_ms:.3f}", flush=True)
    del prefill, cache, step_model, model
    clear_compiled_models()
    gc.collect()
    torch.cuda.empty_cache()
    _sync()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--models", nargs="+", choices=("qwen06", "qwen4", "qwen8"),
                        default=("qwen06", "qwen4", "qwen8"))
    parser.add_argument("--compile-mode", choices=("reduce-overhead", "default"), default="reduce-overhead")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rebuttal/latency_opt")
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_gpu_metadata(output / "run_metadata.json", job="optimized_static_latency_probe",
                       require_idle=True,
                       details={"models": args.models, "compile_mode": args.compile_mode,
                                "repeats": args.repeats, "warmup": args.warmup,
                                "model_revisions": {key: config["models"][key]["revision"] for key in args.models}})
    api, token = HfApi(token=get_token()), get_token()
    mode = args.compile_mode
    try:
        rows = []
        for key in args.models:
            rows.extend(_profile_model(key, config, api, token, mode, args.repeats, args.warmup))
    except Exception as error:
        if mode != "reduce-overhead":
            raise
        print(f"COMPILE_MODE_FALLBACK from=reduce-overhead to=default error={type(error).__name__}: {error}",
              flush=True)
        clear_compiled_models()
        gc.collect()
        torch.cuda.empty_cache()
        torch._dynamo.reset()
        mode = "default"
        rows = []
        for key in args.models:
            rows.extend(_profile_model(key, config, api, token, mode, args.repeats, args.warmup))
    pd.DataFrame(rows).to_csv(output / "overhead_opt.csv", index=False)
    by_key = {row["model"]: row for row in rows if row["operation"] == "cached_single_token"}
    by_verify = {(row["model"], row["meaningful_tokens"]): row for row in rows
                 if row["operation"].startswith("verify_")}
    ratios = []
    for pair, target in (("Q2", "qwen4"), ("Q5", "qwen8")):
        if "qwen06" not in by_key or target not in by_key:
            continue
        draft = by_key["qwen06"]
        for n in (5, 17):
            target_row = by_verify[(target, n)]
            wall_ratio = draft["wall_ms"] / target_row["wall_ms"]
            gpu_ratio = draft["gpu_kernel_ms"] / target_row["gpu_kernel_ms"]
            ratios.append({"pair": pair, "draft_model": "qwen06", "target_model": target,
                           "target_verify_tokens": n, "draft_wall_ms": draft["wall_ms"],
                           "target_verify_wall_ms": target_row["wall_ms"],
                           "draft_gpu_kernel_ms": draft["gpu_kernel_ms"],
                           "target_verify_gpu_kernel_ms": target_row["gpu_kernel_ms"],
                           "c_measured": wall_ratio, "c_measured_wall": wall_ratio,
                           "c_measured_gpu": gpu_ratio,
                           "c_param": float(config["models"]["qwen06"]["size"].removesuffix("B")) /
                                      float(config["models"][target]["size"].removesuffix("B")),
                           "compile_mode": mode})
    pd.DataFrame(ratios).to_csv(output / "cost_ratio_opt.csv", index=False)
    atomic_json(output / "compile_mode.json", {"compile_mode": mode, "fallback_used": mode != args.compile_mode})
    print(f"OPT_LATENCY_COMPLETE models={len(args.models)} cost_rows={len(ratios)} mode={mode}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
