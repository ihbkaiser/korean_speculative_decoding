#!/usr/bin/env python3
"""Probe cached single-token, verification-block, and KV-crop latency."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.naacl_experiment import load_config, load_model, model_path
from src.rebuttal.runtime import write_gpu_metadata
from src.speculative_decoding import _crop_past_key_values


def _test_prompt(target: str, split: str) -> tuple[list[int], list[int], int]:
    directory = ROOT / "results/naacl_experiments/continuations" / target / split
    files = sorted(directory.glob("shard-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No stored continuations found under {directory}")
    rows = pd.concat(
        [pd.read_parquet(path, columns=["pool_index", "prompt_ids", "continuation_ids"]) for path in files],
        ignore_index=True,
    )
    rows = rows[(rows.pool_index % 2 == 1) & rows.prompt_ids.map(len).ge(128)]
    if rows.empty:
        raise RuntimeError(f"No 128-token test prompt found for target={target}, split={split}")
    row = rows.sort_values("pool_index", kind="stable").iloc[0]
    prompt = [int(token) for token in row.prompt_ids[:128]]
    continuation = [int(token) for token in row.continuation_ids]
    if len(continuation) < 128:
        continuation = (continuation + prompt * 2)[:128]
    return prompt, continuation, int(row.pool_index)


def _sync() -> None:
    torch.cuda.synchronize()


@torch.inference_mode()
def _timed_forward(model, base_past, ids, repeats: int, warmup: int) -> list[float]:
    samples: list[float] = []
    for iteration in range(warmup + repeats):
        past = copy.deepcopy(base_past)
        _sync()
        started = time.perf_counter()
        model(input_ids=ids, past_key_values=past, use_cache=True)
        _sync()
        elapsed = (time.perf_counter() - started) * 1000
        if iteration >= warmup:
            samples.append(elapsed)
    return samples


def _cache(model, ids):
    with torch.inference_mode():
        return model(input_ids=ids, use_cache=True).past_key_values


def _summary(model_key: str, operation: str, n_tokens: int, context_len: int, values: list[float], reps: int):
    return {
        "model": model_key,
        "op": operation,
        "n_tokens": n_tokens,
        "context_len": context_len,
        "median_ms": float(np.median(values)),
        "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)),
        "reps": reps,
    }


def _expected_rows(model_key: str, single_repeats: int, verify_repeats: int) -> set[tuple[str, str, int, int]]:
    expected = {
        (model_key, "single_token", 1, 128),
        (model_key, "single_token", 1, 256),
        *((model_key, "verify", n, 128) for n in (1, 2, 3, 5, 9, 17)),
        *((model_key, "cache_crop", n, 256) for n in (1, 4, 16)),
    }
    return expected


def _save_latency(rows: list[dict], destination: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows).sort_values(["model", "op", "context_len", "n_tokens"])
    temporary = destination.with_name(destination.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--prompt-target", required=True)
    parser.add_argument("--split", choices=("common", "all"), default="common")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--single-repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rebuttal/latency")
    parser.add_argument("--no-cost-ratios", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    for key in args.models + [args.prompt_target]:
        if key not in config["models"]:
            parser.error(f"unknown model key {key!r}")
    if args.repeats < 1 or args.single_repeats < 1 or args.warmup < 0:
        parser.error("repeat counts must be positive and warmup non-negative")

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt, continuation, pool_index = _test_prompt(args.prompt_target, args.split)
    write_gpu_metadata(
        output_dir / "run_metadata.json",
        job="latency_probe",
        require_idle=True,
        details={
            "models": args.models,
            "prompt_target": args.prompt_target,
            "split": args.split,
            "pool_index": pool_index,
            "prompt_length": len(prompt),
            "repetitions": {"single_token": args.single_repeats, "verify_and_crop": args.repeats},
            "warmup": args.warmup,
            "model_revisions": {key: config["models"][key]["revision"] for key in args.models},
        },
    )

    latency_path = output_dir / "latency.csv"
    if latency_path.is_file():
        rows = pd.read_csv(latency_path).to_dict("records")
    else:
        rows = []
    for model_key in args.models:
        existing = pd.DataFrame([row for row in rows if row["model"] == model_key])
        expected = _expected_rows(model_key, args.single_repeats, args.repeats)
        observed = set(
            (row.model, row.op, int(row.n_tokens), int(row.context_len))
            for row in existing.itertuples()
        ) if not existing.empty else set()
        required_reps = {"single_token": args.single_repeats, "verify": args.repeats, "cache_crop": args.repeats}
        if observed == expected and all(
            int(row.reps) == required_reps[row.op] for row in existing.itertuples()
        ):
            print(f"LATENCY_RESUME_SKIP model={model_key}", flush=True)
            continue
        rows = [row for row in rows if row["model"] != model_key]
        model = load_model(config, model_key)
        model.eval()
        device = next(model.parameters()).device
        context_by_len = {
            128: prompt,
            256: prompt + continuation[:128],
        }
        for context_len, tokens in context_by_len.items():
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            base_past = _cache(model, input_ids)
            next_id = torch.tensor([[continuation[0]]], dtype=torch.long, device=device)
            values = _timed_forward(model, base_past, next_id, args.single_repeats, args.warmup)
            rows.append(_summary(model_key, "single_token", 1, context_len, values, args.single_repeats))
            if context_len == 128:
                for n_tokens in (1, 2, 3, 5, 9, 17):
                    ids = torch.tensor([continuation[:n_tokens]], dtype=torch.long, device=device)
                    values = _timed_forward(model, base_past, ids, args.repeats, args.warmup)
                    rows.append(_summary(model_key, "verify", n_tokens, context_len, values, args.repeats))

        crop_base = _cache(model, torch.tensor([context_by_len[256]], dtype=torch.long, device=device))
        for crop_by in (1, 4, 16):
            samples = []
            for iteration in range(args.warmup + args.repeats):
                cache_copy = copy.deepcopy(crop_base)
                _sync()
                started = time.perf_counter()
                cropped = _crop_past_key_values(cache_copy, 256 - crop_by)
                _sync()
                elapsed = (time.perf_counter() - started) * 1000
                if cropped is None:
                    raise RuntimeError(f"Cache crop by {crop_by} failed for {model_key}")
                if iteration >= args.warmup:
                    samples.append(elapsed)
            rows.append(_summary(model_key, "cache_crop", crop_by, 256, samples, args.repeats))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _save_latency(rows, latency_path)

    latency = _save_latency(rows, latency_path)
    if not args.no_cost_ratios:
        sizes = {key: float(spec["size"].removesuffix("B")) for key, spec in config["models"].items()}
        measured = latency.pivot_table(index="model", columns=["op", "n_tokens", "context_len"], values="median_ms")
        pair_defs = config.get("pairs", {})
        cost_rows = []
        for pair, spec in pair_defs.items():
            draft, target = spec["draft"], spec["target"]
            if draft not in measured.index or target not in measured.index:
                continue
            draft_step = measured.loc[draft, ("single_token", 1, 128)]
            target_verify = measured.loc[target, ("verify", 5, 128)]
            cost_rows.append({
                "pair": pair,
                "draft_model": draft,
                "target_model": target,
                "t_draft_step_ms": float(draft_step),
                "t_target_verify5_ms": float(target_verify),
                "c_measured": float(draft_step / target_verify),
                "c_param": float(sizes[draft] / sizes[target]),
            })
        pd.DataFrame(cost_rows).to_csv(output_dir / "cost_ratio.csv", index=False)
    print(f"LATENCY_COMPLETE output={latency_path} rows={len(latency)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
