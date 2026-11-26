#!/usr/bin/env python3
"""Check T8 sampled-text language, length, EOS, and target NLL gates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.rebuttal.score_cross_family_judge import _snapshot_dir
from src.naacl_experiment import load_config, load_tokenizer


def _hangul(ch: str) -> bool:
    code = ord(ch)
    return (0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F
            or 0xA960 <= code <= 0xA97F or 0xAC00 <= code <= 0xD7A3
            or 0xD7B0 <= code <= 0xD7FF)


def _target_stats(target: str, sampled_root: Path, overlap_root: Path, config: dict,
                  language_only: bool = False) -> dict:
    directory = sampled_root / "continuations" / target / "common"
    files = sorted(directory.glob("shard-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No sampled continuations under {directory}")
    continuations = pd.concat([
        pd.read_parquet(file, columns=["doc_id", "pool_index", "continuation_ids", "continuation_length", "eos_hit"])
        for file in files
    ], ignore_index=True).sort_values("pool_index", kind="stable")
    snapshot = _snapshot_dir(config, target)
    tokenizer = load_tokenizer(config, target, snapshot=snapshot)
    hangul_shares = []
    for row in continuations.itertuples(index=False):
        ids = list(map(int, row.continuation_ids))
        text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        hangul_shares.append(sum(_hangul(ch) for ch in text) / len(text) if text else 0.0)
    if language_only:
        nll = entropy = difference = None
    else:
        overlap_dir = overlap_root / ("Q2" if target == "qwen4" else "Q5") / "sampled"
        score_files = sorted(overlap_dir.glob("shard-*.parquet"))
        if not score_files:
            raise FileNotFoundError(f"No sampled overlap scores under {overlap_dir}")
        scores = pd.concat([pd.read_parquet(file, columns=["target_logp_sampled_token", "target_entropy"])
                            for file in score_files], ignore_index=True)
        nll = float(-scores.target_logp_sampled_token.mean())
        entropy = float(scores.target_entropy.mean())
        difference = abs(nll - entropy) / entropy if entropy else float("inf")
    lengths = continuations.continuation_length.to_numpy(dtype=np.int64)
    result = {
        "target": target,
        "documents": int(len(continuations)),
        "continuation_length_tokens": {
            "min": int(lengths.min()), "p10": float(np.quantile(lengths, .1)),
            "median": float(np.median(lengths)), "mean": float(lengths.mean()),
            "p90": float(np.quantile(lengths, .9)), "max": int(lengths.max()),
        },
        "eos_hit_rate": float(continuations.eos_hit.mean()),
        "hangul_share_by_document": {
            "median": float(np.median(hangul_shares)),
            "p10": float(np.quantile(hangul_shares, .1)),
        },
        "sampled_target_mean_nll": nll,
        "target_mean_entropy": entropy,
        "relative_nll_entropy_difference": difference,
        "unfiltered_sampling_gate_pass": None if language_only else difference <= .05,
        "hangul_median_gate_pass": float(np.median(hangul_shares)) >= .6,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--sampled-root", type=Path, default=ROOT / "results/rebuttal/sampled_T1")
    parser.add_argument("--overlap-root", type=Path, default=ROOT / "results/rebuttal/overlap")
    parser.add_argument("--language-only", action="store_true",
                        help="Write the language/length gate before morphology or overlap scoring")
    args = parser.parse_args()
    config = load_config(args.config)
    sampled_root = args.sampled_root if args.sampled_root.is_absolute() else ROOT / args.sampled_root
    overlap_root = args.overlap_root if args.overlap_root.is_absolute() else ROOT / args.overlap_root
    results = [_target_stats(target, sampled_root, overlap_root, config, args.language_only)
               for target in ("qwen4", "qwen8")]
    output_dir = sampled_root / "gates"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"temperature": 1.0, "seed": 20261125, "results": results}
    name = "language_gate" if args.language_only else "summary"
    (output_dir / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# T8 sampled-text gates", "",
        "| Target | Docs | Length median (p10–p90) | EOS rate | Hangul median (p10) | Mean sampled NLL | Mean entropy | Relative difference | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        maybe = lambda value: "—" if value is None else f"{value:.4f}"
        difference = ("—" if row["relative_nll_entropy_difference"] is None
                      else f"{row['relative_nll_entropy_difference']:.4%}")
        lines.append(
            f"| {row['target']} | {row['documents']} | {row['continuation_length_tokens']['median']:.1f} "
            f"({row['continuation_length_tokens']['p10']:.1f}–{row['continuation_length_tokens']['p90']:.1f}) | "
            f"{row['eos_hit_rate']:.4f} | {row['hangul_share_by_document']['median']:.4f} "
            f"({row['hangul_share_by_document']['p10']:.4f}) | {maybe(row['sampled_target_mean_nll'])} | "
            f"{maybe(row['target_mean_entropy'])} | {difference} | "
            f"{row['hangul_median_gate_pass'] if args.language_only else row['unfiltered_sampling_gate_pass'] and row['hangul_median_gate_pass']} |"
        )
    (output_dir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if not all(row["hangul_median_gate_pass"] for row in results):
        return 2
    if args.language_only:
        return 0
    return 0 if all(row["unfiltered_sampling_gate_pass"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
