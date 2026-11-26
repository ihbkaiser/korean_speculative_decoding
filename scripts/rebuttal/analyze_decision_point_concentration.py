#!/usr/bin/env python3
"""Compare expected sampling rejections and greedy mismatches at target DPs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261125
BOOTSTRAPS = 500
SOURCES = {
    "Q2": ("human", "generated"),
    "Q5": ("human",),
    "M1": ("human", "generated"),
    "G1": ("generated",),
}


def _analyze(frame: pd.DataFrame, pair: str, continuation: str, rng: np.random.Generator) -> list[dict]:
    data = frame.copy()
    data["dp"] = data.target_top2_margin.to_numpy(dtype=np.float64) < 1.0
    data["disagree"] = data.target_argmax_id.to_numpy() != data.draft_argmax_id.to_numpy()
    data["dp_disagree"] = data.disagree & data.dp
    data["rejection_T1"] = 1.0 - data.alpha_T1.to_numpy(dtype=np.float64)
    data["dp_rejection_T1"] = data.rejection_T1 * data.dp.to_numpy(dtype=np.float64)
    data["rejection_T07"] = 1.0 - data.alpha_T07.to_numpy(dtype=np.float64)
    data["dp_rejection_T07"] = data.rejection_T07 * data.dp.to_numpy(dtype=np.float64)
    doc = data.groupby("doc_id", sort=True).agg(
        positions=("dp", "size"),
        dp_positions=("dp", "sum"),
        disagreements=("disagree", "sum"),
        dp_disagreements=("dp_disagree", "sum"),
        rejection_T1=("rejection_T1", "sum"),
        dp_rejection_T1=("dp_rejection_T1", "sum"),
        rejection_T07=("rejection_T07", "sum"),
        dp_rejection_T07=("dp_rejection_T07", "sum"),
    ).reset_index(drop=True)
    n = len(doc)
    columns = {name: doc[name].to_numpy(dtype=np.float64) for name in doc.columns if name != "doc_id"}

    def metrics(weights: np.ndarray) -> dict[str, float]:
        sums = {key: float(np.dot(weights, col)) for key, col in columns.items()}
        out = {"decision_point_position_share": sums["dp_positions"] / sums["positions"]}
        out["greedy_disagreement_mass_at_target_dp"] = (
            sums["dp_disagreements"] / sums["disagreements"] if sums["disagreements"] else float("nan")
        )
        for suffix in ("T1", "T07"):
            out[f"sampling_rejection_mass_at_target_dp_{suffix}"] = (
                sums[f"dp_rejection_{suffix}"] / sums[f"rejection_{suffix}"]
                if sums[f"rejection_{suffix}"] else float("nan")
            )
            out[f"sampling_minus_greedy_dp_mass_share_{suffix}"] = (
                out[f"sampling_rejection_mass_at_target_dp_{suffix}"]
                - out["greedy_disagreement_mass_at_target_dp"]
            )
        return out

    point = metrics(np.ones(n, dtype=np.float64))
    samples = {key: [] for key in point}
    for _ in range(BOOTSTRAPS):
        selected = rng.integers(0, n, size=n)
        weights = np.bincount(selected, minlength=n).astype(np.float64)
        for key, value in metrics(weights).items():
            if np.isfinite(value):
                samples[key].append(value)
    rows = []
    for temperature in ("T1", "T07"):
        row = {"pair": pair, "continuation": continuation, "temperature": temperature, **point}
        for key, vals in samples.items():
            if vals:
                row[f"{key}_ci_low"] = float(np.percentile(vals, 2.5))
                row[f"{key}_ci_high"] = float(np.percentile(vals, 97.5))
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "results/rebuttal/overlap")
    parser.add_argument("--output", type=Path, default=ROOT / "results/rebuttal/overlap/decision_point_comparison.csv")
    parser.add_argument("--pair", nargs="+", choices=tuple(SOURCES), default=list(SOURCES))
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    columns = ["doc_id", "alpha_T1", "alpha_T07", "target_argmax_id", "target_top2_margin", "draft_argmax_id"]
    for pair in args.pair:
        for continuation in SOURCES[pair]:
            files = sorted((input_root / pair / continuation).glob("shard-*.parquet"))
            if not files:
                continue
            frame = pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)
            rows.extend(_analyze(frame, pair, continuation, rng))
            print(f"ANALYZED_DP_CONCENTRATION pair={pair} continuation={continuation} positions={len(frame)}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    print(f"DP_CONCENTRATION_COMPLETE rows={len(result)} output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
