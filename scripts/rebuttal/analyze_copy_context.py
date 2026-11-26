#!/usr/bin/env python3
"""Analyze disagreement under the four human-text context-copy interventions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/rebuttal/cpu"))

from ksdlib import read_aligned, std_ratio

PAIRS = {"Q2": ("qwen4", "qwen06"), "M1": ("ministral8", "ministral3")}
CONDITIONS = ("A", "B", "C", "D")
PTYPES = ("word_onset", "morph_onset", "fused", "internal")


def _condition_data(pair: str, condition: str, target: str, draft: str) -> pd.DataFrame:
    directory = ROOT / "results/rebuttal/copy_context" / pair / condition
    target_parts = [pd.read_parquet(path) for path in sorted((directory / target).glob("shard-*.parquet"))]
    draft_parts = [pd.read_parquet(path) for path in sorted((directory / draft).glob("shard-*.parquet"))]
    if not target_parts or not draft_parts:
        raise FileNotFoundError(f"Missing context-score shards for {pair}/{condition}")
    t = pd.concat(target_parts, ignore_index=True)
    d = pd.concat(draft_parts, ignore_index=True)
    merged = t[["doc_id", "pool_index", "position", "argmax_id", "top2_margin", "entropy"]].merge(
        d[["doc_id", "pool_index", "position", "argmax_id", "top2_margin", "entropy"]],
        on=["doc_id", "pool_index", "position"], suffixes=("_target", "_draft"),
        how="inner", validate="one_to_one",
    )
    if len(merged) != len(t) or len(merged) != len(d):
        raise RuntimeError(f"Target/draft positions do not align for {pair}/{condition}")
    merged["condition"] = condition
    merged["disagreement"] = (merged.argmax_id_target != merged.argmax_id_draft).astype(np.int8)
    return merged


def _shared_bins(frames: list[pd.DataFrame]) -> tuple[np.ndarray, float]:
    maximum = max(float(frame.top2_margin_target.max()) for frame in frames)
    grid = 0.0625
    stride = int(np.ceil(maximum / (12 * grid)))
    step = grid * max(stride, 1)
    edges = np.arange(0.0, np.ceil(maximum / step) * step + 2 * step, step)
    if len(edges) < 3:
        edges = np.array([0.0, step, 2 * step])
    return edges, step


def _position_type_ratios(group: pd.DataFrame) -> dict[str, float]:
    records = {}
    for name, numerator, denominator, column in (
        ("morph_internal", "morph_onset", "internal", "ptype"),
        ("word_internal", "word_onset", "internal", "ptype"),
        ("fused_split", "CROSS", "WITHIN_SPLIT", "aclass"),
    ):
        try:
            matched, raw = std_ratio(group, numerator, denominator, "target_margin", nbins=10, col=column, y="disagreement")
            records[f"{name}_raw"] = raw
            records[f"{name}_margin_matched"] = matched
        except (ValueError, ZeroDivisionError, KeyError):
            records[f"{name}_raw"] = float("nan")
            records[f"{name}_margin_matched"] = float("nan")
    return records


def _fit_logit(frame: pd.DataFrame, include_margin_bins: bool) -> pd.DataFrame:
    valid = frame[(frame.ptype.isin(PTYPES)) & frame.margin_bin.notna()].copy()
    grouped = valid.groupby(["doc_id", "condition", "ptype", "margin_bin"], observed=True).disagreement.agg(
        successes="sum", positions="size"
    ).reset_index()
    grouped["rate"] = grouped.successes / grouped.positions
    formula = "rate ~ C(condition, Treatment(reference='A')) * C(ptype, Treatment(reference='internal'))"
    if include_margin_bins:
        formula += " + C(margin_bin)"
    y, x = patsy.dmatrices(formula, grouped, return_type="dataframe")
    fit = sm.GLM(
        y, x, family=sm.families.Binomial(), freq_weights=grouped.positions
    ).fit(cov_type="cluster", cov_kwds={"groups": grouped.doc_id})
    keep = [term for term in fit.params.index if "condition" in term]
    z = 1.959963984540054
    return pd.DataFrame({
        "coefficient": keep,
        "estimate_log_odds": [float(fit.params[name]) for name in keep],
        "clustered_se": [float(fit.bse[name]) for name in keep],
        "p_value": [float(fit.pvalues[name]) for name in keep],
        "ci_low": [float(fit.params[name] - z * fit.bse[name]) for name in keep],
        "ci_high": [float(fit.params[name] + z * fit.bse[name]) for name in keep],
        "margin_bin_fixed_effects": include_margin_bins,
        "clusters": int(grouped.doc_id.nunique()),
        "aggregated_cells": int(len(grouped)),
    })


def _analyze_pair(pair: str, output_dir: Path):
    target, draft = PAIRS[pair]
    frames = [_condition_data(pair, condition, target, draft) for condition in CONDITIONS]
    ids = [set(frame.doc_id.astype(str)) for frame in frames]
    if any(value != ids[0] for value in ids[1:]):
        raise RuntimeError(f"Document sets differ across conditions for {pair}")
    base_keys = ["doc_id", "pool_index", "position"]
    baseline = frames[0]
    aligned = read_aligned(target, "human", common_only=True).rename(columns={"doc": "pool_index", "pos": "position"})
    aligned = aligned[["pool_index", "position", "ptype", "aclass"]]
    full = []
    for frame in frames:
        full.append(frame.merge(aligned, on=["pool_index", "position"], how="left", validate="one_to_one"))
    edges, step = _shared_bins(full)
    for frame in full:
        frame["ptype"] = frame.ptype.fillna("excluded")
        frame["aclass"] = frame.aclass.fillna("OTHER")
        frame["target_margin"] = frame.top2_margin_target
        frame["target_decision_point"] = frame.target_margin < 1.0
        frame["margin_bin"] = pd.cut(frame.target_margin, edges, right=False, include_lowest=True, labels=False)
    curve_rows = []
    summary_rows = []
    for condition, frame in zip(CONDITIONS, full):
        row = {
            "pair": pair, "condition": condition, "positions": len(frame),
            "documents": frame.doc_id.nunique(),
            "disagreement_rate": float(frame.disagreement.mean()),
            "mean_target_margin": float(frame.target_margin.mean()),
            "target_decision_point_share": float(frame.target_decision_point.mean()),
            "margin_bin_step_nats": step,
            **_position_type_ratios(frame),
        }
        by_type = frame[frame.ptype.isin(PTYPES)].groupby("ptype", observed=True).agg(
            positions=("disagreement", "size"), disagreement_rate=("disagreement", "mean"),
            mean_target_margin=("target_margin", "mean"), decision_point_share=("target_decision_point", "mean"),
        )
        for ptype in PTYPES:
            if ptype in by_type.index:
                vals = by_type.loc[ptype]
                row[f"{ptype}_positions"] = int(vals.positions)
                row[f"{ptype}_disagreement"] = float(vals.disagreement_rate)
                row[f"{ptype}_target_margin"] = float(vals.mean_target_margin)
                row[f"{ptype}_decision_point_share"] = float(vals.decision_point_share)
        summary_rows.append(row)
        curve = frame.groupby("margin_bin", observed=True).agg(
            positions=("disagreement", "size"), disagreement_rate=("disagreement", "mean"),
            mean_target_margin=("target_margin", "mean"),
        ).reset_index()
        for _, value in curve.iterrows():
            curve_rows.append({"pair": pair, "condition": condition, "margin_bin": int(value.margin_bin),
                               "bin_left": float(edges[int(value.margin_bin)]),
                               "bin_right": float(edges[int(value.margin_bin) + 1]), **value.drop(labels="margin_bin").to_dict()})

    curve = pd.DataFrame(curve_rows)
    base_curve = curve[curve.condition == "A"].set_index("margin_bin")
    gap_rows = []
    for condition in ("B", "C", "D"):
        c = curve[curve.condition == condition].set_index("margin_bin")
        joined = base_curve[["positions", "disagreement_rate"]].join(
            c[["positions", "disagreement_rate"]], lsuffix="_A", rsuffix=f"_{condition}", how="inner"
        )
        gaps = (joined[f"disagreement_rate_{condition}"] - joined.disagreement_rate_A).abs()
        weights = joined.positions_A.to_numpy(dtype=np.float64)
        gap_rows.append({"pair": pair, "condition": condition, "largest_absolute_gap": float(gaps.max()),
                         "position_weighted_mean_absolute_gap": float(np.average(gaps, weights=weights)),
                         "bins_compared": int(len(joined))})

    coefficients = []
    for include in (False, True):
        fitted = _fit_logit(pd.concat(full, ignore_index=True), include)
        fitted["pair"] = pair
        coefficients.append(fitted)

    a = full[0][base_keys + ["ptype", "margin_bin", "disagreement"]].rename(
        columns={"margin_bin": "margin_bin_A", "disagreement": "disagreement_A"}
    )
    a = a[a.ptype.isin(PTYPES)]
    b = full[1][base_keys + ["margin_bin", "disagreement"]].rename(
        columns={"margin_bin": "margin_bin_B", "disagreement": "disagreement_B"}
    )
    paired = a.merge(b, on=base_keys, how="inner", validate="one_to_one")
    paired["change_B_minus_A"] = paired.disagreement_B - paired.disagreement_A
    paired_summary = paired.groupby(["ptype", "margin_bin_A", "margin_bin_B"], observed=True).agg(
        positions=("change_B_minus_A", "size"), mean_disagreement_change=("change_B_minus_A", "mean"),
        A_disagreement=("disagreement_A", "mean"), B_disagreement=("disagreement_B", "mean"),
    ).reset_index()
    paired_summary.insert(0, "pair", pair)

    summary = pd.DataFrame(summary_rows)
    pd.DataFrame(gap_rows).to_csv(output_dir / f"{pair}_curve_gaps.csv", index=False)
    curve.to_csv(output_dir / f"{pair}_curves.csv", index=False)
    pd.concat(coefficients, ignore_index=True).to_csv(output_dir / f"{pair}_logistic.csv", index=False)
    paired_summary.to_csv(output_dir / f"{pair}_paired_A_B.csv", index=False)
    return summary, pd.DataFrame(gap_rows), pd.concat(coefficients, ignore_index=True), paired_summary, curve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs="+", choices=tuple(PAIRS), default=list(PAIRS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rebuttal/copy_context")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries, gaps, models, paired, curves = [], [], [], [], []
    for pair in args.pair:
        result = _analyze_pair(pair, output_dir)
        summaries.append(result[0]); gaps.append(result[1]); models.append(result[2]); paired.append(result[3]); curves.append(result[4])
        print(f"ANALYZED_COPY_CONTEXT pair={pair}", flush=True)
    pd.concat(summaries, ignore_index=True).to_csv(output_dir / "summary.csv", index=False)
    pd.concat(gaps, ignore_index=True).to_csv(output_dir / "curve_gaps.csv", index=False)
    pd.concat(models, ignore_index=True).to_csv(output_dir / "logistic.csv", index=False)
    pd.concat(paired, ignore_index=True).to_csv(output_dir / "paired_A_B.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(output_dir / "curves.csv", index=False)
    summary = pd.concat(summaries, ignore_index=True)
    gap_data = pd.concat(gaps, ignore_index=True)
    model_data = pd.concat(models, ignore_index=True)
    lines = [
        "# Copy-context intervention",
        "",
        "This intervention changes decision strength while keeping the scored human continuation tokens and their aligned position types fixed. It does not manipulate morphology. In condition B, both target and draft can copy the continuation from the preceding context.",
        "",
        "| Pair | Condition | Positions | Disagreement | Mean target margin | Target DP share | Morph/internal raw → matched | Word/internal raw → matched | Fused/split raw → matched |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        f = lambda key: f"{row[key]:.3f}" if pd.notna(row.get(key)) else "NA"
        lines.append(
            f"| {row['pair']} | {row['condition']} | {int(row['positions'])} | {f('disagreement_rate')} | "
            f"{f('mean_target_margin')} | {f('target_decision_point_share')} | "
            f"{f('morph_internal_raw')} → {f('morph_internal_margin_matched')} | "
            f"{f('word_internal_raw')} → {f('word_internal_margin_matched')} | "
            f"{f('fused_split_raw')} → {f('fused_split_margin_matched')} |"
        )
    lines += ["", "## Common margin-curve gaps versus A", "",
              "| Pair | Condition | Largest absolute gap | Position-weighted mean absolute gap | Bins compared |",
              "|---|---|---:|---:|---:|"]
    for row in gap_data.to_dict("records"):
        lines.append(f"| {row['pair']} | {row['condition']} | {row['largest_absolute_gap']:.4f} | {row['position_weighted_mean_absolute_gap']:.4f} | {int(row['bins_compared'])} |")
    lines += ["", "## Logistic condition and interaction coefficients", "",
              "Coefficients are log odds with document-clustered standard errors; models are shown with and without margin-bin fixed effects. See `logistic.csv` for intervals and p-values.", "",
              "| Pair | Margin-bin fixed effects | Coefficient | Estimate | Clustered SE | p-value |",
              "|---|---|---|---:|---:|---:|"]
    for row in model_data.to_dict("records"):
        lines.append(f"| {row['pair']} | {row['margin_bin_fixed_effects']} | {row['coefficient']} | {row['estimate_log_odds']:.4f} | {row['clustered_se']:.4f} | {row['p_value']:.4g} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
