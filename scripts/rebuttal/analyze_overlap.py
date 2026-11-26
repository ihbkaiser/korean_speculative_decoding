#!/usr/bin/env python3
"""Summarize speculative-sampling overlap scores with document bootstraps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/rebuttal/cpu"))

from ksdlib import read_aligned

SEED = 20261125
BOOTSTRAPS = 500
TEXTS = {
    "Q2": ("human", "generated"),
    "Q5": ("human",),
    "M1": ("human", "generated"),
    "G1": ("generated",),
}


def _align(frame: pd.DataFrame, pair: str, text: str) -> pd.DataFrame:
    target = {"Q2": "qwen4", "Q5": "qwen8", "M1": "ministral8", "G1": "gemma12"}[pair]
    split = "human" if text == "human" else ("all" if pair == "Q2" else "common")
    aligned = read_aligned(target, split, common_only=True)
    aligned = aligned.rename(columns={"doc": "pool_index", "pos": "position"})
    frame = frame.merge(aligned, on=["pool_index", "position"], how="left", validate="one_to_one")
    frame["ptype"] = frame.ptype.fillna("excluded")
    frame["aclass"] = frame.aclass.fillna("OTHER")
    return frame


def _quantile_bins(values: np.ndarray, bins: int = 10) -> np.ndarray:
    # Match ksdlib.std_ratio: rank first to keep ties from collapsing bins.
    ranks = pd.Series(values).rank(method="first")
    return pd.qcut(ranks, bins, labels=False).to_numpy(dtype=np.int16)


def _weighted_rate_ratio(
    group: np.ndarray,
    outcome: np.ndarray,
    bins: np.ndarray,
    weights: np.ndarray,
    first: str,
    second: str,
    n_bins: int = 10,
) -> tuple[float, float]:
    # Keep all eligible rows for population weights. The paper's helper
    # includes EXACT rows when standardizing CROSS against WITHIN_SPLIT.
    g = group
    y = outcome
    b = bins
    w = weights
    first_mask = g == first
    second_mask = g == second
    raw_a = np.sum(w[first_mask] * y[first_mask]) / max(np.sum(w[first_mask]), 1e-30)
    raw_b = np.sum(w[second_mask] * y[second_mask]) / max(np.sum(w[second_mask]), 1e-30)
    idx_a = b[first_mask]
    idx_b = b[second_mask]
    ca = np.bincount(idx_a, weights=w[first_mask], minlength=n_bins)
    cb = np.bincount(idx_b, weights=w[second_mask], minlength=n_bins)
    sa = np.bincount(idx_a, weights=w[first_mask] * y[first_mask], minlength=n_bins)
    sb = np.bincount(idx_b, weights=w[second_mask] * y[second_mask], minlength=n_bins)
    common = (ca > 0) & (cb > 0)
    if not common.any():
        return float(raw_a / raw_b), float("nan")
    # The standardization weights are the distribution of all eligible positions.
    population = np.bincount(b, weights=w, minlength=n_bins)
    population = population[common]
    population = population / population.sum()
    matched_a = np.sum((sa[common] / ca[common]) * population)
    matched_b = np.sum((sb[common] / cb[common]) * population)
    return float(raw_a / raw_b), float(matched_a / matched_b)


def _metric_vector(frame: pd.DataFrame, temperature: str, bootstrap: bool) -> dict[str, float]:
    alpha = frame[temperature].to_numpy(dtype=np.float64)
    rejection = 1.0 - alpha
    disagreement = (frame.target_argmax_id.to_numpy() != frame.draft_argmax_id.to_numpy()).astype(np.float64)
    margin = frame.target_top2_margin.to_numpy(dtype=np.float64)
    entropy = frame.target_entropy.to_numpy(dtype=np.float64)
    low_margin20 = margin <= np.quantile(margin, .2)
    ids, doc_values = pd.factorize(frame.doc_id.astype(str), sort=True)
    n_docs = len(doc_values)
    doc_weights = np.ones(n_docs, dtype=np.float64)
    row_weights = doc_weights[ids]
    out: dict[str, float] = {
        "positions": float(len(frame)),
        "documents": float(n_docs),
        "mean_rejection": float(np.average(rejection, weights=row_weights)),
        "greedy_disagreement": float(np.average(disagreement, weights=row_weights)),
        "decision_point_share": float(np.average(margin < 1.0, weights=row_weights)),
        "decision_point_rejection_mass_share": float(rejection[margin < 1.0].sum() / max(rejection.sum(), 1e-30)),
        "low_margin20_position_share": float(low_margin20.mean()),
        "low_margin20_rejection_mass_share": float(rejection[low_margin20].sum() / max(rejection.sum(), 1e-30)),
        "spearman_rejection_target_margin": float(spearmanr(rejection, margin).statistic),
    }
    margin_bins = _quantile_bins(margin)
    entropy_bins = _quantile_bins(entropy)
    group_specs = (
        ("ptype", "morph_onset", "internal", "morpheme_internal"),
        ("ptype", "word_onset", "internal", "word_internal"),
        ("aclass", "CROSS", "WITHIN_SPLIT", "fused_split"),
    )

    def point_metrics(weights: np.ndarray) -> dict[str, float]:
        expanded = weights[ids]
        result = {
            "mean_rejection": float(np.average(rejection, weights=expanded)),
            "greedy_disagreement": float(np.average(disagreement, weights=expanded)),
            "decision_point_share": float(np.average(margin < 1.0, weights=expanded)),
        }
        result["low_margin20_position_share"] = float(
            np.sum(expanded[low_margin20]) / max(expanded.sum(), 1e-30)
        )
        for label, mask in (("decision_point", margin < 1.0),
                            ("low_margin20", low_margin20)):
            result[f"{label}_rejection_mass_share"] = float(
                np.sum(expanded[mask] * rejection[mask]) / max(np.sum(expanded * rejection), 1e-30)
            )
        for column, first, second, name in group_specs:
            group = frame[column].astype(str).to_numpy()
            eligible = (group != "excluded") if column == "ptype" else np.isin(group, ["CROSS", "WITHIN_SPLIT", "EXACT"])
            result[f"{name}_raw"] = float("nan")
            result[f"{name}_margin_matched"] = float("nan")
            result[f"{name}_entropy_matched"] = float("nan")
            if eligible.any():
                for bin_values, suffix in ((margin_bins, "margin_matched"), (entropy_bins, "entropy_matched")):
                    raw, matched = _weighted_rate_ratio(group[eligible], rejection[eligible], bin_values[eligible],
                                                        expanded[eligible], first, second)
                    result[f"{name}_raw"] = raw
                    result[f"{name}_{suffix}"] = matched
        return result

    # Use one paired document bootstrap for all scalar rates and ratios.
    rng = np.random.default_rng(SEED)
    samples = {key: [] for key in point_metrics(doc_weights)}
    if bootstrap:
        for _ in range(BOOTSTRAPS):
            sampled = rng.integers(0, n_docs, size=n_docs)
            multiplicity = np.bincount(sampled, minlength=n_docs).astype(np.float64)
            metrics = point_metrics(multiplicity)
            for key, value in metrics.items():
                if np.isfinite(value):
                    samples[key].append(value)
    point = point_metrics(doc_weights)
    for key, vals in samples.items():
        if vals:
            point[f"{key}_ci_low"] = float(np.percentile(vals, 2.5))
            point[f"{key}_ci_high"] = float(np.percentile(vals, 97.5))

    frame = frame.assign(_rejection=rejection, _disagreement=disagreement,
                         _margin_decile=margin_bins, _entropy_decile=entropy_bins)
    for decile, group in frame.groupby("_margin_decile", sort=True):
        out[f"margin_decile_{int(decile)}_mean_rejection"] = float(group._rejection.mean())
        out[f"margin_decile_{int(decile)}_greedy_disagreement"] = float(group._disagreement.mean())
        out[f"margin_decile_{int(decile)}_mean_margin"] = float(group.target_top2_margin.mean())
    return {**out, **point}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "results/rebuttal/overlap")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rebuttal/overlap")
    parser.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    decile_rows = []
    for pair, texts in TEXTS.items():
        for text in texts:
            directory = input_root / pair / text
            shards = sorted(directory.glob("shard-*.parquet"))
            if not shards:
                continue
            frame = pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True)
            frame = _align(frame, pair, text)
            for temperature in ("alpha_T1", "alpha_T07"):
                metrics = _metric_vector(frame.copy(), temperature, bootstrap=not args.no_bootstrap)
                summaries.append({"pair": pair, "continuation": text, "temperature": temperature.removeprefix("alpha_"), **metrics})
                for decile in range(10):
                    decile_rows.append({
                        "pair": pair, "continuation": text, "temperature": temperature.removeprefix("alpha_"),
                        "margin_decile": decile,
                        "mean_rejection": metrics.get(f"margin_decile_{decile}_mean_rejection"),
                        "greedy_disagreement": metrics.get(f"margin_decile_{decile}_greedy_disagreement"),
                        "mean_target_margin": metrics.get(f"margin_decile_{decile}_mean_margin"),
                    })
                print(f"ANALYZED_OVERLAP pair={pair} continuation={text} temp={temperature}", flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(decile_rows).to_csv(output_dir / "margin_deciles.csv", index=False)
    md_lines = ["# Speculative-sampling overlap", "",
                "Expected rejection is `1 − alpha`; intervals are 95% document-bootstrap intervals (500 resamples).",
                "Ratios compare rejection rates for position types and are directly standardized to the target-margin or target-entropy decile distribution.", "",
                "| Pair | Text | T | Positions | Mean rejection | Greedy disagreement | DP share | DP rejection mass | Morph/internal raw → margin matched | Word/internal raw → margin matched | Fused/split raw → margin matched |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        fmt = lambda key: f"{row[key]:.3f}" if pd.notna(row.get(key)) else "NA"
        md_lines.append(
            f"| {row['pair']} | {row['continuation']} | {row['temperature']} | {int(row['positions'])} | "
            f"{fmt('mean_rejection')} | {fmt('greedy_disagreement')} | {fmt('decision_point_share')} | "
            f"{fmt('decision_point_rejection_mass_share')} | {fmt('morpheme_internal_raw')} → {fmt('morpheme_internal_margin_matched')} | "
            f"{fmt('word_internal_raw')} → {fmt('word_internal_margin_matched')} | {fmt('fused_split_raw')} → {fmt('fused_split_margin_matched')} |"
        )
    (output_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
