#!/usr/bin/env python3
"""Create publication-ready RQ6 margin, overlap, sharing, and morphology figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.naacl_experiment import load_config


MODEL_ORDER = ("qwen4", "qwen8", "qwen14", "gemma12", "ministral8")
LABELS = {
    "qwen4": "Qwen3-4B",
    "qwen8": "Qwen3-8B",
    "qwen14": "Qwen3-14B",
    "gemma12": "Gemma3-12B",
    "ministral8": "Ministral3-8B",
}


def _save(fig, output_dir: Path, stem: str):
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_a(long: pd.DataFrame, output_dir: Path):
    primary = long.loc[long.analysis_set.eq("eojeol_boundary")]
    wide = primary.pivot(index="position_id", columns="model_key", values="target_top2_margin_nats").reindex(columns=MODEL_ORDER)
    ranks = wide.rank(method="average", pct=True)
    pairs = [("qwen4", "qwen8"), ("qwen8", "gemma12"), ("qwen8", "ministral8")]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), constrained_layout=True)
    hexes = []
    for ax, (model_a, model_b) in zip(axes, pairs):
        points = ranks[[model_a, model_b]].dropna()
        hb = ax.hexbin(
            points[model_a], points[model_b], gridsize=38, mincnt=1,
            bins="log", cmap="magma", linewidths=0,
        )
        hexes.append(hb)
        ax.plot([0, 1], [0, 1], color="#4b5563", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{LABELS[model_a]} vs {LABELS[model_b]}", fontsize=10)
        ax.set_xlabel(f"{LABELS[model_a]} margin percentile")
        ax.set_ylabel(f"{LABELS[model_b]} margin percentile")
    fig.colorbar(hexes[-1], ax=axes, shrink=0.82, label="Log10 position count")
    fig.suptitle("Figure A. Cross-target margin agreement on identical prefixes", fontsize=13)
    _save(fig, output_dir, "figure_A_margin_agreement")


def figure_b(overlap: pd.DataFrame, output_dir: Path):
    selected = overlap.loc[overlap.threshold_fraction.eq(0.20)]
    matrix = pd.DataFrame(np.eye(len(MODEL_ORDER)), index=MODEL_ORDER, columns=MODEL_ORDER)
    for row in selected.itertuples():
        matrix.loc[row.model_a, row.model_b] = row.jaccard
        matrix.loc[row.model_b, row.model_a] = row.jaccard
    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(MODEL_ORDER)), [LABELS[key] for key in MODEL_ORDER], rotation=40, ha="right")
    ax.set_yticks(range(len(MODEL_ORDER)), [LABELS[key] for key in MODEL_ORDER])
    for i in range(len(MODEL_ORDER)):
        for j in range(len(MODEL_ORDER)):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if value < 0.58 else "black", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.82, label="Jaccard similarity")
    ax.set_title("Figure B. Bottom-20% decision-point overlap")
    _save(fig, output_dir, "figure_B_pairwise_overlap")


def figure_c(shared: pd.DataFrame, output_dir: Path):
    table = shared.sort_values("n_targets_identifying_position")
    x = table.n_targets_identifying_position.to_numpy(dtype=int)
    observed = table.observed_fraction.to_numpy(dtype=float)
    null = table.permutation_mean_fraction.to_numpy(dtype=float)
    low = table.permutation_ci_low_95.to_numpy(dtype=float)
    high = table.permutation_ci_high_95.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    width = 0.36
    ax.bar(x - width / 2, observed, width, color="#2563eb", label="Observed")
    ax.bar(x + width / 2, null, width, color="#9ca3af", label="Within-document permutation mean")
    ax.errorbar(x + width / 2, null, yerr=[null - low, high - null], fmt="none", ecolor="#374151", capsize=3)
    ax.set_xticks(x, ["0", "1", "2", "3", "4", "5"])
    ax.set_xlabel("Number of targets marking a position as a decision point")
    ax.set_ylabel("Fraction of positions")
    ax.set_title("Figure C. Shared decision points vs permutation reference")
    ax.legend(frameon=False)
    _save(fig, output_dir, "figure_C_shared_decision_points")


def figure_d(morph: pd.DataFrame, output_dir: Path):
    if morph.empty:
        fig, ax = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
        ax.text(0.5, 0.5, "No supported Kiwi matched-position analysis", ha="center", va="center")
        ax.axis("off")
        ax.set_title("Figure D. Morphological enrichment (unsupported)")
        _save(fig, output_dir, "figure_D_morphological_enrichment")
        return
    labels = {
        "exactly_one_target": "Exactly one target",
        "shared_by_three_or_more": "3+ targets",
        "cross_family_shared": "All 3 families",
    }
    table = morph.set_index("outcome").reindex(labels.keys())
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    x = np.arange(len(table))
    differences = table.boundary_minus_internal.to_numpy(dtype=float)
    lows = table.document_bootstrap_ci_low_95.to_numpy(dtype=float)
    highs = table.document_bootstrap_ci_high_95.to_numpy(dtype=float)
    ax.axhline(0, color="#4b5563", linewidth=1)
    ax.errorbar(
        x, differences,
        yerr=[differences - lows, highs - differences],
        fmt="o", color="#b45309", ecolor="#92400e", capsize=4, markersize=7,
    )
    ax.set_xticks(x, [labels[key] for key in table.index])
    ax.set_ylabel("Boundary minus within-morpheme prevalence")
    ax.set_title("Figure D. Decision-point association with Kiwi morphology")
    ax.text(0.01, -0.24, "Matched within the same eojeol; document-bootstrap 95% CIs.", transform=ax.transAxes, fontsize=9)
    _save(fig, output_dir, "figure_D_morphological_enrichment")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cross_target_dp.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = ROOT / config["paths"]["outputs"]
    long = pd.read_parquet(output_dir / "position_margins.parquet")
    figure_a(long, output_dir)
    figure_b(pd.read_csv(output_dir / "pairwise_overlap.csv"), output_dir)
    figure_c(pd.read_csv(output_dir / "shared_target_counts.csv"), output_dir)
    figure_d(pd.read_csv(output_dir / "morphological_enrichment.csv"), output_dir)
    print(f"FIGURES_READY directory={output_dir.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
