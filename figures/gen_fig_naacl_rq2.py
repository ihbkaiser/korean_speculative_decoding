#!/usr/bin/env python3
"""Render NAACL RQ2 boundary-pair and Qwen draft-target grid figures."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


PAIR_ORDER = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "G1", "M1"]
PAIR_LABEL = {
    "Q1": "Qwen 0.6B→1.7B",
    "Q2": "Qwen 0.6B→4B",
    "Q3": "Qwen 4B→8B",
    "Q4": "Qwen 1.7B→4B",
    "Q5": "Qwen 0.6B→8B",
    "Q6": "Qwen 1.7B→8B",
    "G1": "Gemma 4B→12B",
    "M1": "Ministral 3B→8B",
}
BOUNDARY_ORDER = [
    "NOMINAL_TO_PARTICLE",
    "PREDICATE_TO_ENDING",
    "ENDING_TO_ENDING",
    "LEXICAL_TO_LEXICAL",
    "OTHER",
]
BOUNDARY_LABEL = {
    "NOMINAL_TO_PARTICLE": "N→P",
    "PREDICATE_TO_ENDING": "Predicate→ending",
    "ENDING_TO_ENDING": "Ending→ending",
    "LEXICAL_TO_LEXICAL": "Lexical→lexical",
    "OTHER": "Other",
}
MODEL_LABEL = {"qwen06": "0.6B", "qwen17": "1.7B", "qwen4": "4B", "qwen8": "8B"}
MODEL_SIZE = {"qwen06": 0.6, "qwen17": 1.7, "qwen4": 4.0, "qwen8": 8.0}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_boundary_heatmap(effects: pd.DataFrame, output_dir: Path) -> None:
    table = effects.loc[
        effects.setting.str.startswith("core_generated_k4:")
        & effects.boundary_level.eq("coarse")
        & effects.support_200_each
    ].copy()
    table["pair_id"] = table.setting.str.rsplit(":", n=1).str[-1]
    values = np.full((len(PAIR_ORDER), len(BOUNDARY_ORDER)), np.nan)
    supports: dict[tuple[int, int], tuple[int, int]] = {}
    for i, pair in enumerate(PAIR_ORDER):
        for j, boundary in enumerate(BOUNDARY_ORDER):
            row = table.loc[table.pair_id.eq(pair) & table.boundary.eq(boundary)]
            if len(row):
                values[i, j] = float(row.cross_minus_within_pp.iloc[0])
                supports[(i, j)] = (int(row.n_cross.iloc[0]), int(row.n_within.iloc[0]))
    finite = np.abs(values[np.isfinite(values)])
    limit = max(2.0, float(np.ceil(finite.max())) if len(finite) else 2.0)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    fig, ax = plt.subplots(figsize=(7.7, 4.1))
    mesh = ax.imshow(
        np.ma.masked_invalid(values),
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(BOUNDARY_ORDER)), [BOUNDARY_LABEL[x] for x in BOUNDARY_ORDER], rotation=22, ha="right")
    ax.set_yticks(np.arange(len(PAIR_ORDER)), [PAIR_LABEL[x] for x in PAIR_ORDER])
    ax.set_xlabel("Coarse Kiwi boundary family")
    ax.set_ylabel("Draft → target pair")
    ax.set_title("Generated text: k=4 replay rejection difference (CROSS − WITHIN)")
    for i in range(len(PAIR_ORDER)):
        for j in range(len(BOUNDARY_ORDER)):
            if np.isfinite(values[i, j]):
                nx, nw = supports[(i, j)]
                color = "white" if abs(values[i, j]) > limit * 0.58 else "#202020"
                ax.text(j, i, f"{values[i,j]:+.1f}\n{nx:,}/{nw:,}", ha="center", va="center", fontsize=6.7, color=color)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#777777")
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.035)
    colorbar.set_label("CROSS − WITHIN rejection (percentage points)")
    fig.text(0.5, -0.02, "Cells require ≥200 CROSS and ≥200 WITHIN proposals; counts are CROSS/WITHIN.", ha="center", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(output_dir / "figure3a_rq2_boundary_pair_heatmap.pdf")
    fig.savefig(output_dir / "figure3a_rq2_boundary_pair_heatmap.png")
    plt.close(fig)


def save_qwen_grid(grid: pd.DataFrame, output_dir: Path) -> None:
    model_ids = ["qwen06", "qwen17", "qwen4"]
    x_order = ["qwen17", "qwen4", "qwen8"]
    y_order = ["qwen06", "qwen17", "qwen4"]
    columns = [("global", "Global CROSS−WITHIN"), ("nominal_to_particle", "N→P CROSS−WITHIN")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True)
    for ax, (estimand, title) in zip(axes, columns):
        subset = grid.loc[grid.estimand.eq(estimand)]
        values = np.full((3, 3), np.nan)
        cell_text: dict[tuple[int, int], str] = {}
        for i, draft in enumerate(y_order):
            for j, target in enumerate(x_order):
                rows = subset.loc[subset.draft.eq(draft) & subset.target.eq(target)]
                if rows.empty:
                    continue
                row = rows.iloc[0]
                estimate = row.AME_cross_minus_within
                low, high = row.CI_low_95, row.CI_high_95
                if pd.notna(estimate):
                    values[i, j] = float(estimate) * 100
                    cell_text[(i, j)] = f"{estimate*100:+.2f}\n[{low*100:+.2f}, {high*100:+.2f}]"
                else:
                    cell_text[(i, j)] = "NE"
        finite = np.abs(values[np.isfinite(values)])
        limit = max(1.0, float(np.ceil(finite.max())) if len(finite) else 1.0)
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#eeeeee")
        mesh = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="equal")
        ax.set_xticks(np.arange(3), [MODEL_LABEL[x] for x in x_order])
        ax.set_yticks(np.arange(3), [MODEL_LABEL[x] for x in y_order])
        ax.set_xlabel("Target model size")
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                if (i, j) in cell_text:
                    color = "white" if np.isfinite(values[i, j]) and abs(values[i, j]) > limit * 0.58 else "#202020"
                    ax.text(j, i, cell_text[(i, j)], ha="center", va="center", fontsize=6.5, color=color)
                else:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#777777")
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Adjusted AME (pp)")
    axes[0].set_ylabel("Draft model size")
    fig.suptitle("Qwen3 generated-text grid · common 20k prompts", y=1.02, fontsize=10)
    fig.text(0.5, -0.01, "Cells show adjusted AME in percentage points [95% CI]; — = unrun same-size pair.", ha="center", fontsize=7.4)
    fig.tight_layout()
    fig.savefig(output_dir / "figure3b_rq2_qwen_grid.pdf")
    fig.savefig(output_dir / "figure3b_rq2_qwen_grid.png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/naacl_experiments/analysis")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/naacl_experiments/analysis")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    effects = pd.read_csv(args.input_dir / "rq2_boundary_effects.csv")
    grid = pd.read_csv(args.input_dir / "rq2_qwen_grid.csv")
    save_boundary_heatmap(effects, args.output_dir)
    save_qwen_grid(grid, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
