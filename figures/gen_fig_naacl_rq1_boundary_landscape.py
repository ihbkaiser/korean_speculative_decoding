#!/usr/bin/env python3
"""Render the Q2 RQ1 fine-boundary landscape as vector PDF and 300-DPI PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results/naacl_experiments/analysis/rq1_fig2_boundary_landscape.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/naacl_experiments/analysis",
    )
    args = parser.parse_args()
    table = pd.read_csv(args.input)
    table = table.loc[
        table.AME_cross_minus_within.notna()
        & table.p_value.notna()
        & table.q_value_bh.notna()
        & table.status.eq("COMPLETE")
    ].sort_values("AME_cross_minus_within")
    if table.empty:
        raise SystemExit(f"No estimable supported boundaries in {args.input}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linestyle": "-",
    })
    n = len(table)
    fig, ax = plt.subplots(figsize=(6.8, max(3.4, 0.25 * n + 1.25)))
    y = np.arange(n)
    estimate = table.AME_cross_minus_within.to_numpy(dtype=float) * 100.0
    low = table.CI_low_95.to_numpy(dtype=float) * 100.0
    high = table.CI_high_95.to_numpy(dtype=float) * 100.0
    q = table.q_value_bh.to_numpy(dtype=float)
    colors = np.where(estimate >= 0, "#007F73", "#D55E00")
    for index in range(n):
        significant = np.isfinite(q[index]) and q[index] < 0.05
        ax.errorbar(
            estimate[index], y[index],
            xerr=[[estimate[index] - low[index]], [high[index] - estimate[index]]],
            fmt="o", color=colors[index], ecolor=colors[index],
            markerfacecolor=colors[index] if significant else "white",
            markeredgecolor=colors[index], markeredgewidth=1.2,
            markersize=5, capsize=2.2, linewidth=1.35, zorder=3,
        )
    ax.axvline(0.0, color="#444444", linewidth=1, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(table.boundary.astype(str))
    ax.set_xlabel("Adjusted rejection-rate difference (percentage points; CROSS − WITHIN)")
    ax.set_ylabel("Fine Kiwi boundary")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    legend = [
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="#555555",
               markersize=5, linestyle="none", label="BH q < 0.05"),
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="white",
               markeredgewidth=1.2, markersize=5, linestyle="none", label="BH q ≥ 0.05"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, ncol=2)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf = args.output_dir / "figure2_boundary_landscape.pdf"
    png = args.output_dir / "figure2_boundary_landscape.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"FIG2_SAVED pdf={pdf} png={png} boundaries={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
