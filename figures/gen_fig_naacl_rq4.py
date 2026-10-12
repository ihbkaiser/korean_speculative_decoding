#!/usr/bin/env python3
"""Render RQ4 token-type ablation and pseudo-boundary falsification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/naacl_experiments/analysis")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/naacl_experiments/analysis")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.0, "axes.labelsize": 8.2, "axes.titlesize": 9.2,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    removal = pd.read_csv(args.input_dir / "rq4_top_type_removal.csv").sort_values("top_n_per_fine_boundary")
    table = pd.read_csv(args.input_dir / "rq4_table4_robustness.csv")
    baseline = table.loc[table.variant.eq("Q2_all_baseline")].iloc[0]
    pseudo = pd.read_csv(args.input_dir / "rq4_pseudo_boundary_draws.csv")
    metadata = json.loads((args.input_dir / "rq4_metadata.json").read_text(encoding="utf-8"))
    pseudo_meta = metadata["pseudo_boundary"]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": [1, 1.08]})
    ax = axes[0]
    x = np.r_[0, removal.top_n_per_fine_boundary.to_numpy(dtype=float)]
    ame = np.r_[baseline.ame_pp, removal.ame_pp.to_numpy(dtype=float)]
    low = np.r_[baseline.ci_low_pp, removal.ci_low_pp.to_numpy(dtype=float)]
    high = np.r_[baseline.ci_high_pp, removal.ci_high_pp.to_numpy(dtype=float)]
    ax.errorbar(x, ame, yerr=np.vstack([ame - low, high - ame]), marker="o", capsize=3,
                color="#0072B2", linewidth=1.5)
    ax.axhline(0, color="#555555", linewidth=.8, linestyle="--")
    ax.set_xticks(x, ["None", "10", "25", "50", "100"])
    ax.set_xlabel("Top CROSS token IDs removed per fine boundary")
    ax.set_ylabel("Adjusted CROSS−WITHIN AME (pp)")
    ax.set_title("A. Token-type removal")
    ax.grid(axis="y", alpha=.2, linewidth=.5)
    ax.set_axisbelow(True)

    ax = axes[1]
    values = pseudo.pseudo_cross_minus_within_pp.to_numpy(dtype=float)
    ax.hist(values, bins=18, color="#56B4E9", edgecolor="white", alpha=.95)
    real = float(pseudo_meta["real_cross_minus_within_pp"])
    ax.axvline(real, color="#D55E00", linewidth=2.0, label=f"Observed morphology: {real:+.2f} pp")
    q025 = float(pseudo_meta["pseudo_effect_q025_pp"])
    q975 = float(pseudo_meta["pseudo_effect_q975_pp"])
    ax.axvspan(q025, q975, color="#56B4E9", alpha=.12, label="Pseudo 95% interval")
    ax.set_xlabel("Pseudo-CROSS − pseudo-WITHIN rejection rate (pp)")
    ax.set_ylabel("Random-cut draws")
    ax.set_title("B. Random cuts preserve each eojeol's boundary count")
    ax.grid(axis="y", alpha=.2, linewidth=.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")

    fig.suptitle("Q2 RQ4 falsification · k=4 offline replay · 40k prompts", fontsize=10.2)
    fig.text(
        0.5, -0.025,
        f"Pseudo draws use strict aligned CROSS/WITHIN proposals; empirical two-sided tail probability = {pseudo_meta['empirical_two_sided_tail_probability']:.3f}.",
        ha="center", fontsize=7.0,
    )
    fig.tight_layout(rect=[0, .025, 1, .94])
    fig.savefig(args.output_dir / "figure6_rq4_falsification.pdf")
    fig.savefig(args.output_dir / "figure6_rq4_falsification.png")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
