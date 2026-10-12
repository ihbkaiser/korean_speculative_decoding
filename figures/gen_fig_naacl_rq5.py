#!/usr/bin/env python3
"""Render RQ5 replay cost shares and counterfactual accepted length."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BOUNDARY_ORDER = [
    "NOMINAL_TO_PARTICLE", "LEXICAL_TO_LEXICAL", "PREDICATE_TO_ENDING",
    "OTHER", "ENDING_TO_ENDING", "MULTI_BOUNDARY", "OTHER_OR_UNALIGNED",
]
BOUNDARY_LABEL = {
    "NOMINAL_TO_PARTICLE": "N→P",
    "LEXICAL_TO_LEXICAL": "Lexical→lexical",
    "PREDICATE_TO_ENDING": "Predicate→ending",
    "OTHER": "Other boundary",
    "ENDING_TO_ENDING": "Ending→ending",
    "MULTI_BOUNDARY": "Multi-boundary",
    "OTHER_OR_UNALIGNED": "Other / unaligned",
}


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

    cost = pd.read_csv(args.input_dir / "rq5_cost_decomposition.csv")
    cost = cost.loc[cost.block_size.eq(4)].set_index("boundary").reindex(BOUNDARY_ORDER)
    cf = pd.read_csv(args.input_dir / "rq5_counterfactual_accepted_length.csv").sort_values("block_size")

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.55), gridspec_kw={"width_ratios": [1.18, 1]})
    ax = axes[0]
    y = np.arange(len(BOUNDARY_ORDER))
    height = 0.34
    ax.barh(y - height / 2, 100 * cost.share_all_rejections, height,
            color="#0072B2", label="Share of rejections")
    ax.barh(y + height / 2, 100 * cost.share_all_wasted_draft_tokens, height,
            color="#E69F00", label="Share of wasted draft work")
    ax.set_yticks(y, [BOUNDARY_LABEL[x] for x in BOUNDARY_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Share of all rejections / wasted draft work (%)")
    ax.set_title("A. Cost attributed to rejection boundary (k=4)")
    ax.grid(axis="x", alpha=.2, linewidth=.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1, .88))

    ax = axes[1]
    x = cf.block_size.to_numpy()
    observed = cf.observed_mean_tokens_per_target_call.to_numpy()
    counterfactual = cf.counterfactual_mean_tokens_per_target_call.to_numpy()
    low = cf.counterfactual_ci_low_95.to_numpy()
    high = cf.counterfactual_ci_high_95.to_numpy()
    ax.plot(x, observed, marker="o", linewidth=1.7, color="#4D4D4D", label="Observed replay")
    ax.errorbar(x, counterfactual, yerr=np.vstack([counterfactual - low, high - counterfactual]),
                marker="s", linestyle="--", capsize=2.5, linewidth=1.4,
                color="#009E73", label="CROSS at WITHIN rejection rate")
    ax.set_xticks(x)
    ax.set_xlabel("Draft block size k")
    ax.set_ylabel("Target tokens per target call")
    ax.set_title("B. Counterfactual accepted length")
    ax.grid(axis="y", alpha=.2, linewidth=.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    fig.suptitle("Q2 RQ5 · 40k prompts · offline greedy-SD replay", fontsize=10.2)
    fig.text(
        0.5, -0.025,
        "Counterfactual CI resamples documents with within-boundary rates held fixed; replay does not measure live latency.",
        ha="center", fontsize=7.0,
    )
    fig.tight_layout(rect=[0, .025, 1, .94])
    fig.savefig(args.output_dir / "figure7_rq5_cost_and_accepted_length.pdf")
    fig.savefig(args.output_dir / "figure7_rq5_cost_and_accepted_length.png")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
