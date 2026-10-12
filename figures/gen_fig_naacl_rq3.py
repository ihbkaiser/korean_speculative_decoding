#!/usr/bin/env python3
"""Render RQ3 Q2 token-geometry, rank, entropy, and particle diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COARSE = ["NOMINAL_TO_PARTICLE", "PREDICATE_TO_ENDING", "ENDING_TO_ENDING", "LEXICAL_TO_LEXICAL", "OTHER"]
COARSE_LABEL = {
    "NOMINAL_TO_PARTICLE": "N→P", "PREDICATE_TO_ENDING": "Predicate→ending",
    "ENDING_TO_ENDING": "Ending→ending", "LEXICAL_TO_LEXICAL": "Lexical→lexical", "OTHER": "Other",
}
LOCUS = ["0–25%", "25–50%", "50–75%", "75–100%"]
RANK = ["1", "2–5", "6–10", "11–20", ">20"]
PARTICLES = ["TOPIC", "SUBJECT", "OBJECT", "ADDITIVE", "LIMITER", "ADVERBIAL", "OTHER_PARTICLE", "NO_PARTICLE_IN_DRAFT_TOKEN"]
PARTICLE_LABEL = {
    "TOPIC": "Topic", "SUBJECT": "Subject", "OBJECT": "Object", "ADDITIVE": "Additive",
    "LIMITER": "Limiter", "ADVERBIAL": "Adverbial", "OTHER_PARTICLE": "Other particle",
    "NO_PARTICLE_IN_DRAFT_TOKEN": "No particle in draft token",
}
COLORS = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#CC79A7"]


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.2, "axes.labelsize": 8.6, "axes.titlesize": 9.4,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/naacl_experiments/analysis")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/naacl_experiments/analysis")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()

    locus = pd.read_csv(args.input_dir / "rq3_affix_locus_by_boundary.csv")
    locus = locus.groupby(["coarse_boundary", "locus_bin"], as_index=False).n.sum()
    locus["share"] = locus.n / locus.groupby("coarse_boundary").n.transform("sum")
    locus_wide = locus.pivot(index="coarse_boundary", columns="locus_bin", values="share").reindex(index=COARSE, columns=LOCUS).fillna(0)

    rank = pd.read_csv(args.input_dir / "rq3_rejected_draft_target_rank.csv")
    rank = rank.loc[rank.target_consistent.astype(bool) & rank.class_group.eq("CROSS")]
    rank_wide = rank.pivot_table(index="boundary", columns="rank_bucket", values="share", aggfunc="sum").reindex(index=COARSE, columns=RANK).fillna(0)

    entropy = pd.read_csv(args.input_dir / "rq3_rejection_entropy.csv")
    entropy = entropy.loc[entropy.target_consistent.astype(bool) & entropy.class_group.isin(["CROSS", "WITHIN"])]
    entropy = entropy.loc[entropy.boundary.isin(COARSE)]

    confusion = pd.read_csv(args.input_dir / "rq3_particle_confusion.csv")
    confusion_wide = confusion.pivot_table(index="target_particle", columns="draft_particle", values="share_within_target", aggfunc="sum").reindex(index=PARTICLES[:-1], columns=PARTICLES).fillna(0)
    target_n = confusion.groupby("target_particle").n.sum().to_dict()

    fig, axes = plt.subplots(2, 2, figsize=(9.1, 6.7), gridspec_kw={"height_ratios": [1, 1.15]})
    ax = axes[0, 0]
    y = np.arange(len(COARSE))
    left = np.zeros(len(COARSE))
    for j, category in enumerate(LOCUS):
        values = locus_wide[category].to_numpy()
        ax.barh(y, values, left=left, color=COLORS[j], label=category, edgecolor="white", linewidth=.4)
        left += values
    ax.set_yticks(y, [COARSE_LABEL[x] for x in COARSE])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of single-boundary CROSS tokens")
    ax.set_title("A. Morpheme-boundary locus inside token")
    ax.legend(title="Boundary position", fontsize=6.6, title_fontsize=6.8, ncol=2, loc="lower right", frameon=False)

    ax = axes[0, 1]
    y = np.arange(len(COARSE))
    left = np.zeros(len(COARSE))
    rank_colors = ["#4C78A8", "#72A0C1", "#A8C8D8", "#F2B66D", "#D55E00"]
    for j, category in enumerate(RANK):
        values = rank_wide[category].to_numpy()
        ax.barh(y, values, left=left, color=rank_colors[j], label=category, edgecolor="white", linewidth=.4)
        left += values
    ax.set_yticks(y, [COARSE_LABEL[x] for x in COARSE])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share among target-consistent CROSS rejections")
    ax.set_title("B. Target rank of rejected draft argmax")
    ax.legend(title="Rank in target top-20", fontsize=6.6, title_fontsize=6.8, ncol=2, loc="lower right", frameon=False)

    ax = axes[1, 0]
    offsets = {"CROSS": -0.10, "WITHIN": 0.10}
    colors = {"CROSS": "#D55E00", "WITHIN": "#0072B2"}
    for class_group in ["CROSS", "WITHIN"]:
        sub = entropy.loc[entropy.class_group.eq(class_group)].set_index("boundary")
        for i, boundary in enumerate(COARSE):
            if boundary not in sub.index:
                continue
            row = sub.loc[boundary]
            center = float(row["median"])
            low, high = float(row["q25"]), float(row["q75"])
            yy = i + offsets[class_group]
            ax.plot([low, high], [yy, yy], color=colors[class_group], linewidth=2.2, solid_capstyle="round")
            ax.scatter([center], [yy], color=colors[class_group], s=22, zorder=3, label=class_group if i == 0 else None)
    ax.set_yticks(np.arange(len(COARSE)), [COARSE_LABEL[x] for x in COARSE])
    ax.invert_yaxis()
    ax.set_xlabel("Draft entropy at rejected proposal (nats; median and IQR)")
    ax.set_title("C. Draft uncertainty at rejection")
    ax.legend(frameon=False, loc="lower right", fontsize=7)

    ax = axes[1, 1]
    image = ax.imshow(confusion_wide.to_numpy(), cmap="Blues", vmin=0, vmax=max(.05, float(confusion_wide.to_numpy().max())), aspect="auto")
    ax.set_xticks(np.arange(len(PARTICLES)), [PARTICLE_LABEL[x] for x in PARTICLES], rotation=37, ha="right", fontsize=6.5)
    target_categories = list(PARTICLES[:-1])
    ax.set_yticks(np.arange(len(target_categories)), [f"{PARTICLE_LABEL[x]} (n={int(target_n.get(x, 0)):,})" for x in target_categories], fontsize=6.2)
    ax.set_xlabel("Draft candidate particle (Kiwi on nominal + decoded token)")
    ax.set_ylabel("Target continuation particle")
    ax.set_title("D. Particle categories at rejected N→P fusions")
    for i in range(len(target_categories)):
        for j in range(len(PARTICLES)):
            value = confusion_wide.iloc[i, j]
            if value > 0:
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=5.5,
                        color="white" if value > confusion_wide.to_numpy().max() * .55 else "#222222")
    colorbar = fig.colorbar(image, ax=ax, fraction=.047, pad=.025)
    colorbar.set_label("Within-target-particle share")

    fig.suptitle("Q2 token-level diagnostics · k=4 offline replay · 40k prompts", fontsize=10.2)
    fig.text(0.5, -0.015, "Rank and particle panels exclude positions where target TF argmax differs from the saved continuation token. Rank >20 is right-censored.", ha="center", fontsize=7.2)
    fig.tight_layout(rect=[0, .02, 1, .97])
    fig.savefig(args.output_dir / "figure5_rq3_token_diagnostics.pdf")
    fig.savefig(args.output_dir / "figure5_rq3_token_diagnostics.png")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
