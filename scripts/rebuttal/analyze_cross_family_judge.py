#!/usr/bin/env python3
"""Summarize T6-b cross-family margins on generated text."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/rebuttal/cpu"))

from expC_helpers import verified_k4
from expD_lib import EDGES, boot_ci, ratio
from ksdlib import ROOT as NAACL_ROOT, load_pair, read_aligned, read_tf

SEED = 20261125
BOOTSTRAPS = 500
PAIRS = {
    "Q2": ("qwen4", "qwen06", "ministral8", "qwen17"),
    "Q5": ("qwen8", "qwen06", "ministral8", "qwen4"),
    "M1": ("ministral8", "ministral3", "qwen8", None),
}
CONTRASTS = {
    "morph/int": ("morph_onset", "internal", "ptype"),
    "word/int": ("word_onset", "internal", "ptype"),
    "fused/split": ("CROSS", "WITHIN_SPLIT", "aclass"),
}


def _alignment_offsets(target: str, split: str, limit: int) -> pd.DataFrame:
    directory = Path(NAACL_ROOT) / "aligned" / "kiwi" / target / split
    files = sorted(glob.glob(str(directory / "**/shard-*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No alignments under {directory}")
    parts = []
    for file in files:
        part = pd.read_parquet(
            file,
            columns=["pool_index", "generated_token_position", "global_token_char_span"],
        )
        parts.append(part[part.pool_index < limit])
    raw = pd.concat(parts, ignore_index=True)
    return pd.DataFrame({
        "doc": raw.pool_index.to_numpy(dtype=np.int64),
        "pos": raw.generated_token_position.to_numpy(dtype=np.int32),
        "cstart": [int(span[0]) if span is not None and len(span) == 2 else np.nan
                   for span in raw.global_token_char_span],
    })


def _join_judge(frame: pd.DataFrame, target: str, judge: str) -> tuple[pd.DataFrame, dict]:
    directory = ROOT / "results/rebuttal/cross_family" / target / judge
    files = sorted(directory.glob("shard-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No cross-family judge scores under {directory}")
    judge_rows = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
    judge_rows = judge_rows[judge_rows.pool_index < 5000].rename(columns={
        "pool_index": "doc", "top2_margin": "m_judge", "entropy": "H_judge",
    })
    duplicates = int(judge_rows.duplicated(["doc", "cstart"]).sum())
    judge_rows = judge_rows.drop_duplicates(["doc", "cstart"], keep="first")
    joined = frame.merge(judge_rows[["doc", "cstart", "m_judge", "H_judge", "argmax_id"]],
                         on=["doc", "cstart"], how="left", validate="many_to_one")
    return joined, {"judge_rows": int(len(judge_rows)), "duplicate_doc_offsets_dropped": duplicates}


def _contrast_mask(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column == "ptype":
        return frame.ptype.to_numpy() != "excluded"
    return frame[column].isin(["CROSS", "WITHIN_SPLIT", "EXACT"]).to_numpy()


def _margin_keys(frame: pd.DataFrame, margin: str, column: str) -> np.ndarray:
    mask = _contrast_mask(frame, column)
    key = np.full(len(frame), -1, dtype=np.int64)
    if mask.sum() < 10:
        return key
    key[np.flatnonzero(mask)] = pd.qcut(
        frame.loc[mask, margin].rank(method="first"), 10, labels=False
    ).to_numpy(dtype=np.int64)
    return key


def _one_ratio(frame: pd.DataFrame, contrast: tuple[str, str, str], margin: str, rng) -> dict:
    numerator, denominator, column = contrast
    raw_key = np.zeros(len(frame), dtype=np.int64)
    margin_key = _margin_keys(frame, margin, column)
    absolute_key = np.digitize(frame[margin].to_numpy(dtype=float), EDGES)
    point_raw = ratio(frame, numerator, denominator, column, None)
    point_decile = ratio(frame, numerator, denominator, column, margin_key)
    point_absolute = ratio(frame, numerator, denominator, column, absolute_key)
    return {
        "raw": round(point_raw[0], 3),
        "raw_coverage": round(point_raw[1], 3),
        "raw_ci95": boot_ci(frame, numerator, denominator, column, raw_key, rng, reps=BOOTSTRAPS),
        "decile_matched": round(point_decile[0], 3),
        "decile_coverage": round(point_decile[1], 3),
        "decile_ci95": boot_ci(frame, numerator, denominator, column, margin_key, rng, reps=BOOTSTRAPS),
        "absolute_bin_matched": round(point_absolute[0], 3),
        "absolute_bin_coverage": round(point_absolute[1], 3),
        "absolute_bin_ci95": boot_ci(frame, numerator, denominator, column, absolute_key, rng, reps=BOOTSTRAPS),
    }


def _auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def _analyze(pair: str, limit: int) -> dict:
    target, draft, judge, same_family = PAIRS[pair]
    frame = load_pair("generated", target, draft)
    frame = frame[frame.doc < limit].sort_values(["doc", "pos"], kind="stable").reset_index(drop=True)
    if not len(frame):
        raise RuntimeError(f"No proposal rows for {pair} under pool_index<{limit}")
    frame["rej"] = (frame.d_arg.to_numpy() != frame.y.to_numpy()).astype(np.int8)
    frame["verified_k4"] = verified_k4(frame)
    split = "all" if target == "qwen4" else "common"
    aligned = read_aligned(target, split)
    aligned = aligned[aligned.doc < limit]
    offsets = _alignment_offsets(target, split, limit)
    frame = frame.merge(aligned, on=["doc", "pos"], how="left", validate="one_to_one")
    frame = frame.merge(offsets, on=["doc", "pos"], how="left", validate="one_to_one")
    frame["ptype"] = frame.ptype.fillna("excluded")
    frame["aclass"] = frame.aclass.fillna("OTHER")
    frame = frame[frame.verified_k4].reset_index(drop=True)
    frame, join_meta = _join_judge(frame, target, judge)

    total_type = frame.groupby("ptype", dropna=False).size()
    joined_type = frame[frame.m_judge.notna()].groupby("ptype", dropna=False).size()
    coverage_by_type = {
        str(key): round(float(joined_type.get(key, 0) / value), 4)
        for key, value in total_type.items()
    }
    joined = frame[frame.m_judge.notna()].reset_index(drop=True)
    aligned_joined = joined[joined.ptype != "excluded"].reset_index(drop=True)
    rejection_rate_all = float(frame.rej.mean())
    rejection_rate_joined = float(joined.rej.mean()) if len(joined) else float("nan")

    y_join = joined.rej.to_numpy(dtype=np.int8)
    target_auc = _auc(y_join, -joined.mT.to_numpy(dtype=float))
    judge_auc = _auc(y_join, -joined.m_judge.to_numpy(dtype=float))
    boundary_auc = _auc(
        aligned_joined.rej.to_numpy(dtype=np.int8),
        (aligned_joined.ptype.to_numpy() != "internal").astype(np.int8),
    )
    corr = spearmanr(joined.mT, joined.m_judge).statistic if len(joined) > 1 else np.nan

    same_family_auc = None
    if same_family:
        same = read_tf("generated", target, same_family, cols=["pool_index", "position", "top2_margin"])
        same = same[same.pool_index < limit].rename(columns={"pool_index": "doc", "top2_margin": "m_same_family"})
        joined = joined.merge(same[["doc", "position", "m_same_family"]].rename(columns={"position": "pos"}),
                              on=["doc", "pos"], how="left", validate="one_to_one")
        same_rows = joined[joined.m_same_family.notna()]
        same_family_auc = _auc(same_rows.rej.to_numpy(dtype=np.int8), -same_rows.m_same_family.to_numpy(dtype=float))

    ratio_frame = joined[joined.ptype != "excluded"].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    ratio_summary = {}
    for name, contrast in CONTRASTS.items():
        ratio_summary[name] = {
            "judge_margin": _one_ratio(ratio_frame, contrast, "m_judge", rng),
            "target_margin": _one_ratio(ratio_frame, contrast, "mT", rng),
        }
    same_family_ratio_summary = None
    if same_family:
        # Preserve the cross-family and target-margin bootstrap streams above;
        # append same-family intervals afterward for an apples-to-apples check.
        same_family_ratio_summary = {
            name: _one_ratio(ratio_frame, contrast, "m_same_family", rng)
            for name, contrast in CONTRASTS.items()
        }

    return {
        "pair": pair,
        "source_target": target,
        "draft": draft,
        "cross_family_judge": judge,
        "same_family_outside_judge": same_family,
        "documents_requested": limit,
        "verified_k4_positions": int(len(frame)),
        "verified_k4_documents": int(frame.doc.nunique()),
        "joined_positions": int(len(joined)),
        "join_coverage": round(float(len(joined) / len(frame)), 4),
        "join_coverage_by_position_type": coverage_by_type,
        "duplicate_judge_offsets_dropped": join_meta["duplicate_doc_offsets_dropped"],
        "rejection_rate_all_verified_proposals": round(rejection_rate_all, 5),
        "rejection_rate_joined_subset": round(rejection_rate_joined, 5),
        "auc_cross_family_judge_margin": round(judge_auc, 4) if judge_auc is not None else None,
        "auc_target_margin_on_joined_subset": round(target_auc, 4) if target_auc is not None else None,
        "auc_same_family_outside_judge": round(same_family_auc, 4) if same_family_auc is not None else None,
        "auc_boundary_label": round(boundary_auc, 4) if boundary_auc is not None else None,
        "spearman_target_judge_margin": round(float(corr), 4) if np.isfinite(corr) else None,
        "ratios_on_cross_family_joined_subset": ratio_summary,
        "ratios_on_same_family_joined_subset": same_family_ratio_summary,
        "internal_alignment_rows_in_join": int((joined.ptype == "internal").sum()),
        "excluded_alignment_rows_in_join": int((joined.ptype == "excluded").sum()),
    }


def _markdown(results: list[dict]) -> str:
    lines = [
        "# T6-b: cross-family judge on generated text",
        "",
        "All statistics use proposals verified under k=4 and the first 5,000 common documents.",
        "Ratios are morpheme-onset/internal, word-onset/internal, and fused/split.",
        "Matched rows use judge or target margin deciles, plus judge absolute bins from `expD_lib.EDGES`.",
        "Intervals are 95% document-bootstrap intervals (500 resamples; seed 20261125).",
        "",
        "| Pair | Shared-offset coverage | Rejection, all / joined | AUC judge | AUC target | AUC boundary | AUC same-family judge |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        same = "—" if row["auc_same_family_outside_judge"] is None else f"{row['auc_same_family_outside_judge']:.3f}"
        lines.append(
            f"| {row['pair']} | {row['join_coverage']:.3f} | "
            f"{row['rejection_rate_all_verified_proposals']:.3f} / {row['rejection_rate_joined_subset']:.3f} | "
            f"{row['auc_cross_family_judge_margin']:.3f} | {row['auc_target_margin_on_joined_subset']:.3f} | "
            f"{row['auc_boundary_label']:.3f} | {same} |"
        )
    lines.extend(["", "## Position-type ratios", ""])
    for row in results:
        lines.extend([
            f"### {row['pair']} ({row['source_target']} judged by {row['cross_family_judge']})",
            "",
            "| Contrast | Raw [95% CI] | Cross-family decile [95% CI] | Cross-family absolute bin [95% CI] | Same-family decile [95% CI] | Target decile [95% CI] |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for name, vals in row["ratios_on_cross_family_joined_subset"].items():
            j = vals["judge_margin"]
            t = vals["target_margin"]
            fmt = lambda value, ci: f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
            same_summary = row["ratios_on_same_family_joined_subset"]
            same = "—" if same_summary is None else fmt(
                same_summary[name]["decile_matched"], same_summary[name]["decile_ci95"]
            )
            lines.append(
                f"| {name} | {fmt(j['raw'], j['raw_ci95'])} | "
                f"{fmt(j['decile_matched'], j['decile_ci95'])} | "
                f"{fmt(j['absolute_bin_matched'], j['absolute_bin_ci95'])} | "
                f"{same} | {fmt(t['decile_matched'], t['decile_ci95'])} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-prompts", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rebuttal/cross_family")
    args = parser.parse_args()
    if args.max_prompts < 1:
        parser.error("--max-prompts must be positive")
    results = [_analyze(pair, args.max_prompts) for pair in ("Q2", "Q5", "M1")]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "max_prompts": args.max_prompts,
        "results": results,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(_markdown(results), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
