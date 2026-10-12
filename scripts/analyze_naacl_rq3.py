#!/usr/bin/env python3
"""RQ3 token-level diagnostics from Q2's saved k=4 replay and TF scores."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from kiwipiepy import Kiwi

from analyze_naacl_cpu import EVENT_COLUMNS
from src.naacl_experiment import atomic_json, load_config, model_path


ALIGN_COLUMNS = [
    "doc_id", "generated_token_position", "alignment_class", "alignment_status",
    "visible_candidate", "crossed_boundary_count", "fine_boundary", "coarse_boundary",
    "nearest_fine_boundary", "nearest_coarse_boundary", "boundary_position_ratio",
    "morpheme_surfaces", "morpheme_pos_tags",
]
LOCUS_COLUMNS = [
    "alignment_class", "alignment_status", "visible_candidate", "crossed_boundary_count",
    "coarse_boundary", "fine_boundary", "boundary_position_ratio",
]
PARTICLE_CATEGORIES = ["TOPIC", "SUBJECT", "OBJECT", "ADDITIVE", "LIMITER", "ADVERBIAL", "OTHER_PARTICLE"]
RANK_BUCKETS = ["1", "2–5", "6–10", "11–20", ">20"]
LOCUS_BINS = ["0–25%", "25–50%", "50–75%", "75–100%"]
COARSE_ORDER = ["NOMINAL_TO_PARTICLE", "PREDICATE_TO_ENDING", "ENDING_TO_ENDING", "LEXICAL_TO_LEXICAL", "OTHER"]
COARSE_LABEL = {
    "NOMINAL_TO_PARTICLE": "N→P",
    "PREDICATE_TO_ENDING": "Predicate→ending",
    "ENDING_TO_ENDING": "Ending→ending",
    "LEXICAL_TO_LEXICAL": "Lexical→lexical",
    "OTHER": "Other",
}
PARTICLE_SURFACE_CATEGORY = {
    "은": "TOPIC", "는": "TOPIC",
    "이": "SUBJECT", "가": "SUBJECT",
    "을": "OBJECT", "를": "OBJECT",
    "도": "ADDITIVE", "만": "LIMITER",
}
ADVERBIAL_SURFACES = {"에", "에서", "에게", "께", "으로", "로", "와", "과", "보다", "부터", "까지", "처럼", "만큼"}


def read_alignment_shards(directory: Path) -> pd.DataFrame:
    """Aggregate the boundary locus across all single-boundary CROSS tokens."""
    counts: list[pd.DataFrame] = []
    for path in sorted(directory.glob("shard-*.parquet")):
        frame = pd.read_parquet(path, columns=LOCUS_COLUMNS)
        mask = (
            frame.alignment_status.eq("ALIGNED")
            & frame.alignment_class.eq("CROSS")
            & frame.visible_candidate.fillna(False).astype(bool)
            & frame.crossed_boundary_count.eq(1)
            & frame.boundary_position_ratio.notna()
        )
        selected = frame.loc[mask, ["fine_boundary", "coarse_boundary", "boundary_position_ratio"]].copy()
        selected["locus_bin"] = pd.cut(
            selected.boundary_position_ratio.astype(float),
            bins=[0, 0.25, 0.5, 0.75, 1],
            labels=LOCUS_BINS,
            include_lowest=True,
        )
        counts.append(
            selected.groupby(["coarse_boundary", "fine_boundary", "locus_bin"], observed=True)
            .size().rename("n").reset_index()
        )
    grouped = pd.concat(counts, ignore_index=True).groupby(
        ["coarse_boundary", "fine_boundary", "locus_bin"], as_index=False, observed=True
    )["n"].sum()
    totals = grouped.groupby(["coarse_boundary", "fine_boundary"], observed=True).n.transform("sum")
    grouped["share"] = grouped.n / totals
    return grouped


def read_selected_rows(directory: Path, keys: pd.MultiIndex, columns: list[str], pos_col: str) -> pd.DataFrame:
    frames = []
    for path in sorted(directory.glob("shard-*.parquet")):
        frame = pd.read_parquet(path, columns=columns)
        frame["doc_id"] = frame.doc_id.astype(str)
        frame["position"] = frame[pos_col].astype("int64")
        if pos_col != "position":
            frame = frame.drop(columns=[pos_col])
        index = pd.MultiIndex.from_arrays([frame.doc_id, frame.position])
        selected = frame.loc[index.isin(keys)]
        if len(selected):
            frames.append(selected)
    if not frames:
        return pd.DataFrame(columns=["doc_id", "position", *[c for c in columns if c not in {"doc_id", pos_col}]])
    return pd.concat(frames, ignore_index=True)


def read_rejection_alignment(root: Path, keys: pd.MultiIndex) -> pd.DataFrame:
    frames = []
    directory = root / "aligned" / "kiwi" / "qwen4" / "all"
    for path in sorted(directory.glob("shard-*.parquet")):
        frame = pd.read_parquet(path, columns=ALIGN_COLUMNS)
        frame["doc_id"] = frame.doc_id.astype(str)
        frame["position"] = frame.generated_token_position.astype("int64")
        index = pd.MultiIndex.from_arrays([frame.doc_id, frame.position])
        selected = frame.loc[index.isin(keys)].drop(columns=["generated_token_position"])
        if len(selected):
            frames.append(selected)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def target_rank(top20: Any, token_id: int) -> int:
    if top20 is None:
        return 21
    try:
        for i, value in enumerate(top20):
            if int(value) == int(token_id):
                return i + 1
    except (TypeError, ValueError):
        return 21
    return 21


def rank_bucket(rank: int) -> str:
    if rank == 1:
        return "1"
    if rank <= 5:
        return "2–5"
    if rank <= 10:
        return "6–10"
    if rank <= 20:
        return "11–20"
    return ">20"


def effective_coarse(frame: pd.DataFrame) -> pd.Series:
    cross = frame.alignment_class.eq("CROSS")
    return frame.coarse_boundary.where(cross, frame.nearest_coarse_boundary).astype("string")


def target_particle(row: pd.Series) -> tuple[str, str]:
    surfaces, tags = row.morpheme_surfaces, row.morpheme_pos_tags
    if surfaces is None or tags is None:
        return "UNMAPPED_TARGET", ""
    try:
        surfaces, tags = list(surfaces), list(tags)
    except TypeError:
        return "UNMAPPED_TARGET", ""
    if len(surfaces) != len(tags):
        return "UNMAPPED_TARGET", ""
    boundary = str(row.fine_boundary or "")
    if "→" not in boundary:
        return "UNMAPPED_TARGET", ""
    left_tag, right_tag = boundary.split("→", 1)
    for i in range(min(len(tags), len(surfaces)) - 1):
        if str(tags[i]) == left_tag and str(tags[i + 1]) == right_tag:
            surface = str(surfaces[i + 1])
            return classify_particle(surface, right_tag), str(surfaces[i])
    return "UNMAPPED_TARGET", ""


def classify_particle(surface: str, pos: str) -> str:
    surface = surface.strip()
    if surface in PARTICLE_SURFACE_CATEGORY:
        return PARTICLE_SURFACE_CATEGORY[surface]
    if pos == "JKB" or surface in ADVERBIAL_SURFACES:
        return "ADVERBIAL"
    if str(pos).startswith("J"):
        return "OTHER_PARTICLE"
    return "UNMAPPED_TARGET"


def draft_particle(candidate: str, nominal: str, kiwi: Kiwi) -> str:
    candidate = candidate.strip()
    if candidate in PARTICLE_SURFACE_CATEGORY:
        return PARTICLE_SURFACE_CATEGORY[candidate]
    if candidate in ADVERBIAL_SURFACES:
        return "ADVERBIAL"
    parsed = kiwi.tokenize((nominal + candidate) if nominal else candidate)
    particles = [token for token in parsed if str(token.tag).startswith("J")]
    if particles:
        last = particles[-1]
        return classify_particle(str(last.form), str(last.tag))
    return "NO_PARTICLE_IN_DRAFT_TOKEN"


def slope_table(grid_path: Path) -> pd.DataFrame:
    grid = pd.read_csv(grid_path)
    size = {"qwen06": 0.6, "qwen17": 1.7, "qwen4": 4.0, "qwen8": 8.0}
    rows = []
    for target, subset in grid.groupby("target"):
        for estimand, part in subset.groupby("estimand"):
            x = np.array([size[draft] for draft in part.draft], dtype=float)
            y = part.AME_cross_minus_within.to_numpy(dtype=float) * 100
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            if len(x) >= 2:
                slope, intercept = np.polyfit(x, y, 1)
                fitted = intercept + slope * x
                r2 = 1 - float(np.square(y - fitted).sum() / np.square(y - y.mean()).sum()) if np.square(y-y.mean()).sum() > 0 else float("nan")
                status = "DESCRIPTIVE_SLOPE"
            else:
                slope = intercept = r2 = float("nan")
                status = "NOT_ESTIMABLE_LT2_DRAFT_SIZES"
            rows.append({
                "target": target,
                "target_size_b": size[target],
                "estimand": estimand,
                "n_draft_sizes": int(len(x)),
                "draft_sizes_b": ";".join(f"{v:g}" for v in sorted(x)),
                "slope_pp_per_draft_billion": float(slope),
                "intercept_pp": float(intercept),
                "r_squared_descriptive": float(r2),
                "status": status,
            })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/naacl_experiments")
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "results/naacl_experiments/analysis")
    args = parser.parse_args()
    root = args.output_root
    out = args.analysis_dir
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)

    locus = read_alignment_shards(root / "aligned" / "kiwi" / "qwen4" / "all")
    locus.to_csv(out / "rq3_affix_locus_by_boundary.csv", index=False)
    print(f"AFFIX_LOCUS single_boundary_cross_rows={int(locus.n.sum()):,} fine_boundaries={locus.fine_boundary.nunique()}", flush=True)

    events_path = root / "simulated" / "Q2" / "all" / "events_k4.parquet"
    events = pd.read_parquet(events_path, columns=EVENT_COLUMNS)
    events = events.loc[events.eligible.fillna(False).astype(bool) & events.rejected.fillna(False).astype(bool)].copy()
    events["doc_id"] = events.doc_id.astype(str)
    events["position"] = events.position.astype("int64")
    if events.duplicated(["doc_id", "position"]).any():
        raise AssertionError("Rejected Q2 event keys are not unique")
    keys = pd.MultiIndex.from_frame(events[["doc_id", "position"]])
    alignment = read_rejection_alignment(root, keys)
    joined = events.merge(alignment, on=["doc_id", "position"], how="left", validate="one_to_one", indicator="alignment_join")
    target_dir = root / "teacher_forced" / "generated" / "qwen4" / "qwen4" / "all"
    draft_dir = root / "teacher_forced" / "generated" / "qwen4" / "qwen06" / "all"
    target_tf = read_selected_rows(target_dir, keys, ["doc_id", "position", "top20_ids"], "position")
    draft_tf = read_selected_rows(draft_dir, keys, ["doc_id", "position", "entropy"], "position")
    joined = joined.merge(target_tf, on=["doc_id", "position"], how="left", validate="one_to_one")
    joined = joined.merge(draft_tf, on=["doc_id", "position"], how="left", validate="one_to_one")
    if joined.entropy.isna().any() or joined.top20_ids.isna().any():
        raise AssertionError("Missing target/draft TF rows at rejected Q2 positions")
    joined["target_consistent"] = joined.target_argmax_id.astype("int64").eq(joined.target_token_id.astype("int64"))
    joined["candidate_rank"] = [target_rank(ids, int(token)) for ids, token in zip(joined.top20_ids, joined.draft_argmax_id)]
    joined["rank_bucket"] = joined.candidate_rank.map(rank_bucket)
    joined["boundary"] = effective_coarse(joined).fillna("UNALIGNED_OR_OTHER")
    joined["class_group"] = np.select(
        [joined.alignment_join.eq("both") & joined.alignment_class.eq("CROSS"), joined.alignment_join.eq("both") & joined.alignment_class.eq("WITHIN_SPLIT")],
        ["CROSS", "WITHIN"],
        default="UNALIGNED_OR_OTHER",
    )

    rank = joined.groupby(["boundary", "class_group", "target_consistent", "rank_bucket"], observed=True).size().rename("n").reset_index()
    rank["share"] = rank.n / rank.groupby(["boundary", "class_group", "target_consistent"]).n.transform("sum")
    rank.to_csv(out / "rq3_rejected_draft_target_rank.csv", index=False)
    entropy = joined.groupby(["boundary", "class_group", "target_consistent"], observed=True).entropy.agg(
        n="size", mean="mean", median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75)
    ).reset_index()
    entropy.to_csv(out / "rq3_rejection_entropy.csv", index=False)
    entropy_overall = joined.groupby(["class_group", "target_consistent"], observed=True).entropy.agg(
        n="size", mean="mean", median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75)
    ).reset_index()
    entropy_overall.to_csv(out / "rq3_rejection_entropy_overall.csv", index=False)
    print(f"REJECTIONS n={len(joined):,} aligned={int(joined.alignment_join.eq('both').sum()):,}", flush=True)

    config_tokenizer = model_path(config, "qwen06")
    tokenizer = AutoTokenizer.from_pretrained(str(config_tokenizer), use_fast=True, local_files_only=True)
    kiwi = Kiwi()
    confusion_rows = []
    n2p = joined.loc[
        joined.alignment_join.eq("both")
        & joined.alignment_status.eq("ALIGNED")
        & joined.visible_candidate.fillna(False).astype(bool)
        & joined.alignment_class.eq("CROSS")
        & joined.coarse_boundary.eq("NOMINAL_TO_PARTICLE")
        & joined.crossed_boundary_count.eq(1)
        & joined.target_consistent
    ]
    for row in n2p.itertuples(index=False):
        series = pd.Series(row._asdict())
        gold_category, nominal = target_particle(series)
        decoded = tokenizer.decode([int(row.draft_argmax_id)], clean_up_tokenization_spaces=False, skip_special_tokens=False)
        draft_category = draft_particle(decoded, nominal, kiwi)
        confusion_rows.append({"target_particle": gold_category, "draft_particle": draft_category})
    confusion = pd.DataFrame(confusion_rows)
    if len(confusion):
        matrix = confusion.groupby(["target_particle", "draft_particle"], dropna=False).size().rename("n").reset_index()
        matrix["share_within_target"] = matrix.n / matrix.groupby("target_particle").n.transform("sum")
    else:
        matrix = pd.DataFrame(columns=["target_particle", "draft_particle", "n", "share_within_target"])
    matrix.to_csv(out / "rq3_particle_confusion.csv", index=False)
    print(f"PARTICLE_CONFUSION target_consistent_n2p_cross_rejections={len(n2p):,} mapped_targets={int((confusion.target_particle != 'UNMAPPED_TARGET').sum()) if len(confusion) else 0:,}", flush=True)

    slopes = slope_table(out / "rq2_qwen_grid.csv")
    slopes.to_csv(out / "rq3_qwen_grid_draft_size_trends.csv", index=False)
    metadata = {
        "outcome": "Q2 k=4 offline greedy-SD replay rejections on the 40k all split; see V0 shared-context scope diagnostic",
        "aligner": "Kiwi token-to-morpheme alignment on saved Qwen4 target continuation",
        "affix_locus": "single-boundary CROSS tokens with visible aligned candidate; boundary_position_ratio is chars to the left of the internal morpheme boundary divided by token characters; bins are quartiles",
        "rank": "rank of rejected Qwen0.6B draft argmax in Qwen4 target top20; >20 is right-censored because only target top20 IDs were saved",
        "target_consistency": "The event target token is compared with the teacher-forced target argmax. Primary rank plot and particle confusion use target-consistent positions; CSVs also retain target-inconsistent positions as a separate diagnostic group.",
        "entropy": "draft entropy at rejected position from teacher-forced Qwen0.6B scores",
        "particle_confusion": "For single-boundary N→P CROSS rejections, target particle comes from the aligned target eojeol's actual Kiwi morpheme pair. Draft candidate ID is decoded with the pinned local Qwen0.6B tokenizer; Kiwi analyzes nominal+candidate surface and the final particle is grouped into the prespecified categories. This is a token-surface diagnostic, not target-logit decoding.",
        "grid_slope": "descriptive OLS slope of adjusted AME in percentage points against draft parameter size in billions, separately by target and estimand; no inferential CI because there are only 1–3 draft sizes per target",
        "rejected_events": int(len(joined)),
        "rejected_events_with_alignment": int(joined.alignment_join.eq("both").sum()),
        "target_consistent_rejections": int(joined.target_consistent.sum()),
        "target_inconsistent_rejections": int((~joined.target_consistent).sum()),
        "n2p_cross_rejections": int(len(n2p)),
        "tokenizer_path": str(config_tokenizer),
        "outputs": [
            "rq3_affix_locus_by_boundary.csv", "rq3_rejected_draft_target_rank.csv",
            "rq3_rejection_entropy.csv", "rq3_particle_confusion.csv",
            "rq3_rejection_entropy_overall.csv", "rq3_qwen_grid_draft_size_trends.csv",
        ],
    }
    atomic_json(out / "rq3_metadata.json", metadata)
    del joined, events, alignment, target_tf, draft_tf
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
