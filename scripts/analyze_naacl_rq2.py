#!/usr/bin/env python3
"""RQ2 stability analyses for generated, Wikipedia, and human Korean text.

Core and Wikipedia outcomes are k=4 offline greedy-SD replay rejections.
Human outcomes are teacher-forced draft/target argmax disagreements on the
same human-written prefix, so those rows are deliberately labeled separately.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyze_naacl_cpu import (
    EVENT_COLUMNS,
    PAIR_ALIGN_COLUMNS,
    eligible_cross_within,
    fit_cross_ame,
    load_pair_frame,
    read_shards,
    read_tf_for_pair,
)
from src.naacl_experiment import atomic_json, load_config


COARSE_BOUNDARY_LABEL = {
    "NOMINAL_TO_PARTICLE": "N→P",
    "PREDICATE_TO_ENDING": "Predicate→Ending",
    "ENDING_TO_ENDING": "Ending→Ending",
    "LEXICAL_TO_LEXICAL": "Lexical→Lexical",
    "OTHER": "Other",
}
ALIGNMENT_COLUMNS = list(dict.fromkeys(PAIR_ALIGN_COLUMNS + ["coarse_boundary"]))
MIN_SUPPORT_TABLE = 100
MIN_SUPPORT_FIGURE = 200


def boundary_name(frame: pd.DataFrame, level: str) -> pd.Series:
    cross = frame.alignment_class.eq("CROSS")
    if level == "coarse":
        column_cross, column_within = "coarse_boundary", "nearest_coarse_boundary"
    elif level == "fine":
        column_cross, column_within = "fine_boundary", "nearest_fine_boundary"
    else:
        raise ValueError(f"Unknown boundary level: {level}")
    return frame[column_cross].where(cross, frame[column_within]).astype("string")


def aligned_class_rows(frame: pd.DataFrame, outcome: str, boundary_level: str) -> pd.DataFrame:
    if "alignment_join" not in frame:
        frame = frame.copy()
        frame["alignment_join"] = "both"
    usable = frame.loc[
        frame.alignment_join.eq("both")
        & frame.alignment_status.eq("ALIGNED")
        & frame.visible_candidate.fillna(False).astype(bool)
        & frame.alignment_class.isin(["CROSS", "WITHIN_SPLIT"])
        & frame[outcome].notna()
    ].copy()
    usable["boundary"] = boundary_name(usable, boundary_level)
    usable = usable.loc[usable.boundary.notna() & usable.boundary.ne("")]
    usable["boundary"] = usable.boundary.astype(str)
    usable["class_group"] = np.where(usable.alignment_class.eq("CROSS"), "CROSS", "WITHIN")
    usable[outcome] = usable[outcome].astype(int)
    return usable


def summarize_boundaries(
    frame: pd.DataFrame,
    *,
    pair_id: str,
    setting: str,
    outcome_type: str,
    boundary_level: str,
    outcome_column: str = "rejected",
) -> pd.DataFrame:
    usable = aligned_class_rows(frame, outcome_column, boundary_level)
    grouped = usable.groupby(["boundary", "class_group"], observed=True).agg(
        n=(outcome_column, "size"),
        n_docs=("doc_id", "nunique"),
        rate=(outcome_column, "mean"),
    ).reset_index()
    rows: list[dict[str, Any]] = []
    for boundary, subset in grouped.groupby("boundary", sort=True):
        by_class = subset.set_index("class_group")
        if not {"CROSS", "WITHIN"}.issubset(by_class.index):
            continue
        cross, within = by_class.loc["CROSS"], by_class.loc["WITHIN"]
        rows.append({
            "setting": setting,
            "pair_id": pair_id,
            "outcome_type": outcome_type,
            "boundary_level": boundary_level,
            "boundary": str(boundary),
            "n_cross": int(cross.n),
            "n_within": int(within.n),
            "n_cross_docs": int(cross.n_docs),
            "n_within_docs": int(within.n_docs),
            "cross_rate": float(cross.rate),
            "within_rate": float(within.rate),
            "cross_minus_within_pp": float(100 * (cross.rate - within.rate)),
            "support_100_each": bool(cross.n >= MIN_SUPPORT_TABLE and within.n >= MIN_SUPPORT_TABLE),
            "support_200_each": bool(cross.n >= MIN_SUPPORT_FIGURE and within.n >= MIN_SUPPORT_FIGURE),
        })
    return pd.DataFrame(rows)


def load_wikipedia_frame(root: Path, pair_id: str, config: dict[str, Any]) -> pd.DataFrame:
    pair = config["pairs"][pair_id]
    event_path = root / "wikipedia" / "simulated" / pair_id / "common" / "events_k4.parquet"
    events = pd.read_parquet(event_path, columns=EVENT_COLUMNS)
    events = events.loc[events.eligible.fillna(False).astype(bool)].copy()
    events["doc_id"] = events.doc_id.astype(str)
    events["position"] = events.position.astype("int64")
    align_dir = root / "wikipedia" / "aligned" / "kiwi" / pair["target"] / "common"
    alignment = read_shards(align_dir, ALIGNMENT_COLUMNS)
    alignment["doc_id"] = alignment.doc_id.astype(str)
    alignment["generated_token_position"] = alignment.generated_token_position.astype("int64")
    joined = events.merge(
        alignment,
        left_on=["doc_id", "position"],
        right_on=["doc_id", "generated_token_position"],
        how="left",
        validate="one_to_one",
        indicator="alignment_join",
    )
    return joined


def load_human_frame(root: Path, pair_id: str, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    pair = config["pairs"][pair_id]
    target, draft = pair["target"], pair["draft"]
    target_dir = root / "teacher_forced" / "human" / target / target / "common"
    target_docs = set()
    for path in sorted(target_dir.glob("shard-*.parquet")):
        target_docs.update(pd.read_parquet(path, columns=["doc_id"]).doc_id.astype(str).unique())
    if not target_docs:
        raise FileNotFoundError(f"Human TF shards missing for {pair_id}: {target_dir}")
    target_tf = read_tf_for_pair(root, target, target, "common", target_docs, continuation="human")
    draft_tf = read_tf_for_pair(root, target, draft, "common", target_docs, continuation="human")
    target_tf = target_tf[["doc_id", "position", "argmax_id"]].rename(columns={"argmax_id": "target_argmax"})
    draft_tf = draft_tf[["doc_id", "position", "argmax_id"]].rename(columns={"argmax_id": "draft_argmax"})
    scored = target_tf.merge(
        draft_tf,
        on=["doc_id", "position"],
        how="outer",
        validate="one_to_one",
        indicator="tf_join",
    )
    complete_tf = scored.loc[scored.tf_join.eq("both")].copy()
    complete_tf["rejected"] = complete_tf.target_argmax.astype("int64").ne(complete_tf.draft_argmax.astype("int64"))
    align_dir = root / "aligned" / "kiwi" / target / "human" / "common"
    alignment = read_shards(align_dir, ALIGNMENT_COLUMNS)
    alignment["doc_id"] = alignment.doc_id.astype(str)
    alignment["generated_token_position"] = alignment.generated_token_position.astype("int64")
    joined = complete_tf.merge(
        alignment,
        left_on=["doc_id", "position"],
        right_on=["doc_id", "generated_token_position"],
        how="left",
        validate="one_to_one",
        indicator="alignment_join",
    )
    meta = {
        "target_tf_positions": int(len(target_tf)),
        "draft_tf_positions": int(len(draft_tf)),
        "matched_tf_positions": int(len(complete_tf)),
        "unmatched_tf_positions": int(len(scored) - len(complete_tf)),
        "aligned_rows": int(joined.alignment_join.eq("both").sum()),
    }
    return joined, meta


def pairwise_stability(effects: pd.DataFrame) -> pd.DataFrame:
    effects = effects.loc[effects.boundary_level.eq("fine")].copy()
    rows = []
    settings = sorted(effects.setting.unique())
    indexed = {s: g.set_index("boundary") for s, g in effects.groupby("setting")}
    for left, right in itertools.combinations(settings, 2):
        a = indexed[left].loc[indexed[left].support_100_each].copy()
        b = indexed[right].loc[indexed[right].support_100_each].copy()
        shared = sorted(set(a.index).intersection(b.index))
        common = pd.DataFrame({
            "a": a.loc[shared, "cross_minus_within_pp"],
            "b": b.loc[shared, "cross_minus_within_pp"],
        }).dropna()
        rho = float(spearmanr(common.a, common.b).statistic) if len(common) >= 3 else float("nan")
        sign_agreement = float((np.sign(common.a) == np.sign(common.b)).mean()) if len(common) else float("nan")
        top_a = a.sort_values("cross_minus_within_pp", ascending=False, kind="stable").head(5).index
        top_b = b.sort_values("cross_minus_within_pp", ascending=False, kind="stable").head(5).index
        top_a_set, top_b_set = set(top_a), set(top_b)
        union = top_a_set | top_b_set
        rows.append({
            "setting_a": left,
            "pair_a": str(effects.loc[effects.setting.eq(left), "pair_id"].iloc[0]),
            "outcome_a": str(effects.loc[effects.setting.eq(left), "outcome_type"].iloc[0]),
            "setting_b": right,
            "pair_b": str(effects.loc[effects.setting.eq(right), "pair_id"].iloc[0]),
            "outcome_b": str(effects.loc[effects.setting.eq(right), "outcome_type"].iloc[0]),
            "n_shared_supported_boundaries": int(len(common)),
            "spearman_rank_rho": rho,
            "sign_agreement_fraction": sign_agreement,
            "top5_a": ";".join(map(str, top_a)),
            "top5_b": ";".join(map(str, top_b)),
            "top5_overlap_count": int(len(top_a_set & top_b_set)),
            "top5_jaccard": float(len(top_a_set & top_b_set) / len(union)) if union else float("nan"),
            "status": "COMPLETE" if len(common) >= 3 else "NOT_ESTIMABLE_LT3_SHARED_BOUNDARIES",
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/naacl_experiments")
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "results/naacl_experiments/analysis")
    args = parser.parse_args()
    config = load_config(args.config)
    root = args.output_root
    args.analysis_dir.mkdir(parents=True, exist_ok=True)

    qwen_grid_path = args.analysis_dir / "rq2_qwen_grid.csv"
    cached_grid: dict[tuple[str, str], dict[str, Any]] = {}
    if qwen_grid_path.exists():
        existing_grid = pd.read_csv(qwen_grid_path)
        for _, row in existing_grid.iterrows():
            cached_grid[(str(row.pair_id), str(row.estimand))] = row.to_dict()
    rq1_grid_path = args.analysis_dir / "rq1_table2b_cross_within.csv"
    rq1_global: dict[str, dict[str, Any]] = {}
    if rq1_grid_path.exists():
        existing_rq1 = pd.read_csv(rq1_grid_path)
        rq1_global = {str(row.pair_id): row.to_dict() for _, row in existing_rq1.iterrows()}

    all_effects: list[pd.DataFrame] = []
    core_pairs = list(config["pairs"])
    join_meta = []
    grid_rows = []
    for pair_id in core_pairs:
        print(f"LOAD_CORE pair={pair_id}", flush=True)
        frame, meta = load_pair_frame(root, pair_id, config, split_override="common")
        frame = eligible_cross_within(frame)
        frame["coarse_boundary"] = frame.get("coarse_boundary")
        join_meta.append(meta)
        all_effects.append(summarize_boundaries(
            frame,
            pair_id=pair_id,
            setting=f"core_generated_k4:{pair_id}",
            outcome_type="k4_replay_rejection",
            boundary_level="coarse",
        ))
        all_effects.append(summarize_boundaries(
            frame,
            pair_id=pair_id,
            setting=f"core_generated_k4:{pair_id}",
            outcome_type="k4_replay_rejection",
            boundary_level="fine",
        ))
        print(f"CORE_READY pair={pair_id} rows={len(frame):,} docs={frame.doc_id.nunique():,}", flush=True)
        if pair_id.startswith("Q"):
            for label, subset in (
                ("global", frame),
                ("nominal_to_particle", frame.loc[
                    frame.coarse_boundary.eq("NOMINAL_TO_PARTICLE")
                    | (frame.alignment_class.eq("WITHIN_SPLIT") & frame.nearest_coarse_boundary.eq("NOMINAL_TO_PARTICLE"))
                ]),
            ):
                cache_key = (pair_id, label)
                saved = cached_grid.get(cache_key)
                if saved is not None and int(saved.get("n_rows", -1)) == len(subset) and saved.get("status") == "COMPLETE":
                    fitted = saved
                    if "estimate_source" in saved and pd.notna(saved["estimate_source"]):
                        estimate_source = str(saved["estimate_source"])
                    elif label == "global" and pair_id in rq1_global and pair_id not in {"Q1", "Q2"}:
                        estimate_source = "rq1_table2b_cross_within.csv; same common-sample rows and model formula"
                    else:
                        estimate_source = "direct RQ2 fit retained from CPU-RQ2 log"
                    print(f"MODEL_REUSE RQ2-{pair_id}-{label} source=rq2_qwen_grid.csv rows={len(subset):,}", flush=True)
                elif label == "global" and pair_id in rq1_global and int(rq1_global[pair_id].get("n_rows", -1)) == len(subset):
                    cached = rq1_global[pair_id]
                    estimate = float(cached["AME_cross_minus_within"])
                    se = float(cached["clustered_SE"])
                    fitted = {
                        "AME_cross_minus_within": estimate,
                        "raw_AME_before_estimability_audit": estimate,
                        "clustered_SE": se,
                        "CI_low_95": float(cached["CI_low_95"]),
                        "CI_high_95": float(cached["CI_high_95"]),
                        "z": float(cached["z"]),
                        "p_value": float(cached["p_value"]),
                        "n_rows": int(cached["n_rows"]),
                        "n_docs": int(cached["n_docs"]),
                        "n_cross": int(cached["n_cross"]),
                        "n_within": int(cached["n_within"]),
                        "cross_rejection_rate": float(cached["cross_rejection_rate"]),
                        "within_rejection_rate": float(cached["within_rejection_rate"]),
                        "converged": bool(cached["converged"]),
                        "rank_deficient": False,
                        "fit_warnings": "",
                        "status": "COMPLETE" if bool(cached["converged"]) else "NOT_CONVERGED",
                        "deviance": float(cached["deviance"]),
                    }
                    estimate_source = "rq1_table2b_cross_within.csv; same common-sample rows and model formula"
                    print(f"MODEL_REUSE RQ2-{pair_id}-{label} source=rq1_table2b_cross_within.csv rows={len(subset):,}", flush=True)
                else:
                    fitted = fit_cross_ame(subset, label=f"RQ2-{pair_id}-{label}")
                    estimate_source = "fresh fit_cross_ame"
                grid_rows.append({"pair_id": pair_id, "draft": config["pairs"][pair_id]["draft"], "target": config["pairs"][pair_id]["target"], "estimand": label, **fitted, "estimate_source": estimate_source})
                grid_partial = pd.DataFrame(grid_rows)
                grid_partial.to_csv(qwen_grid_path, index=False)
                cached_grid[cache_key] = grid_rows[-1]
                del subset, fitted, grid_partial
                gc.collect()
        del frame
        gc.collect()

    for pair_id in ("Q2", "G1"):
        print(f"LOAD_WIKIPEDIA pair={pair_id}", flush=True)
        frame = load_wikipedia_frame(root, pair_id, config)
        all_effects.append(summarize_boundaries(
            frame,
            pair_id=pair_id,
            setting=f"wikipedia_generated_k4:{pair_id}",
            outcome_type="k4_replay_rejection",
            boundary_level="coarse",
        ))
        all_effects.append(summarize_boundaries(
            frame,
            pair_id=pair_id,
            setting=f"wikipedia_generated_k4:{pair_id}",
            outcome_type="k4_replay_rejection",
            boundary_level="fine",
        ))
        print(f"WIKIPEDIA_READY pair={pair_id} eligible={int(frame.eligible.fillna(False).sum()):,}", flush=True)
        del frame
        gc.collect()

    human_meta: dict[str, Any] = {}
    for pair_id in core_pairs:
        print(f"LOAD_HUMAN pair={pair_id}", flush=True)
        frame, meta = load_human_frame(root, pair_id, config)
        human_meta[pair_id] = meta
        all_effects.append(summarize_boundaries(
            frame,
            pair_id=pair_id,
            setting=f"human_teacher_forced_disagreement:{pair_id}",
            outcome_type="teacher_forced_draft_target_argmax_disagreement",
            boundary_level="coarse",
        ))
        all_effects.append(summarize_boundaries(
            frame,
            pair_id=pair_id,
            setting=f"human_teacher_forced_disagreement:{pair_id}",
            outcome_type="teacher_forced_draft_target_argmax_disagreement",
            boundary_level="fine",
        ))
        print(f"HUMAN_READY pair={pair_id} rows={len(frame):,} aligned={meta['aligned_rows']:,} unmatched_tf={meta['unmatched_tf_positions']:,}", flush=True)
        del frame
        gc.collect()

    effects = pd.concat([f for f in all_effects if not f.empty], ignore_index=True)
    effects_path = args.analysis_dir / "rq2_boundary_effects.csv"
    effects.to_csv(effects_path, index=False)
    print(f"WROTE {effects_path} rows={len(effects)}", flush=True)

    stability = pairwise_stability(effects)
    stability_path = args.analysis_dir / "rq2_table3_setting_stability.csv"
    stability.to_csv(stability_path, index=False)
    print(f"WROTE {stability_path} rows={len(stability)}", flush=True)

    grid = pd.DataFrame(grid_rows)
    grid_path = qwen_grid_path
    grid.to_csv(grid_path, index=False)
    print(f"WROTE {grid_path} rows={len(grid)}", flush=True)

    figure_meta = {
        "core_boundary_settings": [f"core_generated_k4:{p}" for p in core_pairs],
        "core_split": "common 20,000 prompts, including Q2 restricted from its 40,000-prompt all split",
        "figure_boundary_family": "coarse Kiwi boundary label; CROSS uses actual crossed label, WITHIN uses nearest label",
        "table3_boundary_unit": "fine Kiwi POS-transition boundary; CROSS uses actual crossed label, WITHIN uses nearest label",
        "figure_minimum_class_support": MIN_SUPPORT_FIGURE,
        "table_minimum_class_support": MIN_SUPPORT_TABLE,
        "core_outcome": "k=4 offline greedy-SD replay rejection indicator; simulator estimates are scoped by V0 shared-prefix validation",
        "wikipedia_outcome": "k=4 offline greedy-SD replay rejection indicator on frozen Wikipedia sample",
        "human_outcome": "teacher-forced draft-vs-target argmax disagreement at the same human-written prefix; not a live SD rejection",
        "stability_metrics": "Spearman rank correlation and sign agreement over shared fine boundaries supported by at least 100 CROSS and 100 WITHIN rows in each setting; top five are the largest positive CROSS-minus-WITHIN effects among each setting's supported fine boundaries",
        "settings_in_stability_table": int(effects.loc[effects.boundary_level.eq("fine"), "setting"].nunique()),
        "pairwise_rows": int(len(stability)),
        "pairwise_status_counts": {str(k): int(v) for k, v in stability.status.value_counts().items()},
        "supported_fine_boundaries_by_setting": {
            str(setting): int(group.support_100_each.sum())
            for setting, group in effects.loc[effects.boundary_level.eq("fine")].groupby("setting")
        },
        "human_join_meta": human_meta,
        "core_event_join_meta": join_meta,
        "qwen_grid_formula": "Adjusted logistic AME from shared analyze_naacl_cpu GLOBAL_FORMULA; document-clustered sandwich SE",
        "qwen_grid_provenance": "Q1, Q2 common, and N-to-P estimates are retained from direct RQ2 fit logs; global Q3-Q6 estimates are reused from RQ1 Table 2b after exact row-count and common-sample checks",
        "files": {
            "boundary_effects": str(effects_path),
            "stability_table": str(stability_path),
            "qwen_grid": str(grid_path),
        },
    }
    atomic_json(args.analysis_dir / "rq2_metadata.json", figure_meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
