#!/usr/bin/env python3
"""CPU analyses of frozen NAACL teacher-forced SD artifacts.

This script analyzes saved k=4 replay events, target-path morphology alignment,
and teacher-forced controls. It does not run model inference.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

from src.naacl_experiment import atomic_json, load_config, output_root


ALIGN_COLUMNS = [
    "doc_id", "generated_token_position", "token_id", "alignment_class",
    "alignment_status", "visible_candidate", "token_eojeol_count",
    "eojeol_text", "eojeol_char_len", "eojeol_token_count",
    "token_index_in_eojeol", "morpheme_char_spans", "token_char_span",
    "token_start", "token_end", "global_token_char_span", "pool_index",
    "roundtrip_exact", "eojeol_matched", "fine_boundary",
    "nearest_fine_boundary", "nearest_coarse_boundary", "boundary_position_ratio",
]
PAIR_ALIGN_COLUMNS = [
    "doc_id", "generated_token_position", "token_id", "alignment_class",
    "alignment_status", "visible_candidate", "token_eojeol_count",
    "eojeol_text", "morpheme_surfaces", "morpheme_pos_tags", "morpheme_char_spans",
    "token_raw_text", "token_core_text", "token_char_span", "token_start", "token_end",
    "global_token_char_span", "crossed_boundary_count", "roundtrip_exact", "eojeol_matched",
    "eojeol_char_len", "eojeol_token_count", "token_index_in_eojeol",
    "relative_position_in_eojeol", "is_first_token", "is_last_token",
    "fine_boundary", "coarse_boundary", "nearest_fine_boundary", "nearest_coarse_boundary",
    "boundary_position_ratio", "pool_index",
]
EVENT_COLUMNS = [
    "doc_id", "position", "slot", "target_token_id", "draft_argmax_id",
    "target_argmax_id", "rejected", "eligible", "event_type",
]


def read_shards(directory: Path, columns: list[str] | None = None) -> pd.DataFrame:
    paths = sorted(directory.glob("shard-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet shards under {directory}")
    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    return pd.concat(frames, ignore_index=True)


def resolve_split_dir(base: Path, target: str, split: str) -> Path:
    expected = base / target / split
    if expected.exists() and list(expected.glob("shard-*.parquet")):
        return expected
    fallback = base / target / "all"
    if split == "common" and fallback.exists() and list(fallback.glob("shard-*.parquet")):
        return fallback
    raise FileNotFoundError(f"No artifacts for target={target} split={split} under {base}")


def read_tf_for_pair(
    root: Path,
    target: str,
    model: str,
    split: str,
    docs: set[str],
    continuation: str = "generated",
) -> pd.DataFrame:
    directory = root / "teacher_forced" / continuation / target / model / split
    if not list(directory.glob("shard-*.parquet")) and split == "common":
        directory = root / "teacher_forced" / continuation / target / model / "all"
    if not list(directory.glob("shard-*.parquet")):
        raise FileNotFoundError(f"TF shards missing for target={target}, model={model}, split={split}")
    columns = ["doc_id", "position", "target_token_id", "argmax_id", "entropy", "top2_margin", "pool_index"]
    frames = []
    for path in sorted(directory.glob("shard-*.parquet")):
        frame = pd.read_parquet(path, columns=columns)
        frame["doc_id"] = frame.doc_id.astype(str)
        if split == "common" and directory.name == "all":
            frame = frame.loc[frame.pool_index.astype(int) < 20_000]
        frame = frame.loc[frame.doc_id.isin(docs)]
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["position"] = result.position.astype("int64")
    if result.duplicated(["doc_id", "position"]).any():
        raise AssertionError(f"Duplicate TF positions for target={target}, model={model}")
    return result


def load_pair_frame(
    root: Path,
    pair_id: str,
    config: dict[str, Any],
    split_override: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pair = config["pairs"][pair_id]
    target, draft = pair["target"], pair["draft"]
    split = split_override or pair["split"]
    events_path = root / "simulated" / pair_id / split / "events_k4.parquet"
    events = pd.read_parquet(events_path, columns=EVENT_COLUMNS)
    events = events.loc[events.eligible.fillna(False).astype(bool)].copy()
    events["doc_id"] = events.doc_id.astype(str)
    events["position"] = events.position.astype("int64")
    docs = set(events.doc_id.unique())

    alignment_dir = resolve_split_dir(root / "aligned" / "kiwi", target, split)
    alignment = read_shards(alignment_dir, PAIR_ALIGN_COLUMNS)
    alignment["doc_id"] = alignment.doc_id.astype(str)
    if alignment_dir.name == "all" and split == "common":
        alignment = alignment.loc[alignment.pool_index.astype(int) < 20_000]
    alignment["generated_token_position"] = alignment.generated_token_position.astype("int64")

    target_tf = read_tf_for_pair(root, target, target, split, docs).rename(columns={
        "target_token_id": "target_tf_token_id",
        "argmax_id": "target_argmax_tf",
        "entropy": "target_entropy",
        "top2_margin": "target_margin",
    })
    draft_tf = read_tf_for_pair(root, target, draft, split, docs).rename(columns={
        "target_token_id": "draft_tf_target_token_id",
        "argmax_id": "draft_argmax_tf",
        "entropy": "draft_entropy",
        "top2_margin": "draft_margin",
    })

    joined = events.merge(
        alignment,
        left_on=["doc_id", "position"],
        right_on=["doc_id", "generated_token_position"],
        how="left",
        validate="one_to_one",
        indicator="alignment_join",
    )
    joined = joined.merge(target_tf, on=["doc_id", "position"], how="left", validate="one_to_one")
    joined = joined.merge(draft_tf, on=["doc_id", "position"], how="left", validate="one_to_one")
    if joined.target_tf_token_id.isna().any() or joined.draft_tf_target_token_id.isna().any():
        raise AssertionError(f"Missing TF rows after event join for {pair_id}")
    matched = joined.alignment_join.eq("both")
    for left, right in (
        ("target_token_id", "token_id"),
        ("target_token_id", "target_tf_token_id"),
        ("target_token_id", "draft_tf_target_token_id"),
        ("target_argmax_id", "target_argmax_tf"),
        ("draft_argmax_id", "draft_argmax_tf"),
    ):
        if not (joined.loc[matched, left].astype("int64") == joined.loc[matched, right].astype("int64")).all():
            raise AssertionError(f"{pair_id} {left} differs from {right}")
    join_meta = {
        "pair_id": pair_id,
        "target": target,
        "draft": draft,
        "split": split,
        "documents": len(docs),
        "target_positions": int(len(target_tf)),
        "eligible_k4_proposals": int(len(events)),
        "event_positions_with_alignment": int(matched.sum()),
        "event_positions_without_alignment": int((~matched).sum()),
    }
    joined["pair_id"] = pair_id
    joined["rejected"] = joined.rejected.astype("boolean")
    joined["doc_id"] = joined.doc_id.astype(str)
    joined["position"] = joined.position.astype("int64")
    return joined, join_meta


def add_eojeol_keys(alignment: pd.DataFrame) -> pd.DataFrame:
    """Recover the eojeol's absolute start from its token and local token span."""
    alignment = alignment.copy()
    alignment["doc_id"] = alignment.doc_id.astype(str)
    alignment["generated_token_position"] = alignment.generated_token_position.astype("int64")

    global_starts = np.fromiter(
        (int(span[0]) if span is not None and len(span) == 2 else -1 for span in alignment.global_token_char_span),
        dtype=np.int64,
        count=len(alignment),
    )
    local_starts = pd.to_numeric(alignment.token_start, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    starts = global_starts - local_starts
    starts[(global_starts < 0) | (local_starts < 0)] = -1
    alignment["eojeol_global_start"] = starts
    return alignment


def _span_values(value: Any) -> list[tuple[int, int]]:
    if value is None:
        return []
    try:
        spans = []
        for span in value:
            if span is None or len(span) != 2:
                return []
            a, b = int(span[0]), int(span[1])
            spans.append((a, b))
        return spans
    except (TypeError, ValueError):
        return []


def build_eojeol_scores(alignment: pd.DataFrame) -> pd.DataFrame:
    """Compute exact boundary-set precision, recall and F1 for each eojeol.

    Gold boundaries are Kiwi morpheme-span ends inside the eojeol. Predicted
    boundaries are the ends of all tokenizer pieces except the last. An empty
    predicted/gold set scores 1 only when both sets are empty; otherwise a
    missing set has precision/recall 0 for the absent side.
    """
    keys = ["doc_id", "eojeol_global_start", "eojeol_text"]
    eligible = alignment.loc[
        (alignment.eojeol_global_start >= 0)
        & alignment.alignment_status.eq("ALIGNED")
        & alignment.eojeol_matched.fillna(False).astype(bool)
        & alignment.roundtrip_exact.fillna(False).astype(bool)
        & (alignment.token_eojeol_count.astype("Int64") == 1)
        & alignment.token_char_span.notna()
    ]
    print(f"RQ1_BOUNDARY_INPUT rows={len(eligible):,}", flush=True)
    group_stats = eligible.groupby(keys, as_index=False, sort=False, observed=True).agg(
        n_tokens=("token_index_in_eojeol", "size"),
        expected_tokens=("eojeol_token_count", "first"),
        distinct_indices=("token_index_in_eojeol", "nunique"),
        min_index=("token_index_in_eojeol", "min"),
        max_index=("token_index_in_eojeol", "max"),
        eojeol_char_len=("eojeol_char_len", "first"),
    )
    complete = group_stats.loc[
        (group_stats.n_tokens == group_stats.expected_tokens)
        & (group_stats.distinct_indices == group_stats.expected_tokens)
        & (group_stats.min_index == 0)
        & (group_stats.max_index == group_stats.expected_tokens - 1)
    ].copy()
    print(f"RQ1_BOUNDARY_GROUPS complete={len(complete):,}", flush=True)
    eligible = eligible.merge(complete[keys], on=keys, how="inner", validate="many_to_one")

    # Predicted boundaries are distinct token ends except at the eojeol end.
    predicted = eligible.loc[
        (eligible.token_index_in_eojeol.astype("int64") < eligible.eojeol_token_count.astype("int64") - 1)
        & (eligible.token_end.astype("float64") > 0)
        & (eligible.token_end.astype("float64") < eligible.eojeol_char_len.astype("int64")),
        keys + ["token_end"],
    ].drop_duplicates(keys + ["token_end"])
    pred_counts = predicted.groupby(keys, as_index=False, sort=False, observed=True).size().rename(columns={"size": "token_boundary_count"})

    # Kiwi span arrays repeat on every token in an eojeol; retain one copy and
    # explode only those complete groups into their gold internal boundaries.
    first_spans = eligible[keys + ["morpheme_char_spans", "eojeol_char_len"]].drop_duplicates(keys)
    gold = first_spans.explode("morpheme_char_spans", ignore_index=True)
    ends = np.fromiter(
        (int(span[1]) if isinstance(span, (list, tuple, np.ndarray)) and len(span) == 2 else -1
         for span in gold.morpheme_char_spans),
        dtype=np.int64,
        count=len(gold),
    )
    gold["gold_boundary"] = ends
    gold = gold.loc[
        (gold.gold_boundary > 0)
        & (gold.gold_boundary < gold.eojeol_char_len.astype("int64")),
        keys + ["gold_boundary"],
    ].drop_duplicates(keys + ["gold_boundary"])
    gold_counts = gold.groupby(keys, as_index=False, sort=False, observed=True).size().rename(columns={"size": "gold_boundary_count"})
    matches = predicted.merge(
        gold,
        left_on=keys + ["token_end"],
        right_on=keys + ["gold_boundary"],
        how="inner",
        validate="one_to_one",
    )
    tp_counts = matches.groupby(keys, as_index=False, sort=False, observed=True).size().rename(columns={"size": "true_positive_count"})

    scores = complete[keys + ["eojeol_char_len"]]
    scores = scores.merge(gold_counts, on=keys, how="left", validate="one_to_one")
    scores = scores.merge(pred_counts, on=keys, how="left", validate="one_to_one")
    scores = scores.merge(tp_counts, on=keys, how="left", validate="one_to_one")
    for column in ("gold_boundary_count", "token_boundary_count", "true_positive_count"):
        scores[column] = scores[column].fillna(0).astype("int32")
    gold_n = scores.gold_boundary_count.to_numpy(dtype=np.float64)
    pred_n = scores.token_boundary_count.to_numpy(dtype=np.float64)
    tp_n = scores.true_positive_count.to_numpy(dtype=np.float64)
    precision = np.divide(tp_n, pred_n, out=np.where(gold_n == 0, 1.0, 0.0), where=pred_n > 0)
    recall = np.divide(tp_n, gold_n, out=np.where(pred_n == 0, 1.0, 0.0), where=gold_n > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    scores["boundary_precision"] = precision
    scores["boundary_recall"] = recall
    scores["boundary_f1"] = f1
    scores["one_minus_precision"] = 1.0 - precision
    scores["one_minus_recall"] = 1.0 - recall
    scores["one_minus_f1"] = 1.0 - f1
    scores = scores.rename(columns={"eojeol_char_len": "score_eojeol_char_len"})
    return scores


def load_q2_frame(root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair = config["pairs"]["Q2"]
    target, draft, split = pair["target"], pair["draft"], pair["split"]
    events_dir = root / "simulated" / "Q2" / split
    print("RQ1_LOAD events", flush=True)
    events = pd.read_parquet(events_dir / "events_k4.parquet")
    events = events.loc[events.eligible.fillna(False).astype(bool)].copy()
    events["doc_id"] = events.doc_id.astype(str)
    events["position"] = events.position.astype("int64")
    align_dir = resolve_split_dir(root / "aligned" / "kiwi", target, split)
    print("RQ1_LOAD alignment", flush=True)
    alignment = read_shards(align_dir, ALIGN_COLUMNS)
    if align_dir.name == "all" and split == "common":
        alignment = alignment.loc[alignment.pool_index.astype(int) < 20_000]
    alignment = add_eojeol_keys(alignment)

    target_dir = root / "teacher_forced" / "generated" / target / target / split
    draft_dir = root / "teacher_forced" / "generated" / target / draft / split
    if not list(target_dir.glob("shard-*.parquet")) and split == "common":
        target_dir = root / "teacher_forced" / "generated" / target / target / "all"
    if not list(draft_dir.glob("shard-*.parquet")) and split == "common":
        draft_dir = root / "teacher_forced" / "generated" / target / draft / "all"
    print("RQ1_LOAD target_tf", flush=True)
    target_tf = read_shards(target_dir, ["doc_id", "position", "target_token_id", "entropy", "top2_margin", "argmax_id"])
    print("RQ1_LOAD draft_tf", flush=True)
    draft_tf = read_shards(draft_dir, ["doc_id", "position", "target_token_id", "argmax_id", "entropy", "top2_margin"])
    if split == "common":
        # Qwen4's 40k target pass is the source for Q2's common validation split.
        if "pool_index" in target_tf:
            target_tf = target_tf.loc[target_tf.pool_index.astype(int) < 20_000]
        else:
            common_docs = set(events.doc_id.unique())
            target_tf = target_tf.loc[target_tf.doc_id.astype(str).isin(common_docs)]
        if "pool_index" in draft_tf:
            draft_tf = draft_tf.loc[draft_tf.pool_index.astype(int) < 20_000]
        else:
            common_docs = set(events.doc_id.unique())
            draft_tf = draft_tf.loc[draft_tf.doc_id.astype(str).isin(common_docs)]
    for frame in (target_tf, draft_tf):
        frame["doc_id"] = frame.doc_id.astype(str)
        frame["position"] = frame.position.astype("int64")

    # The target and draft TF files have intentionally overlapping field names.
    target_tf = target_tf.rename(columns={
        "target_token_id": "target_tf_token_id",
        "entropy": "target_entropy",
        "top2_margin": "target_margin",
        "argmax_id": "target_argmax_tf",
    })
    draft_tf = draft_tf.rename(columns={
        "target_token_id": "draft_tf_target_token_id",
        "entropy": "draft_entropy",
        "top2_margin": "draft_margin",
        "argmax_id": "draft_argmax_tf",
    })
    event_align = events.merge(
        alignment,
        left_on=["doc_id", "position"],
        right_on=["doc_id", "generated_token_position"],
        how="inner",
        validate="one_to_one",
    )
    if not (event_align.target_token_id.astype("int64") == event_align.token_id.astype("int64")).all():
        raise AssertionError("Q2 event and morphology target token IDs differ")
    joined = event_align.merge(target_tf, on=["doc_id", "position"], how="left", validate="one_to_one")
    joined = joined.merge(draft_tf, on=["doc_id", "position"], how="left", validate="one_to_one")
    if joined.target_tf_token_id.isna().any() or joined.draft_tf_target_token_id.isna().any():
        raise AssertionError("Q2 event rows are missing target or draft TF records")
    if not (joined.target_token_id.astype("int64") == joined.target_tf_token_id.astype("int64")).all():
        raise AssertionError("Q2 target TF token IDs differ from simulated events")
    if not (joined.target_token_id.astype("int64") == joined.draft_tf_target_token_id.astype("int64")).all():
        raise AssertionError("Q2 draft TF token IDs differ from simulated events")
    if not (joined.target_argmax_id.astype("int64") == joined.target_argmax_tf.astype("int64")).all():
        raise AssertionError("Q2 target argmax in replay events differs from target TF")
    if not (joined.draft_argmax_id.astype("int64") == joined.draft_argmax_tf.astype("int64")).all():
        raise AssertionError("Q2 draft argmax in replay events differs from draft TF")
    scores = build_eojeol_scores(alignment)
    joined = joined.merge(
        scores,
        on=["doc_id", "eojeol_global_start", "eojeol_text"],
        how="left",
        validate="many_to_one",
    )
    return joined, scores


def fit_rq1(frame: pd.DataFrame, output: Path) -> dict[str, Any]:
    frame = frame.loc[
        frame.alignment_status.eq("ALIGNED")
        & frame.visible_candidate.fillna(False).astype(bool)
        & frame.alignment_class.isin(["EXACT", "WITHIN_SPLIT", "CROSS", "CROSS_MORPHEME"])
        & frame.one_minus_f1.notna()
    ].copy()
    # Align class spelling in the stored artifacts (CROSS is also used by older runs).
    frame = frame.loc[frame.alignment_class.isin(["EXACT", "WITHIN_SPLIT", "CROSS", "CROSS_MORPHEME"])]
    frame["rejected"] = frame.rejected.astype(int)
    frame["slot"] = frame.slot.astype(str)
    frame["eojeol_token_count"] = frame.eojeol_token_count.astype(int)
    frame["eojeol_char_len"] = frame.eojeol_char_len.astype(float)
    frame["position"] = frame.position.astype(float)
    frame["doc_id"] = frame.doc_id.astype(str)
    controls = frame[[
        "rejected", "doc_id", "slot", "eojeol_token_count", "eojeol_char_len",
        "position", "draft_entropy", "target_entropy", "draft_margin", "target_margin",
        "one_minus_f1", "one_minus_recall", "one_minus_precision",
    ]].replace([np.inf, -np.inf], np.nan).dropna()
    controls["log_eojeol_len"] = np.log1p(controls.eojeol_char_len)
    formula_base = (
        "rejected ~ {effect} + C(eojeol_token_count) + C(slot) + "
        "bs(position, df=5, degree=3, include_intercept=False) + log_eojeol_len + "
        "draft_entropy + target_entropy + draft_margin + target_margin"
    )
    model_rows = []
    model_meta = {}
    output.mkdir(parents=True, exist_ok=True)
    for effect in ("one_minus_f1", "one_minus_recall", "one_minus_precision"):
        fitdata = controls.copy()
        formula = formula_base.format(effect=effect)
        print(f"RQ1_FIT_START predictor={effect} rows={len(fitdata):,} docs={fitdata.doc_id.nunique():,}", flush=True)
        result = smf.glm(
            formula,
            data=fitdata,
            family=sm.families.Binomial(),
        ).fit(
            maxiter=50,
        )
        design = np.asarray(result.model.exog, dtype=np.float64)
        y = np.asarray(result.model.endog, dtype=np.float64)
        beta = np.asarray(result.params, dtype=np.float64)
        eta = design @ beta
        probability = 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))
        variance = probability * (1.0 - probability)
        predictor_index = result.model.exog_names.index(effect)
        mean_variance = float(variance.mean())
        ame = float(beta[predictor_index] * mean_variance)
        gradient = beta[predictor_index] * (design.T @ (variance * (1.0 - 2.0 * probability))) / len(y)
        gradient[predictor_index] += mean_variance

        # Cluster sandwich covariance for the binomial-logit score. Aggregate
        # per-row x*(y-p) scores by document, then form the meat matrix.
        scores = design * (y - probability)[:, None]
        cluster_codes, cluster_names = pd.factorize(fitdata.doc_id, sort=False)
        score_frame = pd.DataFrame(scores)
        score_frame["_cluster"] = cluster_codes
        cluster_scores = score_frame.groupby("_cluster", sort=False).sum().to_numpy(dtype=np.float64)
        bread = np.asarray(result.normalized_cov_params, dtype=np.float64)
        meat = cluster_scores.T @ cluster_scores
        n_rows = len(y)
        n_clusters = len(cluster_names)
        correction = (n_clusters / (n_clusters - 1)) * ((n_rows - 1) / (n_rows - design.shape[1]))
        robust_cov = correction * bread @ meat @ bread
        ame_var = float(gradient @ robust_cov @ gradient)
        ame_se = math.sqrt(max(0.0, ame_var))
        z_value = ame / ame_se if ame_se else float("nan")
        p_value = float(2.0 * norm.sf(abs(z_value))) if math.isfinite(z_value) else float("nan")
        model_rows.append({
            "predictor": effect,
            "AME": ame,
            "cluster_robust_SE": ame_se,
            "z": z_value,
            "p_value": p_value,
            "CI_low_95": ame - 1.96 * ame_se,
            "CI_high_95": ame + 1.96 * ame_se,
            "n_rows": int(result.nobs),
            "n_docs": n_clusters,
            "converged": bool(result.converged),
        })
        model_meta[effect] = {
            "formula": formula,
            "converged": bool(result.converged),
            "deviance": float(result.deviance),
            "aic": float(result.aic),
            "covariance": "document-clustered sandwich; binomial-logit score aggregation",
            "AME_delta_method": "AME = beta_j * mean(p*(1-p)); delta-method SE with clustered coefficient covariance",
        }
        print(f"RQ1_FIT_DONE predictor={effect} converged={result.converged} AME={ame:.7g} SE={ame_se:.7g}", flush=True)
        pd.DataFrame(model_rows).to_csv(output / "rq1_table2a_ame.csv", index=False)
        pd.DataFrame(model_rows).to_markdown(output / "rq1_table2a_ame.md", index=False, floatfmt=".6g")
        atomic_json(output / "rq1_table2a_metadata.json", {
            "analysis": "RQ1 Table 2a; Q2 k=4 eligible replay proposals",
            "population": "visible, ALIGNED target tokens with complete single-eojeol tokenization and exact tokenizer roundtrip",
            "outcome": "k=4 replay rejection indicator",
            "predictors": ["1-boundary F1", "1-boundary recall (fusion)", "1-boundary precision (splitting)"],
            "boundary_definition": "Tokenizer boundaries are internal ends of all but the last token in an eojeol; gold boundaries are Kiwi morpheme-span ends inside the eojeol.",
            "empty_set_convention": "When both sets are empty precision=recall=F1=1; when only one set is empty its precision or recall is 0.",
            "controls": ["eojeol token-count fixed effects", "block slot fixed effects", "5-df cubic spline of generation position", "log eojeol character length", "draft entropy", "target entropy", "draft top-two margin", "target top-two margin"],
            "standard_errors": "Document-clustered sandwich covariance; delta-method AME standard errors.",
            "models": model_meta,
            "eligible_morphology_rows": int(len(controls)),
            "eligible_documents": int(controls.doc_id.nunique()),
            "excluded_rows_before_complete_case": int(len(frame) - len(controls)),
        })
        del result, design, scores, score_frame, cluster_scores, robust_cov, fitdata
    table = pd.DataFrame(model_rows)
    table.to_csv(output / "rq1_table2a_ame.csv", index=False)
    table.to_markdown(output / "rq1_table2a_ame.md", index=False, floatfmt=".6g")
    metadata = {
        "analysis": "RQ1 Table 2a; Q2 k=4 eligible replay proposals",
        "population": "visible, ALIGNED target tokens with complete single-eojeol tokenization and exact tokenizer roundtrip",
        "outcome": "k=4 replay rejection indicator",
        "predictors": ["1-boundary F1", "1-boundary recall (fusion)", "1-boundary precision (splitting)"],
        "boundary_definition": "Tokenizer boundaries are internal ends of all but the last token in an eojeol; gold boundaries are Kiwi morpheme-span ends inside the eojeol.",
        "empty_set_convention": "When both sets are empty precision=recall=F1=1; when only one set is empty its precision or recall is 0.",
        "controls": ["eojeol token-count fixed effects", "block slot fixed effects", "5-df cubic spline of generation position", "log eojeol character length", "draft entropy", "target entropy", "draft top-two margin", "target top-two margin"],
        "standard_errors": "Document-clustered sandwich covariance; delta-method standard errors for AMEs.",
        "models": model_meta,
        "eligible_morphology_rows": int(len(controls)),
        "eligible_documents": int(controls.doc_id.nunique()),
        "excluded_rows_before_complete_case": int(len(frame) - len(controls)),
    }
    atomic_json(output / "rq1_table2a_metadata.json", metadata)
    return {"rows": int(len(controls)), "docs": int(controls.doc_id.nunique()), "table": str(output / "rq1_table2a_ame.csv")}


GLOBAL_FORMULA = (
    "rejected ~ is_cross + C(eojeol_token_count) + C(slot) + "
    "bs(position, df=5, degree=3, include_intercept=False) + log_eojeol_len + "
    "draft_entropy + target_entropy + draft_margin + target_margin"
)


def fit_cross_ame(
    frame: pd.DataFrame,
    *,
    label: str,
    coarse_fragmentation: bool = False,
    formula_override: str | None = None,
) -> dict[str, Any]:
    """Fit a CROSS-vs-WITHIN logit and compute its discrete AME and cluster CI."""
    model_data = frame.copy()
    model_data["rejected"] = model_data.rejected.astype(int)
    model_data["is_cross"] = model_data.alignment_class.eq("CROSS").astype(int)
    model_data["slot"] = model_data.slot.astype(str)
    model_data["eojeol_token_count"] = model_data.eojeol_token_count.astype(int)
    if coarse_fragmentation:
        count = model_data.eojeol_token_count
        model_data["fragmentation_bin"] = pd.Categorical(
            np.select([count <= 1, count == 2], ["1", "2"], default="3+"),
            categories=["1", "2", "3+"],
        )
    model_data["position"] = model_data.position.astype(float)
    model_data["log_eojeol_len"] = np.log1p(model_data.eojeol_char_len.astype(float))
    required = [
        "rejected", "is_cross", "doc_id", "slot", "eojeol_token_count", "position",
        "log_eojeol_len", "draft_entropy", "target_entropy", "draft_margin", "target_margin",
    ]
    if coarse_fragmentation:
        required.append("fragmentation_bin")
    numeric_columns = model_data.select_dtypes(include=[np.number]).columns
    model_data.loc[:, numeric_columns] = model_data.loc[:, numeric_columns].replace([np.inf, -np.inf], np.nan)
    model_data = model_data.dropna(subset=required)
    if model_data.is_cross.nunique() != 2 or model_data.rejected.nunique() != 2:
        raise ValueError(f"{label}: CROSS/WITHIN or rejection outcome lacks both levels")
    print(f"MODEL_START {label} rows={len(model_data):,} docs={model_data.doc_id.nunique():,}", flush=True)
    formula = formula_override or GLOBAL_FORMULA
    if coarse_fragmentation:
        formula = formula.replace("C(eojeol_token_count)", "C(fragmentation_bin)")
    with warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter("always")
        result = smf.glm(formula, data=model_data, family=sm.families.Binomial()).fit(maxiter=50)
    warning_names = sorted({warning.category.__name__ for warning in fit_warnings})
    rank_deficient = "SingularMatrixWarning" in warning_names
    design = np.asarray(result.model.exog, dtype=np.float64)
    y = np.asarray(result.model.endog, dtype=np.float64)
    beta = np.asarray(result.params, dtype=np.float64)
    j = result.model.exog_names.index("is_cross")
    eta = design @ beta
    eta0 = eta - beta[j] * design[:, j]
    p0 = 1.0 / (1.0 + np.exp(-np.clip(eta0, -35.0, 35.0)))
    p1 = 1.0 / (1.0 + np.exp(-np.clip(eta0 + beta[j], -35.0, 35.0)))
    raw_ame = float(np.mean(p1 - p0))
    w0, w1 = p0 * (1.0 - p0), p1 * (1.0 - p1)
    gradient = design.T @ (w1 - w0) / len(y)
    gradient[j] = float(w1.mean())

    scores = design * (y - (1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))))[:, None]
    cluster_codes, cluster_names = pd.factorize(model_data.doc_id.astype(str), sort=False)
    score_frame = pd.DataFrame(scores)
    score_frame["_cluster"] = cluster_codes
    cluster_scores = score_frame.groupby("_cluster", sort=False).sum().to_numpy(dtype=np.float64)
    bread = np.asarray(result.normalized_cov_params, dtype=np.float64)
    meat = cluster_scores.T @ cluster_scores
    n_rows, n_params = design.shape
    n_clusters = len(cluster_names)
    correction = (n_clusters / (n_clusters - 1)) * ((n_rows - 1) / (n_rows - n_params))
    covariance = correction * bread @ meat @ bread
    ame_se = math.sqrt(max(0.0, float(gradient @ covariance @ gradient)))
    estimable = bool(result.converged and not rank_deficient and np.isfinite(ame_se) and ame_se > 1e-12)
    ame = raw_ame if estimable else float("nan")
    z_value = ame / ame_se if estimable else float("nan")
    p_value = float(2.0 * norm.sf(abs(z_value))) if math.isfinite(z_value) else float("nan")
    result_row = {
        "AME_cross_minus_within": ame,
        "raw_AME_before_estimability_audit": raw_ame,
        "clustered_SE": ame_se,
        "CI_low_95": ame - 1.96 * ame_se,
        "CI_high_95": ame + 1.96 * ame_se,
        "z": z_value,
        "p_value": p_value,
        "n_rows": int(n_rows),
        "n_docs": int(n_clusters),
        "n_cross": int((model_data.is_cross == 1).sum()),
        "n_within": int((model_data.is_cross == 0).sum()),
        "cross_rejection_rate": float(model_data.loc[model_data.is_cross == 1, "rejected"].mean()),
        "within_rejection_rate": float(model_data.loc[model_data.is_cross == 0, "rejected"].mean()),
        "converged": bool(result.converged),
        "rank_deficient": rank_deficient,
        "fit_warnings": ";".join(warning_names),
        "status": "COMPLETE" if estimable else (
            "NOT_ESTIMABLE_RANK_DEFICIENT" if rank_deficient else
            "NOT_ESTIMABLE_ZERO_OR_INVALID_SE" if result.converged else "NOT_CONVERGED"
        ),
        "deviance": float(result.deviance),
    }
    print(f"MODEL_DONE {label} status={result_row['status']} converged={result.converged} AME={ame:.7g} raw_AME={raw_ame:.7g} SE={ame_se:.7g} warnings={','.join(warning_names)}", flush=True)
    del result, design, scores, score_frame, cluster_scores, covariance, model_data
    return result_row


def eligible_cross_within(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        frame.alignment_join.eq("both")
        & frame.alignment_status.eq("ALIGNED")
        & frame.visible_candidate.fillna(False).astype(bool)
        & frame.alignment_class.isin(["CROSS", "WITHIN_SPLIT"])
        & frame.rejected.notna()
    ].copy()


def produce_table1_and_rq1b(root: Path, config: dict[str, Any], output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_summaries = []
    class_tables = []
    rq1_rows = []
    pair_ids = list(config["pairs"])
    for pair_id in pair_ids:
        frame, join_meta = load_pair_frame(root, pair_id, config)
        pair_summary = {
            **join_meta,
            "overall_k4_rejection_rate": float(frame.rejected.mean()),
            "aligned_visible_proposals": int((frame.alignment_status.eq("ALIGNED") & frame.visible_candidate.fillna(False)).sum()),
            "cross_proposals": int((frame.alignment_class == "CROSS").sum()),
            "within_proposals": int((frame.alignment_class == "WITHIN_SPLIT").sum()),
        }
        pair_summaries.append(pair_summary)
        class_frame = frame.copy()
        class_frame["alignment_class"] = class_frame.alignment_class.fillna("NO_ALIGNMENT")
        class_summary = class_frame.groupby("alignment_class", dropna=False, sort=True).agg(
            eligible_proposals=("rejected", "size"),
            rejections=("rejected", "sum"),
            rejection_rate=("rejected", "mean"),
            aligned_rows=("alignment_status", lambda s: int(s.eq("ALIGNED").sum())),
        ).reset_index()
        class_summary["pair_id"] = pair_id
        class_summary["class_share"] = class_summary.eligible_proposals / len(class_frame)
        class_tables.append(class_summary)

        target, draft = config["pairs"][pair_id]["target"], config["pairs"][pair_id]["draft"]
        model_frame = eligible_cross_within(frame)
        model = fit_cross_ame(model_frame, label=f"{pair_id}:{draft}->{target}")
        rq1_rows.append({"pair_id": pair_id, "draft": draft, "target": target, **join_meta, **model})
        del frame, model_frame

    summary = pd.DataFrame(pair_summaries)
    classes = pd.concat(class_tables, ignore_index=True)
    rq1b = pd.DataFrame(rq1_rows)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "table1_pair_summary.csv", index=False)
    classes.to_csv(output / "table1_alignment_classes.csv", index=False)
    rq1b.to_csv(output / "rq1_table2b_cross_within.csv", index=False)
    rq1b.to_markdown(output / "rq1_table2b_cross_within.md", index=False, floatfmt=".5g")
    atomic_json(output / "rq1_table2b_metadata.json", {
        "analysis": "Pairwise global CROSS versus WITHIN rejection risk, k=4 replay",
        "pair_data": pair_summaries,
        "population": "eligible k=4 proposals with visible target token, ALIGNED morphology, and class CROSS or WITHIN_SPLIT",
        "formula": GLOBAL_FORMULA,
        "standard_errors": "Document-clustered sandwich covariance; discrete-change AME delta method.",
        "qwen_q2_uses": "40k all split; all other pair jobs use their frozen 20k common split.",
        "outputs": ["table1_pair_summary.csv", "table1_alignment_classes.csv", "rq1_table2b_cross_within.csv"],
    })
    return rq1b, summary


def produce_q2_boundary_landscape(root: Path, config: dict[str, Any], output: Path) -> pd.DataFrame:
    frame, join_meta = load_pair_frame(root, "Q2", config)
    rows = eligible_cross_within(frame)
    rows["boundary"] = np.where(
        rows.alignment_class.eq("CROSS"),
        rows.fine_boundary.astype("string"),
        rows.nearest_fine_boundary.astype("string"),
    )
    rows["boundary"] = rows.boundary.replace({"<NA>": np.nan, "nan": np.nan, "None": np.nan})
    rows = rows.dropna(subset=["boundary"])
    support = rows.groupby(["boundary", "alignment_class"], observed=True).size().unstack(fill_value=0)
    if "CROSS" not in support:
        support["CROSS"] = 0
    if "WITHIN_SPLIT" not in support:
        support["WITHIN_SPLIT"] = 0
    supported = support.loc[(support.CROSS >= 200) & (support.WITHIN_SPLIT >= 200)]
    print(f"FIG2_SUPPORT supported_boundaries={len(supported)} cross_within_min=200", flush=True)
    boundary_rows = []
    for boundary in supported.index:
        subset = rows.loc[rows.boundary.eq(boundary)].copy()
        try:
            fit = fit_cross_ame(subset, label=f"Fig2:{boundary}")
            status = fit["status"]
        except Exception as exc:
            fit = {"AME_cross_minus_within": np.nan, "clustered_SE": np.nan,
                   "raw_AME_before_estimability_audit": np.nan,
                   "CI_low_95": np.nan, "CI_high_95": np.nan, "z": np.nan,
                   "p_value": np.nan, "n_rows": len(subset),
                   "n_docs": subset.doc_id.nunique(), "n_cross": int((subset.alignment_class == "CROSS").sum()),
                   "n_within": int((subset.alignment_class == "WITHIN_SPLIT").sum()),
                   "cross_rejection_rate": float(subset.loc[subset.alignment_class == "CROSS", "rejected"].mean()),
                   "within_rejection_rate": float(subset.loc[subset.alignment_class == "WITHIN_SPLIT", "rejected"].mean()),
                   "converged": False, "rank_deficient": False, "fit_warnings": "",
                   "deviance": np.nan}
            status = f"FAILED: {type(exc).__name__}: {exc}"
            print(f"FIG2_MODEL_FAILURE boundary={boundary} reason={status}", flush=True)
        boundary_rows.append({"boundary": boundary, **fit, "status": status})
        del subset
    result = pd.DataFrame(boundary_rows)
    result["q_value_bh"] = np.nan
    p_mask = result.p_value.notna() & result.status.eq("COMPLETE")
    if p_mask.any():
        from statsmodels.stats.multitest import multipletests
        result.loc[p_mask, "q_value_bh"] = multipletests(result.loc[p_mask, "p_value"].to_numpy(), method="fdr_bh")[1]
    result = result.sort_values("AME_cross_minus_within", na_position="last").reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "rq1_fig2_boundary_landscape.csv", index=False)
    result.to_markdown(output / "rq1_fig2_boundary_landscape.md", index=False, floatfmt=".5g")
    atomic_json(output / "rq1_fig2_metadata.json", {
        "analysis": "Q2 boundary-specific adjusted CROSS minus WITHIN rejection probability",
        "pair_data": join_meta,
        "boundary_assignment": "CROSS uses its observed fine_boundary; WITHIN_SPLIT uses nearest_fine_boundary.",
        "support_threshold": "At least 200 eligible ALIGNED visible proposals in each class per boundary.",
        "model": GLOBAL_FORMULA,
        "multiplicity": "Benjamini-Hochberg q-values over all estimable supported-boundary tests.",
        "notes": "Boundary-specific associations are observational; the within class is assigned to its nearest Kiwi boundary. Fragmentation fixed effects use the observed eojeol-token-count categories. Non-converged, rank-deficient, zero-SE, or invalid-SE fits are retained in the CSV but excluded from Fig. 2 and the BH family.",
        "n_supported_boundaries": int(len(supported)),
        "n_estimable_tests": int(p_mask.sum()),
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--analysis", choices=["rq1a", "rq1b-table1", "rq1b-table1-fig2", "rq1-fig2"], default="rq1a")
    args = parser.parse_args()
    config = load_config(args.config)
    root = output_root(config, args.output_dir)
    output = root / "analysis"
    if args.analysis == "rq1a":
        joined, scores = load_q2_frame(root, config)
        print(f"Q2_JOIN rows={len(joined)} aligned_eojeols={len(scores)}", flush=True)
        summary = fit_rq1(joined, output)
        print("RQ1A_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)
    elif args.analysis == "rq1b-table1":
        table2b, table1 = produce_table1_and_rq1b(root, config, output)
        print(f"TABLE1_RQ1B_COMPLETE pairs={len(table2b)} table1_pairs={len(table1)}", flush=True)
    elif args.analysis == "rq1b-table1-fig2":
        table2b, table1 = produce_table1_and_rq1b(root, config, output)
        boundary = produce_q2_boundary_landscape(root, config, output)
        print(f"RQ1_COMPLETE pairs={len(table2b)} table1_pairs={len(table1)} q2_boundaries={len(boundary)}", flush=True)
    else:
        boundary = produce_q2_boundary_landscape(root, config, output)
        print(f"FIG2_COMPLETE q2_boundaries={len(boundary)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
