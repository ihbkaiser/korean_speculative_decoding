#!/usr/bin/env python3
"""CPU analysis of frozen NAACL teacher-forced SD artifacts.

The initial implementation produces the Q2 RQ1 Table 2a models from the
saved k=4 replay and target-path Kiwi alignment. It does not run inference.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pair", default="Q2", choices=["Q2"])
    args = parser.parse_args()
    config = load_config(args.config)
    root = output_root(config, args.output_dir)
    output = root / "analysis"
    joined, scores = load_q2_frame(root, config)
    print(f"Q2_JOIN rows={len(joined)} aligned_eojeols={len(scores)}", flush=True)
    summary = fit_rq1(joined, output)
    print("RQ1_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
