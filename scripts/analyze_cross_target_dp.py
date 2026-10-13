#!/usr/bin/env python3
"""Analyze target decision-point consistency with document-level uncertainty."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cross_target_dp import (
    exact_bottom_labels,
    pairwise_overlap,
    rank_correlation,
    within_document_permutation,
)
from src.naacl_experiment import atomic_json, atomic_parquet, load_config, sha256_file


THRESHOLDS = (0.10, 0.20, 0.30)
MODEL_ORDER = ("qwen4", "qwen8", "qwen14", "gemma12", "ministral8")


def _load_complete_runs(config: dict, output_dir: Path, positions: pd.DataFrame,
                        model_keys: tuple[str, ...] = MODEL_ORDER):
    frames = []
    run_meta = {}
    expected_ids = positions.position_id.astype(str).tolist()
    expected_hashes = positions.set_index("position_id").shared_prefix_sha256.to_dict()
    for key in model_keys:
        metadata_path = output_dir / "model_runs" / key / "run_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing target run manifest: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "COMPLETE":
            raise RuntimeError(f"Target run is not complete: {key} status={metadata.get('status')}")
        if metadata.get("model_revision") != config["models"][key]["revision"]:
            raise RuntimeError(f"Target run revision mismatch for {key}")
        parts = [pd.read_parquet(ROOT / path) for path in metadata["outputs"]]
        frame = pd.concat(parts, ignore_index=True)
        if frame.position_id.astype(str).tolist() != expected_ids:
            raise RuntimeError(f"Saved positions do not match the frozen position manifest for {key}")
        for row in frame[["position_id", "shared_prefix_sha256"]].itertuples(index=False):
            if expected_hashes[str(row.position_id)] != row.shared_prefix_sha256:
                raise RuntimeError(f"Shared-prefix hash mismatch for {key} at {row.position_id}")
        frames.append(frame)
        run_meta[key] = metadata
    long = pd.concat(frames, ignore_index=True)
    counts = long.groupby("position_id").model_key.nunique()
    if not counts.eq(len(model_keys)).all():
        raise RuntimeError(f"Position rows are not complete across selected targets: {model_keys}")
    family_order = {key: index for index, key in enumerate(model_keys)}
    long["_model_order"] = long.model_key.map(family_order)
    position_order = {value: index for index, value in enumerate(expected_ids)}
    long["_position_order"] = long.position_id.map(position_order)
    long = long.sort_values(["_position_order", "_model_order"], kind="stable").drop(
        columns=["_model_order", "_position_order"]
    ).reset_index(drop=True)
    return long, run_meta


def _wide(long: pd.DataFrame, positions: pd.DataFrame, analysis_set: str,
          model_keys: tuple[str, ...] = MODEL_ORDER):
    subset = long.loc[long.analysis_set.eq(analysis_set)].copy()
    wide = subset.pivot(index="position_id", columns="model_key", values="target_top2_margin_nats")
    wide = wide.reindex(columns=model_keys)
    if wide.isna().any().any():
        raise RuntimeError(f"Incomplete model margins in analysis set {analysis_set}")
    meta_columns = [
        "position_id", "doc_id", "char_offset", "shared_prefix_sha256", "context_chars",
        "analysis_set", "position_type", "matched_word_id", "morph_left_morpheme",
        "morph_left_pos", "morph_right_morpheme", "morph_right_pos", "morpheme_form",
        "morpheme_pos",
    ]
    available = [column for column in meta_columns if column in positions.columns]
    metadata = positions.loc[positions.analysis_set.eq(analysis_set), available].drop_duplicates("position_id")
    metadata = metadata.set_index("position_id").reindex(wide.index)
    return wide, metadata


def _pair_records(margins: np.ndarray, doc_ids: np.ndarray, keys: tuple[str, ...], fractions=THRESHOLDS):
    labels_by_fraction = {
        fraction: np.column_stack([
            exact_bottom_labels(margins[:, index], fraction)
            for index in range(margins.shape[1])
        ])
        for fraction in fractions
    }
    overlap_rows = []
    correlation_rows = []
    pair_values = {}
    for a_index, b_index in itertools.combinations(range(len(keys)), 2):
        key_a, key_b = keys[a_index], keys[b_index]
        family_a = "Qwen3" if key_a.startswith("qwen") else ("Gemma 3" if key_a == "gemma12" else "Ministral 3")
        family_b = "Qwen3" if key_b.startswith("qwen") else ("Gemma 3" if key_b == "gemma12" else "Ministral 3")
        pair_id = f"{key_a}__{key_b}"
        correlation_rows.append({
            "model_a": key_a,
            "model_b": key_b,
            "family_a": family_a,
            "family_b": family_b,
            "same_family": family_a == family_b,
            "n_documents": int(len(np.unique(doc_ids))),
            "n_positions": int(len(doc_ids)),
            "spearman_margin": rank_correlation(margins[:, a_index], margins[:, b_index]),
            "normalized_margin_rank_correlation": rank_correlation(margins[:, a_index], margins[:, b_index]),
        })
        for fraction, labels in labels_by_fraction.items():
            metrics = pairwise_overlap(labels[:, a_index], labels[:, b_index])
            overlap_rows.append({
                "model_a": key_a,
                "model_b": key_b,
                "pair_id": pair_id,
                "family_a": family_a,
                "family_b": family_b,
                "same_family": family_a == family_b,
                "threshold_fraction": fraction,
                **metrics,
            })
            if fraction == 0.20:
                pair_values[pair_id] = metrics
    return overlap_rows, correlation_rows, labels_by_fraction, pair_values


def _bootstrap_main(margins: np.ndarray, doc_ids: np.ndarray, keys: tuple[str, ...],
                    replicates: int, seed: int):
    unique_docs = np.unique(doc_ids)
    groups = {doc: np.flatnonzero(doc_ids == doc) for doc in unique_docs}
    pairs = list(itertools.combinations(range(len(keys)), 2))
    metrics_to_keep = ("jaccard", "overlap_pct", "conditional_b_given_a")
    collected: dict[tuple[str, str], list[float]] = {}
    family_values = {"within_qwen_jaccard": [], "cross_family_jaccard": [], "within_minus_cross_jaccard": []}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        sampled_docs = rng.choice(unique_docs, size=len(unique_docs), replace=True)
        rows = np.concatenate([groups[doc] for doc in sampled_docs])
        sample_margins = margins[rows]
        sample_doc_ids = doc_ids[rows]
        labels = np.column_stack([
            exact_bottom_labels(sample_margins[:, index], 0.20)
            for index in range(sample_margins.shape[1])
        ])
        within = []
        cross = []
        for a_index, b_index in pairs:
            key_a, key_b = keys[a_index], keys[b_index]
            pair_id = f"{key_a}__{key_b}"
            values = pairwise_overlap(labels[:, a_index], labels[:, b_index])
            same_family = key_a.startswith("qwen") and key_b.startswith("qwen")
            if same_family:
                within.append(float(values["jaccard"]))
            else:
                cross.append(float(values["jaccard"]))
            for metric in metrics_to_keep:
                collected.setdefault((pair_id, metric), []).append(float(values[metric]))
            corr = rank_correlation(sample_margins[:, a_index], sample_margins[:, b_index])
            collected.setdefault((pair_id, "spearman_margin"), []).append(corr)
        if within and cross:
            within_mean = float(np.mean(within))
            cross_mean = float(np.mean(cross))
            family_values["within_qwen_jaccard"].append(within_mean)
            family_values["cross_family_jaccard"].append(cross_mean)
            family_values["within_minus_cross_jaccard"].append(within_mean - cross_mean)

    rows_out = []
    for (pair_id, metric), values in collected.items():
        values = np.asarray(values, dtype=np.float64)
        model_a, model_b = pair_id.split("__")
        rows_out.append({
            "analysis_set": "eojeol_boundary",
            "model_a": model_a,
            "model_b": model_b,
            "pair_id": pair_id,
            "metric": metric,
            "threshold_fraction": 0.20 if metric != "spearman_margin" else None,
            "bootstrap_replicates": replicates,
            "ci_low_95": float(np.nanquantile(values, 0.025)),
            "ci_high_95": float(np.nanquantile(values, 0.975)),
            "bootstrap_mean": float(np.nanmean(values)),
        })
    family_rows = []
    for metric, values in family_values.items():
        if not values:
            continue
        values = np.asarray(values, dtype=np.float64)
        family_rows.append({
            "metric": metric,
            "bootstrap_replicates": replicates,
            "ci_low_95": float(np.quantile(values, 0.025)),
            "ci_high_95": float(np.quantile(values, 0.975)),
            "bootstrap_mean": float(values.mean()),
            "within_family_definition": "mean Jaccard among selected Qwen3 target pairs",
            "cross_family_definition": "mean Jaccard among selected target pairs from different model families",
        })
    if not family_rows:
        family_rows.append({
            "metric": "within_vs_cross_family",
            "status": "NOT_ESTIMABLE",
            "reason": "The selected targets contain no cross-family pair; use the pairwise bootstrap interval instead.",
        })
    return rows_out, family_rows


def _permutation_main(labels: np.ndarray, doc_ids: np.ndarray, keys: tuple[str, ...],
                      replicates: int, seed: int):
    observed = {
        f"{keys[a]}__{keys[b]}": pairwise_overlap(labels[:, a], labels[:, b])
        for a, b in itertools.combinations(range(len(keys)), 2)
    }
    null = {pair: {"jaccard": [], "overlap_pct": [], "intersection_size": []} for pair in observed}
    shared_hist = np.zeros((replicates, len(keys) + 1), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        permuted = within_document_permutation(labels, doc_ids, rng)
        for a, b in itertools.combinations(range(len(keys)), 2):
            pair_id = f"{keys[a]}__{keys[b]}"
            metrics = pairwise_overlap(permuted[:, a], permuted[:, b])
            for metric in null[pair_id]:
                null[pair_id][metric].append(float(metrics[metric]))
        shared_hist[replicate] = np.bincount(permuted.sum(axis=1), minlength=len(keys) + 1) / len(permuted)

    rows = []
    for pair_id, metrics in observed.items():
        for metric in ("jaccard", "overlap_pct"):
            values = np.asarray(null[pair_id][metric], dtype=np.float64)
            obs = float(metrics[metric])
            rows.append({
                "analysis_set": "eojeol_boundary",
                "model_a": pair_id.split("__")[0],
                "model_b": pair_id.split("__")[1],
                "pair_id": pair_id,
                "threshold_fraction": 0.20,
                "metric": metric,
                "observed": obs,
                "null_mean": float(values.mean()),
                "null_sd": float(values.std(ddof=1)),
                "null_ci_low_95": float(np.quantile(values, 0.025)),
                "null_ci_high_95": float(np.quantile(values, 0.975)),
                "observed_minus_null_mean": obs - float(values.mean()),
                "empirical_upper_tail_p": float((1 + np.sum(values >= obs)) / (len(values) + 1)),
                "permutation_replicates": replicates,
                "permutation_rule": "Independently permute each target's DP labels within document, preserving each target's per-document DP count.",
            })
    shared_rows = []
    for count in range(len(keys) + 1):
        null_values = shared_hist[:, count]
        shared_rows.append({
            "n_targets_identifying_position": count,
            "observed_position_count": int(np.sum(labels.sum(axis=1) == count)),
            "observed_fraction": float(np.mean(labels.sum(axis=1) == count)),
            "permutation_mean_fraction": float(null_values.mean()),
            "permutation_ci_low_95": float(np.quantile(null_values, 0.025)),
            "permutation_ci_high_95": float(np.quantile(null_values, 0.975)),
            "permutation_replicates": replicates,
        })
    bucket_rows = []
    buckets = [("0", (0,)), ("exactly_1", (1,))]
    if len(keys) >= 2:
        buckets.append(("exactly_2", (2,)))
    if len(keys) >= 3:
        buckets.append(("3+", tuple(range(3, len(keys) + 1))))
    observed_counts = labels.sum(axis=1)
    for label, counts in buckets:
        null_values = shared_hist[:, list(counts)].sum(axis=1)
        observed_mask = np.isin(observed_counts, counts)
        bucket_rows.append({
            "target_count_bucket": label,
            "observed_position_count": int(observed_mask.sum()),
            "observed_fraction": float(observed_mask.mean()),
            "permutation_mean_fraction": float(null_values.mean()),
            "permutation_ci_low_95": float(np.quantile(null_values, 0.025)),
            "permutation_ci_high_95": float(np.quantile(null_values, 0.975)),
            "permutation_replicates": replicates,
        })
    return rows, bucket_rows, shared_rows


def _doc_bootstrap_difference(values: pd.DataFrame, outcome: str, replicates: int, seed: int):
    paired = values.pivot(index="matched_word_id", columns="position_type", values=outcome)
    paired = paired.dropna(subset=["morpheme_boundary", "morpheme_internal"])
    meta = values.drop_duplicates("matched_word_id").set_index("matched_word_id")
    paired["doc_id"] = meta.loc[paired.index, "doc_id"].astype(str)
    paired["difference"] = paired["morpheme_boundary"].astype(float) - paired["morpheme_internal"].astype(float)
    per_doc = paired.groupby("doc_id").difference.mean().to_numpy(dtype=np.float64)
    observed = float(per_doc.mean()) if len(per_doc) else float("nan")
    rng = np.random.default_rng(seed)
    samples = np.asarray([
        rng.choice(per_doc, size=len(per_doc), replace=True).mean()
        for _ in range(replicates)
    ]) if len(per_doc) else np.array([], dtype=np.float64)
    return paired, observed, samples


def _morphology_analysis(long: pd.DataFrame, positions: pd.DataFrame, *, replicates: int, seed: int,
                         model_keys: tuple[str, ...] = MODEL_ORDER):
    wide, meta = _wide(long, positions, "morph_extended", model_keys)
    if wide.empty:
        return [], {"status": "UNSUPPORTED_NO_RELIABLE_KIWI_POSITIONS"}
    margins = wide.to_numpy(dtype=np.float64)
    labels = np.column_stack([exact_bottom_labels(margins[:, i], 0.20) for i in range(margins.shape[1])])
    counts = labels.sum(axis=1)
    enriched = meta.copy()
    enriched["n_targets_dp"] = counts
    enriched["exactly_one_target"] = counts == 1
    if len(model_keys) == 2:
        enriched["shared_by_all_targets"] = counts == 2
        outcomes = ("exactly_one_target", "shared_by_all_targets")
    else:
        enriched["shared_by_three_or_more"] = counts >= 3
        outcomes = ["exactly_one_target", "shared_by_three_or_more"]
        families_present = {"Qwen3" if key.startswith("qwen") else ("Gemma 3" if key == "gemma12" else "Ministral 3") for key in model_keys}
        if {"Qwen3", "Gemma 3", "Ministral 3"}.issubset(families_present):
            family_columns = []
            for family in ("Qwen3", "Gemma 3", "Ministral 3"):
                family_keys = [index for index, key in enumerate(model_keys)
                               if ("Qwen3" if key.startswith("qwen") else ("Gemma 3" if key == "gemma12" else "Ministral 3")) == family]
                family_columns.append(labels[:, family_keys].any(axis=1))
            enriched["cross_family_shared"] = np.column_stack(family_columns).all(axis=1)
            outcomes.append("cross_family_shared")
    enriched = enriched.reset_index()
    rows = []
    total_matched = int(enriched.matched_word_id.nunique())
    total_docs = int(enriched.doc_id.nunique())
    for outcome_index, outcome in enumerate(outcomes):
        paired, difference, draws = _doc_bootstrap_difference(
            enriched, outcome, replicates, seed + outcome_index
        )
        boundary_rate = float(paired.morpheme_boundary.mean()) if len(paired) else float("nan")
        internal_rate = float(paired.morpheme_internal.mean()) if len(paired) else float("nan")
        rows.append({
            "outcome": outcome,
            "matched_word_pairs": int(len(paired)),
            "documents": int(paired.doc_id.nunique()) if len(paired) else 0,
            "morpheme_boundary_prevalence": boundary_rate,
            "morpheme_internal_prevalence": internal_rate,
            "boundary_minus_internal": difference,
            "document_bootstrap_ci_low_95": float(np.quantile(draws, 0.025)) if len(draws) else None,
            "document_bootstrap_ci_high_95": float(np.quantile(draws, 0.975)) if len(draws) else None,
            "bootstrap_replicates": replicates,
            "interpretation": "Matched within the same eojeol and averaged within document; association only.",
        })
    metadata = {
        "status": "COMPLETE",
        "morphology_positions": int(len(enriched)),
        "matched_word_pairs": total_matched,
        "documents": total_docs,
        "decision_definition": "Bottom 20% of margins among the common morph_extended positions for each target.",
        "models_compared": list(model_keys),
        "shared_definition": "All selected targets mark the position as a decision point.",
    }
    return rows, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cross_target_dp.yaml")
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--permutation-replicates", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--models", default=",".join(MODEL_ORDER),
        help="Comma-separated completed model keys (default: all configured targets).",
    )
    args = parser.parse_args()
    model_keys = tuple(key.strip() for key in args.models.split(",") if key.strip())
    if len(model_keys) < 2 or len(set(model_keys)) != len(model_keys):
        parser.error("--models must contain at least two distinct model keys")

    config = load_config(args.config)
    unknown = [key for key in model_keys if key not in config["models"]]
    if unknown:
        parser.error(f"unknown model keys: {', '.join(unknown)}")
    output_dir = ROOT / config["paths"]["outputs"]
    position_path = output_dir / "evaluation_positions.parquet"
    positions = pd.read_parquet(position_path)
    long, run_meta = _load_complete_runs(config, output_dir, positions, model_keys)
    long_path = output_dir / "position_margins.parquet"
    atomic_parquet(long.to_dict(orient="records"), long_path)

    wide, meta = _wide(long, positions, "eojeol_boundary", model_keys)
    margins = wide.to_numpy(dtype=np.float64)
    doc_ids = meta.doc_id.astype(str).to_numpy()
    if len(set(np.bincount(pd.factorize(doc_ids, sort=True)[0]))) != 1:
        raise RuntimeError("Primary position design must have equal positions per document")
    overlap_rows, correlation_rows, labels_by_fraction, pair_values = _pair_records(
        margins, doc_ids, model_keys
    )

    bootstrap_reps = args.bootstrap_replicates or int(config["inference"]["bootstrap_replicates"])
    permutation_reps = args.permutation_replicates or int(config["inference"]["permutation_replicates"])
    seed = args.seed or int(config["inference"]["seed"])
    bootstrap_rows, family_bootstrap_rows = _bootstrap_main(
        margins, doc_ids, model_keys, bootstrap_reps, seed
    )
    permutation_rows, shared_bucket_rows, shared_count_rows = _permutation_main(
        labels_by_fraction[0.20], doc_ids, model_keys, permutation_reps, seed + 1
    )

    prevalence_rows = []
    for index, key in enumerate(model_keys):
        for fraction in THRESHOLDS:
            labels = labels_by_fraction[fraction][:, index]
            prevalence_rows.append({
                "model_key": key,
                "model_name": config["models"][key]["id"],
                "model_family": config["models"][key]["family"],
                "decision_rule": f"bottom_{int(fraction * 100)}_percent",
                "margin_threshold_nats": float(np.quantile(margins[:, index], fraction)),
                "positions": int(labels.sum()),
                "prevalence": float(labels.mean()),
                "n_documents": int(len(np.unique(doc_ids))),
                "n_positions": int(len(doc_ids)),
            })
        absolute = margins[:, index] < 1.0
        prevalence_rows.append({
            "model_key": key,
            "model_name": config["models"][key]["id"],
            "model_family": config["models"][key]["family"],
            "decision_rule": "absolute_margin_lt_1_nats_secondary",
            "margin_threshold_nats": 1.0,
            "positions": int(absolute.sum()),
            "prevalence": float(absolute.mean()),
            "n_documents": int(len(np.unique(doc_ids))),
            "n_positions": int(len(doc_ids)),
        })

    morph_rows, morph_meta = _morphology_analysis(
        long, positions, replicates=bootstrap_reps, seed=seed + 2, model_keys=model_keys
    )
    outputs = {
        "pairwise_overlap.csv": pd.DataFrame(overlap_rows),
        "pairwise_correlations.csv": pd.DataFrame(correlation_rows),
        "bootstrap_ci.csv": pd.DataFrame(bootstrap_rows),
        "permutation_results.csv": pd.DataFrame(permutation_rows),
        "decision_point_prevalence.csv": pd.DataFrame(prevalence_rows),
        "shared_target_counts.csv": pd.DataFrame(shared_count_rows),
        "shared_target_buckets.csv": pd.DataFrame(shared_bucket_rows),
        "family_consistency_bootstrap.csv": pd.DataFrame(family_bootstrap_rows),
        "morphological_enrichment.csv": pd.DataFrame(morph_rows),
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)

    pair_frame = pd.DataFrame(overlap_rows)
    p20 = pair_frame.loc[pair_frame.threshold_fraction.eq(0.20)]
    within_p20 = p20.loc[p20.same_family, "jaccard"].astype(float).tolist()
    cross_p20 = p20.loc[~p20.same_family, "jaccard"].astype(float).tolist()
    family_order = {key: ("Qwen3" if key.startswith("qwen") else ("Gemma 3" if key == "gemma12" else "Ministral 3")) for key in model_keys}
    family_shared_supported = {"Qwen3", "Gemma 3", "Ministral 3"}.issubset(set(family_order.values()))
    family_shared = None
    if family_shared_supported:
        family_masks = []
        for family in ("Qwen3", "Gemma 3", "Ministral 3"):
            family_indices = [i for i, key in enumerate(model_keys) if family_order[key] == family]
            family_masks.append(labels_by_fraction[0.20][:, family_indices].any(axis=1))
        family_shared = np.column_stack(family_masks).all(axis=1)
    target_count = labels_by_fraction[0.20].sum(axis=1)
    # Absolute margins are secondary and are not used for cross-tokenizer ranking.
    position_metadata = json.loads((output_dir / "position_metadata.json").read_text())
    model_summaries = {}
    for key in model_keys:
        model_rows = long.loc[long.model_key.eq(key) & long.analysis_set.eq("eojeol_boundary")]
        token_lengths = model_rows.context_model_tokens.to_numpy(dtype=np.float64)
        model_summaries[key] = {
            "id": config["models"][key]["id"],
            "revision": config["models"][key]["revision"],
            "family": config["models"][key]["family"],
            "positions": run_meta[key]["positions_written"],
            "tokens_processed": run_meta[key]["input_tokens_processed"],
            "context_tokens_min": int(token_lengths.min()),
            "context_tokens_median": float(np.median(token_lengths)),
            "context_tokens_p95": float(np.quantile(token_lengths, 0.95)),
            "context_tokens_max": int(token_lengths.max()),
            "full_forward_check_max_abs_error": run_meta[key]["full_forward_check_max_abs_error"],
        }
    exact_distribution = {str(count): int(np.sum(target_count == count)) for count in range(len(model_keys) + 1)}
    grouped_distribution = {
        "zero": exact_distribution["0"],
        "exactly_one": exact_distribution["1"],
        "exactly_two": exact_distribution.get("2", 0),
        "three_or_more": int(sum(exact_distribution[str(count)] for count in range(3, len(model_keys) + 1))),
    }
    pair_null = pd.DataFrame(permutation_rows)
    jaccard_null = pair_null.loc[pair_null.metric.eq("jaccard")]
    pair_meta = p20[["model_a", "model_b", "same_family"]].drop_duplicates()
    null_with_family = jaccard_null.merge(pair_meta, on=["model_a", "model_b"], validate="one_to_one")
    within_null_values = null_with_family.loc[null_with_family.same_family, "null_mean"].astype(float).tolist()
    cross_null_values = null_with_family.loc[~null_with_family.same_family, "null_mean"].astype(float).tolist()
    family_ci_by_metric = {row["metric"]: row for row in family_bootstrap_rows}
    summary = {
        "schema": config["schema"],
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Are low-margin next-token decision points shared across target models on identical Korean text prefixes?",
        "analysis_scope": {
            "models_compared": list(model_keys),
            "scope_note": ("Two-model interim analysis; supports conclusions only about the Qwen3-4B/Qwen3-8B pair. "
                           "No cross-family comparison is available." if model_keys == ("qwen4", "qwen8") else
                           "Selected-target analysis; conclusions are limited to the configured models in this run."),
        },
        "primary_definition": "For each target, mark exactly the bottom 20% of margins on the same common eojeol-onset positions. Stable ordering resolves exact ties.",
        "primary_metric": "Pairwise bottom-20% Jaccard, compared to within-document permutation null.",
        "data": {
            "source": "Frozen FineWeb2 Korean kor_Hang common split; raw NFC/stripped document text.",
            "documents_considered": int(position_metadata["primary"]["documents_considered"]),
            "documents_evaluated": int(len(np.unique(doc_ids))),
            "positions_evaluated": int(len(doc_ids)),
            "unique_shared_prefixes": int(meta.shared_prefix_sha256.nunique()),
            "positions_excluded_reasons": position_metadata["primary"],
            "context_window_chars": int(config["data"]["context_window_chars"]),
            "minimum_context_chars": int(config["data"]["min_context_chars"]),
            "context_chars_min": int(meta.context_chars.min()),
            "context_chars_median": float(meta.context_chars.median()),
            "context_chars_p95": float(meta.context_chars.quantile(0.95)),
            "context_chars_max": int(meta.context_chars.max()),
            "position_sampling": "20 evenly spaced eojeol starts per eligible document; the prefix is text[context_start:char_offset].",
        },
        "models": model_summaries,
        "uncertainty": {
            "bootstrap_replicates": bootstrap_reps,
            "bootstrap_unit": "document",
            "permutation_replicates": permutation_reps,
            "permutation_unit": "independent within-document DP-label permutation for each target, preserving each model's per-document count",
            "random_seed": seed,
        },
        "primary_results": {
            "within_qwen_mean_jaccard": float(np.mean(within_p20)) if within_p20 else None,
            "cross_family_mean_jaccard": float(np.mean(cross_p20)) if cross_p20 else None,
            "within_minus_cross_mean_jaccard": (float(np.mean(within_p20) - np.mean(cross_p20)) if within_p20 and cross_p20 else None),
            "within_qwen_permutation_mean_jaccard": float(np.mean(within_null_values)) if within_null_values else None,
            "cross_family_permutation_mean_jaccard": float(np.mean(cross_null_values)) if cross_null_values else None,
            "within_minus_cross_bootstrap_ci_95": family_ci_by_metric.get("within_minus_cross_jaccard"),
            "cross_family_shared_position_fraction": float(np.mean(family_shared)) if family_shared is not None else None,
            "pairwise_bottom20_jaccard": {row.pair_id: float(row.jaccard) for row in p20.itertuples()},
            "pairwise_spearman_margin": {f"{row.model_a}__{row.model_b}": float(row.spearman_margin)
                                         for row in pd.DataFrame(correlation_rows).itertuples()},
            "target_count_distribution": exact_distribution,
            "target_count_distribution_grouped": grouped_distribution,
        },
        "morphology": {**morph_meta, "exclusions": position_metadata["morphology"]["exclusions"]},
        "outputs": {name: sha256_file(output_dir / name) for name in outputs},
        "position_margins_sha256": sha256_file(long_path),
        "interpretation_note": "Raw margins and the absolute <1-nat rule are not calibrated across different tokenizers/families. Cross-target consistency is assessed primarily by within-model ranks and overlap on identical textual prefixes. Associations are descriptive, not causal.",
    }
    atomic_json(output_dir / "summary.json", summary)
    _write_findings(output_dir, summary, outputs)
    pair_text = ",".join(f"{key}={value:.4f}" for key, value in summary["primary_results"]["pairwise_bottom20_jaccard"].items())
    print(f"ANALYSIS_COMPLETE models={','.join(model_keys)} positions={len(doc_ids)} pairwise_jaccard={pair_text}", flush=True)
    return 0


def _write_findings(output_dir: Path, summary: dict, outputs: dict[str, pd.DataFrame]):
    overlap = outputs["pairwise_overlap.csv"]
    p20 = overlap.loc[overlap.threshold_fraction.eq(0.20)]
    correlations = outputs["pairwise_correlations.csv"]
    permutation = outputs["permutation_results.csv"]
    jaccard_null = permutation.loc[permutation.metric.eq("jaccard")]
    bootstrap = outputs["bootstrap_ci.csv"]
    bootstrap_map = {(row.pair_id, row.metric): row for row in bootstrap.itertuples()}
    corr_map = {(row.model_a, row.model_b): row.spearman_margin for row in correlations.itertuples()}
    if len(summary["models"]) == 2:
        _write_two_model_findings(output_dir, summary, outputs, bootstrap_map, corr_map)
        return
    null_map = {(row.model_a, row.model_b): row for row in jaccard_null.itertuples()}
    family_ci = outputs["family_consistency_bootstrap.csv"].set_index("metric")
    within = summary["primary_results"]["within_qwen_mean_jaccard"]
    cross = summary["primary_results"]["cross_family_mean_jaccard"]
    within_null = summary["primary_results"]["within_qwen_permutation_mean_jaccard"]
    cross_null = summary["primary_results"]["cross_family_permutation_mean_jaccard"]
    difference = family_ci.loc["within_minus_cross_jaccard"]
    grouped = summary["primary_results"]["target_count_distribution_grouped"]

    if within > within_null and difference.ci_low_95 > 0:
        outcome = "B"
        outcome_sentence = (
            "The pattern is closest to Outcome B: Qwen3 within-family overlap is above its permutation mean, "
            "and the document-bootstrap interval places it above cross-family overlap. This supports family- or "
            "target-dependent decision-point locations in this model set."
        )
    elif cross > cross_null and difference.ci_low_95 <= 0:
        outcome = "A"
        outcome_sentence = (
            "The pattern is closest to Outcome A: cross-family overlap is above its within-document permutation "
            "mean, and the bootstrap interval does not establish a lower cross-family mean than within-Qwen overlap. "
            "This is evidence for shared hotspots in the evaluated sample, not proof of intrinsic ambiguity."
        )
    elif within <= within_null and cross <= cross_null:
        outcome = "C"
        outcome_sentence = (
            "The pattern is closest to Outcome C: mean within-Qwen and cross-family overlaps do not exceed their "
            "within-document permutation means. The existing RQ4 should therefore remain interpreted as shared "
            "failure locations conditional on a fixed target."
        )
    else:
        outcome = "Mixed"
        outcome_sentence = (
            "The estimates do not cleanly match one of Outcomes A–C: observed family means and their permutation "
            "references give a mixed pattern. The pairwise intervals and permutation results below should guide "
            "interpretation without a categorical claim."
        )

    lines = [
        "# RQ6: Cross-target decision-point consistency",
        "",
        "## Design and scope",
        "",
        f"The analysis evaluated {summary['data']['positions_evaluated']:,} shared eojeol-onset prefixes from {summary['data']['documents_evaluated']:,} of {summary['data']['documents_considered']:,} considered FineWeb2 Korean documents. {summary['data']['positions_excluded_reasons']['documents_excluded_insufficient_positions']} documents were excluded because they had too few eligible eojeol onsets. Each target received the identical NFC text prefix at each document/character offset, then tokenized it independently. The primary decision-point definition is the exact bottom 20% of margins within each target on the common position set.",
        "",
        f"Prefixes contain {summary['data']['context_chars_min']}–{summary['data']['context_chars_max']} characters after snapping the nominal {summary['data']['context_window_chars']}-character window back to an eojeol boundary (median {summary['data']['context_chars_median']:.0f}). Per-model token-length summaries are in `summary.json`. Raw margins and the absolute margin <1 rule are secondary because next-token vocabularies and margin scales differ. This is a diagnostic comparison, not a causal test.",
        "",
        "## Pairwise results",
        "",
        "| Target pair | Family relation | Spearman rho (95% doc-bootstrap CI) | Bottom-20% Jaccard (95% CI) | Observed / independence overlap (%) | P(DP_B &#124; DP_A) | Overlap enrichment vs independence | Null Jaccard mean | Observed − null | Upper-tail p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in p20.itertuples():
        null = null_map[(row.model_a, row.model_b)]
        rho = corr_map[(row.model_a, row.model_b)]
        rho_ci = bootstrap_map[(row.pair_id, "spearman_margin")]
        jaccard_ci = bootstrap_map[(row.pair_id, "jaccard")]
        relation = "within Qwen3" if row.same_family else "cross-family"
        lines.append(
            f"| {row.model_a} vs {row.model_b} | {relation} | {rho:.3f} [{rho_ci.ci_low_95:.3f}, {rho_ci.ci_high_95:.3f}] | {row.jaccard:.3f} [{jaccard_ci.ci_low_95:.3f}, {jaccard_ci.ci_high_95:.3f}] | {row.overlap_pct:.2f} / {row.independence_expected_overlap_pct:.2f} | {row.conditional_b_given_a:.3f} | {row.overlap_enrichment:.3f} | {null.null_mean:.3f} | {null.observed_minus_null_mean:+.3f} | {null.empirical_upper_tail_p:.3f} |"
        )

    lines.extend([
        "",
        "Empirical permutation p-values are upper-tail, unadjusted across pairs. The overlap column gives observed and independence-reference percentages; P(DP_B | DP_A) is directional in the displayed pair order, and enrichment is observed overlap divided by the independence expectation. The within-document permutation is the main null reference; independence is only a descriptive baseline.",
        "",
        "### Threshold sensitivity",
        "",
        "| Bottom fraction | Mean pairwise Jaccard | Mean within-Qwen3 Jaccard | Mean cross-family Jaccard |",
        "|---:|---:|---:|---:|",
    ])
    for fraction, group in overlap.groupby("threshold_fraction"):
        within_values = group.loc[group.same_family, "jaccard"]
        cross_values = group.loc[~group.same_family, "jaccard"]
        lines.append(f"| {fraction:.0%} | {group.jaccard.mean():.3f} | {within_values.mean():.3f} | {cross_values.mean():.3f} |")

    lines.extend([
        "",
        "## Family and shared-position summaries",
        "",
        f"Mean Qwen3 within-family Jaccard is {within:.3f} (within-document permutation mean {within_null:.3f}); mean cross-family Jaccard is {cross:.3f} (permutation mean {cross_null:.3f}). The within-minus-cross document-bootstrap 95% CI is [{difference.ci_low_95:.3f}, {difference.ci_high_95:.3f}]. Only Qwen3 has multiple targets in this set, so there is no within-Gemma or within-Ministral estimate.",
        "",
        f"The fraction marked by at least one target from each of the three families (Qwen3, Gemma 3, Ministral 3) is {summary['primary_results']['cross_family_shared_position_fraction']:.3f}. Across positions, {grouped['zero']:,} are selected by zero targets, {grouped['exactly_one']:,} by exactly one, {grouped['exactly_two']:,} by exactly two, and {grouped['three_or_more']:,} by three or more. `shared_target_buckets.csv` compares these groups with the permutation distribution; `shared_target_counts.csv` gives exact counts zero through five.",
        "",
        "## Morphological comparison",
        "",
    ])
    morph = outputs["morphological_enrichment.csv"]
    if len(morph):
        exclusions = summary["morphology"].get("exclusions", {})
        excluded_text = ", ".join(f"{key}: {value}" for key, value in exclusions.items()) or "none"
        lines.append(
            f"Kiwi-derived morpheme-boundary and within-morpheme character offsets were matched within the same sampled eojeol ({summary['morphology']['matched_word_pairs']:,} word pairs from {summary['morphology']['documents']:,} documents). Automatic alignment exclusions: {excluded_text}. The comparison is associational."
        )
        lines.extend([
            "",
            "| Outcome | Matched words | Boundary prevalence | Within-morpheme prevalence | Difference (boundary − internal) | Document-bootstrap 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in morph.itertuples():
            lines.append(f"| {row.outcome} | {row.matched_word_pairs} | {row.morpheme_boundary_prevalence:.3f} | {row.morpheme_internal_prevalence:.3f} | {row.boundary_minus_internal:+.3f} | [{row.document_bootstrap_ci_low_95:+.3f}, {row.document_bootstrap_ci_high_95:+.3f}] |")
    else:
        lines.append("The Kiwi positions did not provide a supported matched morphology comparison; no morphology claim is made.")

    lines.extend([
        "",
        "## Interpretation for KORSpec",
        "",
        f"**Observed pattern: Outcome {outcome}.** {outcome_sentence}",
        "",
        "Outcome A would support model-robust textual hotspots without establishing causal or intrinsic linguistic ambiguity. Outcome B indicates family dependence. Outcome C limits the existing RQ4 interpretation to draft failures conditional on a fixed target. If estimates do not meet a clean pattern, the report labels them mixed rather than forcing an outcome.",
        "",
        "Figures A–D show normalized margin ranks, pairwise overlap, shared-target counts against the within-document permutation baseline, and the matched morphological comparison. Full CSVs, model revisions, per-model token lengths, exclusions, and output hashes are saved under this directory.",
        "",
        "### Proposed RQ6 paragraph",
        "",
    ])
    q48 = p20.loc[p20.model_a.eq("qwen4") & p20.model_b.eq("qwen8")].iloc[0]
    q8g = p20.loc[p20.model_a.eq("qwen8") & p20.model_b.eq("gemma12")].iloc[0]
    q8m = p20.loc[p20.model_a.eq("qwen8") & p20.model_b.eq("ministral8")].iloc[0]
    paragraph = (
        f"On identical Korean text prefixes from {summary['data']['documents_evaluated']:,} FineWeb2 documents "
        f"({summary['data']['positions_evaluated']:,} eojeol onsets), bottom-20% decision-point Jaccard averaged "
        f"{within:.3f} within Qwen3 and {cross:.3f} across families, against corresponding within-document "
        f"permutation means of {within_null:.3f} and {cross_null:.3f}. The within-Qwen minus cross-family mean "
        f"difference was {difference.bootstrap_mean:+.3f} (document-bootstrap 95% CI "
        f"[{difference.ci_low_95:+.3f}, {difference.ci_high_95:+.3f}]); selected pairwise overlaps included "
        f"Qwen3-4B/8B {q48.jaccard:.3f}, Qwen3-8B/Gemma3-12B {q8g.jaccard:.3f}, and "
        f"Qwen3-8B/Ministral3-8B {q8m.jaccard:.3f}. The observed pattern is closest to Outcome {outcome}. "
        "These results describe normalized low-margin locations in the evaluated prefixes; they do not establish "
        "intrinsic linguistic ambiguity, and the paper's RQ4 remains conditional on a fixed target."
    )
    lines.extend([paragraph, ""])
    (output_dir / "cross_target_findings.md").write_text("\n".join(lines), encoding="utf-8")


def _write_two_model_findings(output_dir: Path, summary: dict, outputs: dict[str, pd.DataFrame],
                              bootstrap_map: dict, corr_map: dict):
    overlap = outputs["pairwise_overlap.csv"]
    p20 = overlap.loc[overlap.threshold_fraction.eq(0.20)]
    row = p20.iloc[0]
    pair_id = row.pair_id
    permutation = outputs["permutation_results.csv"]
    null = permutation.loc[(permutation.metric == "jaccard") & (permutation.pair_id == pair_id)].iloc[0]
    rho_ci = bootstrap_map[(pair_id, "spearman_margin")]
    jaccard_ci = bootstrap_map[(pair_id, "jaccard")]
    grouped = summary["primary_results"]["target_count_distribution_grouped"]
    models = list(summary["models"])
    names = [summary["models"][key]["id"] for key in models]
    lines = [
        "# RQ6 interim result: Qwen3-4B vs Qwen3-8B",
        "",
        "## Scope",
        "",
        summary["analysis_scope"]["scope_note"],
        "This is a two-target, within-family result. It cannot determine whether low-margin decision points are shared across model families or generalize to other target architectures.",
        "",
        f"The analysis uses {summary['data']['positions_evaluated']:,} identical Korean text prefixes at eojeol onsets from {summary['data']['documents_evaluated']:,} FineWeb2 documents. Each model tokenized the same NFC character prefix independently. For each model, a decision point is one of the exact bottom 20% of next-token top-2 margins across the common positions.",
        f"Prefixes span {summary['data']['context_chars_min']}–{summary['data']['context_chars_max']} characters (median {summary['data']['context_chars_median']:.0f}); per-model token counts and pinned revisions are recorded in `summary.json`.",
        "",
        "## Main result",
        "",
        "| Pair | Margin Spearman rho (95% document-bootstrap CI) | Bottom-20% Jaccard (95% document-bootstrap CI) | Shared positions | Fraction of all positions | Qwen4 DP positions shared with Qwen8 | Within-document null Jaccard mean [95% permutation interval] | Observed − null | Upper-tail permutation p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {names[0]} vs {names[1]} | {corr_map[(row.model_a, row.model_b)]:.3f} [{rho_ci.ci_low_95:.3f}, {rho_ci.ci_high_95:.3f}] | {row.jaccard:.3f} [{jaccard_ci.ci_low_95:.3f}, {jaccard_ci.ci_high_95:.3f}] | {int(row.intersection_size):,} | {row.overlap_pct:.2f}% | {row.conditional_b_given_a:.1%} | {null.null_mean:.3f} [{null.null_ci_low_95:.3f}, {null.null_ci_high_95:.3f}] | {null.observed_minus_null_mean:+.3f} | {null.empirical_upper_tail_p:.3f} |",
        "",
        "The permutation null independently relocates each target’s selected positions within each document while preserving that document’s decision-point count. The reported p-value is an unadjusted upper-tail empirical value. The document bootstrap resamples documents, retaining each selected document’s positions.",
        "",
        "## Threshold sensitivity",
        "",
        "| Bottom fraction per model | Pairwise Jaccard | Positions shared | Shared fraction |",
        "|---:|---:|---:|---:|",
    ]
    for fraction, values in overlap.groupby("threshold_fraction"):
        metric = values.iloc[0]
        lines.append(f"| {fraction:.0%} | {metric.jaccard:.3f} | {int(metric.intersection_size):,} | {metric.overlap_pct:.2f}% |")
    lines.extend([
        "",
        "## Positions selected by both models",
        "",
        f"Among all common prefixes, {grouped['zero']:,} are selected by neither model, {grouped['exactly_one']:,} by exactly one, and {grouped['exactly_two']:,} by both. The permutation comparison for each count bucket is in `shared_target_buckets.csv`.",
        "",
        "## Morphological comparison",
        "",
    ])
    morph = outputs["morphological_enrichment.csv"]
    if len(morph):
        exclusions = summary["morphology"].get("exclusions", {})
        excluded_text = ", ".join(f"{key}: {value}" for key, value in exclusions.items()) or "none"
        lines.extend([
            f"Kiwi-derived boundary and within-morpheme offsets are matched within the same sampled eojeol ({summary['morphology']['matched_word_pairs']:,} word pairs across {summary['morphology']['documents']:,} documents). Alignment exclusions: {excluded_text}. Differences are descriptive associations.",
            "",
            "| Outcome | Matched words | Boundary prevalence | Internal prevalence | Boundary − internal (95% document-bootstrap CI) |",
            "|---|---:|---:|---:|---:|",
        ])
        for item in morph.itertuples():
            lines.append(f"| {item.outcome} | {item.matched_word_pairs:,} | {item.morpheme_boundary_prevalence:.3f} | {item.morpheme_internal_prevalence:.3f} | {item.boundary_minus_internal:+.3f} [{item.document_bootstrap_ci_low_95:+.3f}, {item.document_bootstrap_ci_high_95:+.3f}] |")
    else:
        lines.append("The Kiwi positions did not provide a supported matched morphology comparison; no morphology claim is made.")
    lines.extend([
        "",
        "## Interpretation and limitation",
        "",
        "This interim result estimates consistency between two sizes in the Qwen3 family only. It does not support the planned cross-family comparison or the original multi-target outcome classification. Low-margin overlap is descriptive and does not establish intrinsic linguistic ambiguity or causality. The complete five-target RQ6 remains pending the other model runs.",
        "",
        "### Proposed interim paragraph",
        "",
        f"On identical Korean text prefixes from {summary['data']['documents_evaluated']:,} FineWeb2 documents ({summary['data']['positions_evaluated']:,} eojeol onsets), the bottom-20% decision-point sets for Qwen3-4B and Qwen3-8B had a Jaccard similarity of {row.jaccard:.3f} (document-bootstrap 95% CI [{jaccard_ci.ci_low_95:.3f}, {jaccard_ci.ci_high_95:.3f}]) and shared {int(row.intersection_size):,} positions ({row.overlap_pct:.2f}% of all tested positions). Their margin ranks correlated at Spearman ρ = {corr_map[(row.model_a, row.model_b)]:.3f} (95% CI [{rho_ci.ci_low_95:.3f}, {rho_ci.ci_high_95:.3f}]). The observed overlap was {row.jaccard:.3f}, compared with a within-document relocation null mean of {null.null_mean:.3f} (95% permutation interval [{null.null_ci_low_95:.3f}, {null.null_ci_high_95:.3f}], upper-tail p = {null.empirical_upper_tail_p:.3f}). This interim comparison is limited to two Qwen3 targets and cannot establish cross-family generality.",
        "",
    ])
    (output_dir / "cross_target_findings.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
