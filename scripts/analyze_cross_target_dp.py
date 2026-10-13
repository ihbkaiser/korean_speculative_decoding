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


def _load_complete_runs(config: dict, output_dir: Path, positions: pd.DataFrame):
    frames = []
    run_meta = {}
    expected_ids = positions.position_id.astype(str).tolist()
    expected_hashes = positions.set_index("position_id").shared_prefix_sha256.to_dict()
    for key in MODEL_ORDER:
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
    if not counts.eq(len(MODEL_ORDER)).all():
        raise RuntimeError("Position rows are not complete across all five targets")
    family_order = {key: index for index, key in enumerate(MODEL_ORDER)}
    long["_model_order"] = long.model_key.map(family_order)
    position_order = {value: index for index, value in enumerate(expected_ids)}
    long["_position_order"] = long.position_id.map(position_order)
    long = long.sort_values(["_position_order", "_model_order"], kind="stable").drop(
        columns=["_model_order", "_position_order"]
    ).reset_index(drop=True)
    return long, run_meta


def _wide(long: pd.DataFrame, positions: pd.DataFrame, analysis_set: str):
    subset = long.loc[long.analysis_set.eq(analysis_set)].copy()
    wide = subset.pivot(index="position_id", columns="model_key", values="target_top2_margin_nats")
    wide = wide.reindex(columns=MODEL_ORDER)
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
        values = np.asarray(values, dtype=np.float64)
        family_rows.append({
            "metric": metric,
            "bootstrap_replicates": replicates,
            "ci_low_95": float(np.quantile(values, 0.025)),
            "ci_high_95": float(np.quantile(values, 0.975)),
            "bootstrap_mean": float(values.mean()),
            "within_family_definition": "mean Jaccard among the three Qwen3 target pairs",
            "cross_family_definition": "mean Jaccard among the seven target pairs from different model families",
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
    return rows, shared_rows


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


def _morphology_analysis(long: pd.DataFrame, positions: pd.DataFrame, *, replicates: int, seed: int):
    wide, meta = _wide(long, positions, "morph_extended")
    if wide.empty:
        return [], {"status": "UNSUPPORTED_NO_RELIABLE_KIWI_POSITIONS"}
    margins = wide.to_numpy(dtype=np.float64)
    labels = np.column_stack([exact_bottom_labels(margins[:, i], 0.20) for i in range(margins.shape[1])])
    counts = labels.sum(axis=1)
    families = np.column_stack([labels[:, :3].any(axis=1), labels[:, 3], labels[:, 4]])
    enriched = meta.copy()
    enriched["n_targets_dp"] = counts
    enriched["exactly_one_target"] = counts == 1
    enriched["shared_by_three_or_more"] = counts >= 3
    enriched["cross_family_shared"] = families.all(axis=1)
    enriched = enriched.reset_index()
    outcomes = ("exactly_one_target", "shared_by_three_or_more", "cross_family_shared")
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
        "family_shared_definition": "At least one Qwen3 target, Gemma-3-12B, and Ministral-3-8B all mark the position.",
    }
    return rows, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cross_target_dp.yaml")
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--permutation-replicates", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ROOT / config["paths"]["outputs"]
    position_path = output_dir / "evaluation_positions.parquet"
    positions = pd.read_parquet(position_path)
    long, run_meta = _load_complete_runs(config, output_dir, positions)
    long_path = output_dir / "position_margins.parquet"
    atomic_parquet(long.to_dict(orient="records"), long_path)

    wide, meta = _wide(long, positions, "eojeol_boundary")
    margins = wide.to_numpy(dtype=np.float64)
    doc_ids = meta.doc_id.astype(str).to_numpy()
    if len(set(np.bincount(pd.factorize(doc_ids, sort=True)[0]))) != 1:
        raise RuntimeError("Primary position design must have equal positions per document")
    overlap_rows, correlation_rows, labels_by_fraction, pair_values = _pair_records(
        margins, doc_ids, MODEL_ORDER
    )

    bootstrap_reps = args.bootstrap_replicates or int(config["inference"]["bootstrap_replicates"])
    permutation_reps = args.permutation_replicates or int(config["inference"]["permutation_replicates"])
    seed = args.seed or int(config["inference"]["seed"])
    bootstrap_rows, family_bootstrap_rows = _bootstrap_main(
        margins, doc_ids, MODEL_ORDER, bootstrap_reps, seed
    )
    permutation_rows, shared_count_rows = _permutation_main(
        labels_by_fraction[0.20], doc_ids, MODEL_ORDER, permutation_reps, seed + 1
    )

    prevalence_rows = []
    for index, key in enumerate(MODEL_ORDER):
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
        long, positions, replicates=bootstrap_reps, seed=seed + 2
    )
    outputs = {
        "pairwise_overlap.csv": pd.DataFrame(overlap_rows),
        "pairwise_correlations.csv": pd.DataFrame(correlation_rows),
        "bootstrap_ci.csv": pd.DataFrame(bootstrap_rows),
        "permutation_results.csv": pd.DataFrame(permutation_rows),
        "decision_point_prevalence.csv": pd.DataFrame(prevalence_rows),
        "shared_target_counts.csv": pd.DataFrame(shared_count_rows),
        "family_consistency_bootstrap.csv": pd.DataFrame(family_bootstrap_rows),
        "morphological_enrichment.csv": pd.DataFrame(morph_rows),
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)

    within = [row["jaccard"] for row in pair_values.values() if row]
    pair_frame = pd.DataFrame(overlap_rows)
    p20 = pair_frame.loc[pair_frame.threshold_fraction.eq(0.20)]
    within_p20 = p20.loc[p20.same_family, "jaccard"].astype(float).tolist()
    cross_p20 = p20.loc[~p20.same_family, "jaccard"].astype(float).tolist()
    families = labels_by_fraction[0.20][:, :3].any(axis=1) & labels_by_fraction[0.20][:, 3] & labels_by_fraction[0.20][:, 4]
    target_count = labels_by_fraction[0.20].sum(axis=1)
    # Absolute margins are secondary and are not used for cross-tokenizer ranking.
    summary = {
        "schema": config["schema"],
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Are low-margin next-token decision points shared across target models on identical Korean text prefixes?",
        "primary_definition": "For each target, mark exactly the bottom 20% of margins on the same common eojeol-onset positions. Stable ordering resolves exact ties.",
        "primary_metric": "Pairwise bottom-20% Jaccard, compared to within-document permutation null; family contrast is Qwen3 within-family mean minus cross-family mean.",
        "data": {
            "source": "Frozen FineWeb2 Korean kor_Hang common split; raw NFC/stripped document text.",
            "documents_considered": int(json.loads((output_dir / "position_metadata.json").read_text())["primary"]["documents_considered"]),
            "documents_evaluated": int(len(np.unique(doc_ids))),
            "positions_evaluated": int(len(doc_ids)),
            "unique_shared_prefixes": int(meta.shared_prefix_sha256.nunique()),
            "positions_excluded_reasons": json.loads((output_dir / "position_metadata.json").read_text())["primary"],
            "context_window_chars": int(config["data"]["context_window_chars"]),
            "minimum_context_chars": int(config["data"]["min_context_chars"]),
            "position_sampling": "20 evenly spaced eojeol starts per eligible document; the prefix is text[context_start:char_offset].",
        },
        "models": {
            key: {
                "id": config["models"][key]["id"],
                "revision": config["models"][key]["revision"],
                "family": config["models"][key]["family"],
                "positions": run_meta[key]["positions_written"],
                "tokens_processed": run_meta[key]["input_tokens_processed"],
                "full_forward_check_max_abs_error": run_meta[key]["full_forward_check_max_abs_error"],
            }
            for key in MODEL_ORDER
        },
        "uncertainty": {
            "bootstrap_replicates": bootstrap_reps,
            "bootstrap_unit": "document",
            "permutation_replicates": permutation_reps,
            "permutation_unit": "independent within-document DP-label permutation for each target, preserving each model's per-document count",
            "random_seed": seed,
        },
        "primary_results": {
            "within_qwen_mean_jaccard": float(np.mean(within_p20)),
            "cross_family_mean_jaccard": float(np.mean(cross_p20)),
            "within_minus_cross_mean_jaccard": float(np.mean(within_p20) - np.mean(cross_p20)),
            "cross_family_shared_position_fraction": float(np.mean(families)),
            "target_count_distribution": {
                str(count): int(np.sum(target_count == count)) for count in range(len(MODEL_ORDER) + 1)
            },
        },
        "morphology": morph_meta,
        "outputs": {name: sha256_file(output_dir / name) for name in outputs},
        "position_margins_sha256": sha256_file(long_path),
        "interpretation_note": "Raw margins and the absolute <1-nat rule are not calibrated across different tokenizers/families. Cross-target consistency is assessed primarily by within-model ranks and overlap on identical textual prefixes. Associations are descriptive, not causal.",
    }
    atomic_json(output_dir / "summary.json", summary)
    _write_findings(output_dir, summary, outputs)
    print(
        f"ANALYSIS_COMPLETE positions={len(doc_ids)} within_qwen_jaccard={summary['primary_results']['within_qwen_mean_jaccard']:.4f} "
        f"cross_family_jaccard={summary['primary_results']['cross_family_mean_jaccard']:.4f} "
        f"cross_family_shared={summary['primary_results']['cross_family_shared_position_fraction']:.4f}",
        flush=True,
    )
    return 0


def _write_findings(output_dir: Path, summary: dict, outputs: dict[str, pd.DataFrame]):
    overlap = outputs["pairwise_overlap.csv"]
    p20 = overlap.loc[overlap.threshold_fraction.eq(0.20)]
    correlations = outputs["pairwise_correlations.csv"]
    permutation = outputs["permutation_results.csv"]
    jaccard_null = permutation.loc[permutation.metric.eq("jaccard")]
    lines = [
        "# RQ6: Cross-target decision-point consistency",
        "",
        "## Design and scope",
        "",
        f"The analysis evaluated {summary['data']['positions_evaluated']:,} shared eojeol-onset prefixes from {summary['data']['documents_evaluated']:,} FineWeb2 Korean documents. Every target received the identical NFC text prefix at each document/character offset; each model tokenized that text independently. The primary decision-point definition is the exact bottom 20% of margins within each target on the common position set. This is a diagnostic comparison, not a causal test.",
        "",
        "Raw margins and the absolute margin <1 rule are secondary because tokenizers and model families do not calibrate probabilities identically. The primary comparison uses normalized ranks and bottom-20% overlap. Permutations shuffle labels within each document while preserving each target's per-document decision-point count. Confidence intervals use document bootstrap resampling.",
        "",
        "## Pairwise results",
        "",
        "| Target pair | Family relation | Spearman margin rho | Bottom-20% Jaccard | Overlap (% positions) | Permutation-null Jaccard | Observed − null | Upper-tail p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    corr_map = {(row.model_a, row.model_b): row.spearman_margin for row in correlations.itertuples()}
    null_map = {(row.model_a, row.model_b): row for row in jaccard_null.itertuples()}
    for row in p20.itertuples():
        null = null_map[(row.model_a, row.model_b)]
        rho = corr_map[(row.model_a, row.model_b)]
        relation = "within Qwen3" if row.same_family else "cross-family"
        lines.append(
            f"| {row.model_a} vs {row.model_b} | {relation} | {rho:.3f} | {row.jaccard:.3f} | {100*row.overlap_pct:.2f} | {null.null_mean:.3f} | {null.observed_minus_null_mean:+.3f} | {null.empirical_upper_tail_p:.3f} |"
        )
    family_ci = pd.read_csv(output_dir / "family_consistency_bootstrap.csv").set_index("metric")
    within = summary["primary_results"]["within_qwen_mean_jaccard"]
    cross = summary["primary_results"]["cross_family_mean_jaccard"]
    difference = family_ci.loc["within_minus_cross_jaccard"]
    lines.extend([
        "",
        "## Family and shared-position summaries",
        "",
        f"Mean Qwen3 within-family Jaccard is {within:.3f}; mean cross-family Jaccard is {cross:.3f}. The within-minus-cross document-bootstrap 95% CI is [{difference.ci_low_95:.3f}, {difference.ci_high_95:.3f}]. Only Qwen3 has multiple targets in this set, so there is no within-Gemma or within-Ministral estimate.",
        "",
        f"The fraction marked as a decision point by at least one target from each of the three families (Qwen3, Gemma 3, Ministral 3) is {summary['primary_results']['cross_family_shared_position_fraction']:.3f}.",
        "",
        "The distribution by number of targets and a within-document permutation reference are in `shared_target_counts.csv`; pairwise overlap and permutation details are in `pairwise_overlap.csv` and `permutation_results.csv`. Margin-correlation bootstrap intervals are in `bootstrap_ci.csv`.",
        "",
        "## Morphological comparison",
        "",
    ])
    morph = outputs["morphological_enrichment.csv"]
    if len(morph):
        lines.append("Kiwi-derived morpheme-boundary and within-morpheme character offsets were matched within the same sampled eojeol. The comparison is associational and relies on automatic Kiwi segmentation.")
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
        "## Figures and artifacts",
        "",
        "Figures A–D visualize margin ranks, pairwise overlap, shared-target counts against the permutation baseline, and the matched morphological comparison. `summary.json` records model revisions, input token counts, exclusion counts, and output hashes.",
        "",
        "## Interpretation for KORSpec",
        "",
    ])
    null_mean = float(jaccard_null.null_mean.mean()) if len(jaccard_null) else float("nan")
    if cross > null_mean and difference.ci_low_95 > 0:
        outcome = "The observed pattern is closest to Outcome B: Qwen3 decision-point locations are more consistent within-family than across families, while cross-family overlap must be judged against its permutation baseline."
    elif cross > null_mean:
        outcome = "Cross-family overlap is above its within-document permutation reference on average, but the family contrast is uncertain; the evidence does not establish model-independent ambiguity locations."
    else:
        outcome = "The observed pattern is closest to Outcome C if most pairwise overlaps remain near their within-document permutation references; this supports interpreting the paper's existing RQ4 as conditional on a fixed target."
    lines.append(outcome)
    lines.extend([
        "",
        "### Proposed RQ6 paragraph",
        "",
        "On identical Korean text prefixes, target models identified low-margin decision points using within-model normalized margin ranks. Pairwise overlap and document-bootstrap intervals showed [insert measured within-Qwen and cross-family estimates], while a within-document permutation preserved each model's decision-point count. The comparison indicates [insert outcome from the results], so decision-point locations should be interpreted as [model-robust textual hotspots / partly family-dependent / target-specific] within the evaluated sample. This diagnostic does not establish intrinsic linguistic ambiguity, and the existing draft-disagreement results remain conditional on a fixed target model.",
        "",
    ])
    (output_dir / "cross_target_findings.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
