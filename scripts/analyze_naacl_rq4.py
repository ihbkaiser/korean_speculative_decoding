#!/usr/bin/env python3
"""Q2 RQ4 falsification and robustness analyses on saved k=4 replay data."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from analyze_naacl_cpu import (
    GLOBAL_FORMULA,
    eligible_cross_within,
    fit_cross_ame,
    load_pair_frame,
)
from src.naacl_experiment import atomic_json, load_config, output_root


SEED = 20261012
PSEUDO_REPS = 100
HUMAN_PER_STRATUM = 50
BOUNDARY_TYPES = [
    "NOMINAL_TO_PARTICLE", "PREDICATE_TO_ENDING", "ENDING_TO_ENDING",
    "LEXICAL_TO_LEXICAL", "OTHER",
]


def truthy(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def strict_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = eligible_cross_within(frame)
    return rows.loc[
        truthy(rows.roundtrip_exact)
        & truthy(rows.eojeol_matched)
    ].copy()


def add_boundary_key(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["boundary_key"] = np.where(
        out.alignment_class.eq("CROSS"),
        out.fine_boundary.astype("string"),
        out.nearest_fine_boundary.astype("string"),
    )
    out["boundary_key"] = out.boundary_key.replace({"<NA>": np.nan, "nan": np.nan, "None": np.nan})
    return out


def fit_row(label: str, rows: pd.DataFrame, *, formula: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    model_columns = [
        "rejected", "alignment_class", "doc_id", "slot", "eojeol_token_count",
        "position", "eojeol_char_len", "draft_entropy", "target_entropy",
        "draft_margin", "target_margin",
    ]
    model_columns.extend(
        col for col in ["fragmentation_bin", "external_log_token_count", "external_freq_decile"]
        if col in rows.columns
    )
    result = fit_cross_ame(rows.loc[:, model_columns], label=label, formula_override=formula)
    readable = {
        "variant": label,
        "status": result["status"],
        "ame_pp": 100.0 * result["AME_cross_minus_within"] if pd.notna(result["AME_cross_minus_within"]) else np.nan,
        "se_pp": 100.0 * result["clustered_SE"] if pd.notna(result["clustered_SE"]) else np.nan,
        "ci_low_pp": 100.0 * result["CI_low_95"] if pd.notna(result["CI_low_95"]) else np.nan,
        "ci_high_pp": 100.0 * result["CI_high_95"] if pd.notna(result["CI_high_95"]) else np.nan,
        "p_value": result["p_value"],
        "n_rows": result["n_rows"],
        "n_docs": result["n_docs"],
        "n_cross": result["n_cross"],
        "n_within": result["n_within"],
        "cross_rejection_rate": result["cross_rejection_rate"],
        "within_rejection_rate": result["within_rejection_rate"],
    }
    return readable, result


def top_type_removal(rows: pd.DataFrame, top_n: int, *, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    needed = [
        "alignment_class", "fine_boundary", "nearest_fine_boundary", "target_token_id",
        "rejected", "doc_id", "slot", "eojeol_token_count", "position", "eojeol_char_len",
        "draft_entropy", "target_entropy", "draft_margin", "target_margin",
    ]
    data = add_boundary_key(rows.loc[:, needed]).dropna(subset=["boundary_key"]).copy()
    cross = data.loc[data.alignment_class.eq("CROSS")]
    counts = cross.groupby(["boundary_key", "target_token_id"], observed=True).size().rename("n").reset_index()
    counts = counts.sort_values(["boundary_key", "n", "target_token_id"], ascending=[True, False, True])
    selected = counts.groupby("boundary_key", observed=True, sort=False).head(top_n)
    selected_keys = pd.MultiIndex.from_frame(selected[["boundary_key", "target_token_id"]])
    row_keys = pd.MultiIndex.from_frame(data[["boundary_key", "target_token_id"]])
    remove = row_keys.isin(selected_keys)
    kept = data.loc[~remove].copy()
    meta = {
        "variant": label,
        "top_n_per_fine_boundary": int(top_n),
        "selected_boundary_token_pairs": int(len(selected)),
        "removed_rows_both_classes": int(remove.sum()),
        "removed_cross_rows": int((remove & data.alignment_class.eq("CROSS").to_numpy()).sum()),
        "removed_within_rows": int((remove & data.alignment_class.eq("WITHIN_SPLIT").to_numpy()).sum()),
    }
    return kept, meta


def pseudo_boundary_analysis(rows: pd.DataFrame, reps: int = PSEUDO_REPS) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = strict_rows(rows)
    data = data.loc[data.alignment_class.isin(["CROSS", "WITHIN_SPLIT"])].copy()
    data["target_token_id"] = data.target_token_id.astype("int64")
    span = np.asarray(data.global_token_char_span.tolist(), dtype=np.int64)
    starts = data.token_start.to_numpy(dtype=np.int64)
    ends = data.token_end.to_numpy(dtype=np.int64)
    data["eojeol_global_start"] = span[:, 0] - starts
    data["pseudo_eojeol_key"] = data.doc_id.astype(str) + "\x1f" + data.eojeol_global_start.astype(str)
    group_codes, _ = pd.factorize(data.pseudo_eojeol_key, sort=False)
    data["pseudo_group"] = group_codes.astype(np.int32)

    first_rows = data.drop_duplicates("pseudo_group", keep="first")
    n_groups = int(group_codes.max()) + 1 if len(group_codes) else 0
    lengths = np.zeros(n_groups, dtype=np.int16)
    boundary_counts = np.zeros(n_groups, dtype=np.int16)
    for row in first_rows.itertuples(index=False):
        g = int(row.pseudo_group)
        length = int(row.eojeol_char_len)
        morphs = row.morpheme_char_spans
        if morphs is None or (isinstance(morphs, float) and np.isnan(morphs)):
            spans = []
        else:
            spans = []
            for value in morphs:
                try:
                    pair = np.asarray(value, dtype=np.int64).reshape(-1)
                except (TypeError, ValueError):
                    continue
                if len(pair) == 2:
                    spans.append((int(pair[0]), int(pair[1])))
            spans = sorted(spans)
        internal = [left[1] for left, right in zip(spans, spans[1:]) if left[1] == right[0] and 0 < left[1] < length]
        lengths[g] = length
        boundary_counts[g] = len(internal)

    # Keep eojeols with at least one Kiwi boundary; this matches the morphology
    # affected population, then randomizes the cut locations while preserving k.
    keep_groups = boundary_counts[group_codes] > 0
    data = data.loc[keep_groups].reset_index(drop=True)
    group_codes = data.pseudo_group.to_numpy(dtype=np.int32)
    starts = data.token_start.to_numpy(dtype=np.int64)
    ends = data.token_end.to_numpy(dtype=np.int64)
    rejection = data.rejected.astype(int).to_numpy(dtype=np.int8)
    boundary_counts = boundary_counts
    max_k = int(boundary_counts.max()) if len(boundary_counts) else 0
    if max_k <= 0 or data.empty:
        raise ValueError("No strict aligned eojeols with internal Kiwi boundaries for pseudo-boundary analysis")

    observed_cross = data.alignment_class.eq("CROSS").to_numpy()
    observed_within = data.alignment_class.eq("WITHIN_SPLIT").to_numpy()
    real_cross_rate = float(rejection[observed_cross].mean())
    real_within_rate = float(rejection[observed_within].mean())
    real_difference = real_cross_rate - real_within_rate

    shape_groups: dict[tuple[int, int], np.ndarray] = {}
    active_groups = np.flatnonzero(boundary_counts > 0)
    for k in np.unique(boundary_counts[active_groups]):
        for length in np.unique(lengths[active_groups[boundary_counts[active_groups] == k]]):
            ids = active_groups[(boundary_counts[active_groups] == k) & (lengths[active_groups] == length)]
            if int(k) > max(0, int(length) - 1):
                raise AssertionError(f"Kiwi boundary count {k} exceeds internal character cuts for length {length}")
            shape_groups[(int(k), int(length))] = ids

    rng = np.random.default_rng(SEED)
    draws: list[dict[str, Any]] = []
    for draw in range(reps):
        cuts = np.full((n_groups, max_k), -1, dtype=np.int16)
        for (k, length), ids in shape_groups.items():
            choices = rng.random((len(ids), length - 1), dtype=np.float32)
            selected = np.argpartition(choices, kth=k - 1, axis=1)[:, :k] + 1
            cuts[ids, :k] = selected.astype(np.int16)
        pseudo_cross = np.zeros(len(data), dtype=bool)
        row_groups = group_codes
        for j in range(max_k):
            cut = cuts[row_groups, j]
            pseudo_cross |= (starts < cut) & (ends > cut)
        pseudo_within = ~pseudo_cross
        if not pseudo_cross.any() or not pseudo_within.any():
            continue
        rate_cross = float(rejection[pseudo_cross].mean())
        rate_within = float(rejection[pseudo_within].mean())
        draws.append({
            "draw": draw + 1,
            "pseudo_cross_rejection_rate": rate_cross,
            "pseudo_within_rejection_rate": rate_within,
            "pseudo_cross_minus_within_pp": 100.0 * (rate_cross - rate_within),
            "pseudo_cross_n": int(pseudo_cross.sum()),
            "pseudo_within_n": int(pseudo_within.sum()),
        })
        if (draw + 1) % 20 == 0:
            print(f"PSEUDO_BOUNDARY_PROGRESS draws={draw + 1}/{reps}", flush=True)

    result = pd.DataFrame(draws)
    if len(result) != reps:
        raise RuntimeError(f"Only {len(result)} of {reps} pseudo-boundary draws had both classes")
    real_pp = 100.0 * real_difference
    p_empirical = float((1 + result.pseudo_cross_minus_within_pp.abs().ge(abs(real_pp)).sum()) / (reps + 1))
    metadata = {
        "seed": SEED,
        "requested_draws": reps,
        "real_cross_n": int(observed_cross.sum()),
        "real_within_n": int(observed_within.sum()),
        "pseudo_population_rows": int(len(data)),
        "pseudo_population_docs": int(data.doc_id.nunique()),
        "real_cross_rejection_rate": real_cross_rate,
        "real_within_rejection_rate": real_within_rate,
        "real_cross_minus_within_pp": real_pp,
        "pseudo_effect_q025_pp": float(result.pseudo_cross_minus_within_pp.quantile(.025)),
        "pseudo_effect_median_pp": float(result.pseudo_cross_minus_within_pp.median()),
        "pseudo_effect_q975_pp": float(result.pseudo_cross_minus_within_pp.quantile(.975)),
        "empirical_two_sided_tail_probability": p_empirical,
        "within_95_percent_pseudo_interval": bool(
            result.pseudo_cross_minus_within_pp.quantile(.025) <= real_pp <= result.pseudo_cross_minus_within_pp.quantile(.975)
        ),
        "pseudo_definition": "For each strict aligned eojeol containing at least one Kiwi boundary, sample the same number of distinct internal character cuts uniformly without replacement. A token is pseudo-CROSS if its character span contains any sampled cut; all other eligible CROSS/WITHIN target proposals in the morphology-affected population are pseudo-WITHIN. Report raw rate differences, not adjusted GLM effects.",
    }
    return result, metadata


def mecab_relabel(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    import mecab_ko

    data = strict_rows(rows)
    data = data.loc[data.alignment_class.isin(["CROSS", "WITHIN_SPLIT"])].copy()
    texts, text_codes = np.unique(data.eojeol_text.astype(str).to_numpy(), return_inverse=True)
    tagger = mecab_ko.Tagger()
    sequences: list[list[int] | None] = []
    valid_texts = 0
    for i, text in enumerate(texts):
        node = tagger.parseToNode(text)
        cursor = 0
        cuts = [0]
        surfaces: list[str] = []
        while node is not None:
            surface = str(node.surface or "")
            if surface:
                surfaces.append(surface)
                cursor += len(surface)
                cuts.append(cursor)
            node = node.next
        if "".join(surfaces) == text and cursor == len(text):
            sequences.append(cuts)
            valid_texts += 1
        else:
            sequences.append(None)
        if (i + 1) % 100_000 == 0:
            print(f"MECAB_PROGRESS unique_eojeols={i + 1:,}/{len(texts):,}", flush=True)

    max_edges = max((len(x) for x in sequences if x is not None), default=0)
    edge_matrix = np.full((len(texts), max_edges), -1, dtype=np.int16)
    valid_text = np.zeros(len(texts), dtype=bool)
    for i, edges in enumerate(sequences):
        if edges is not None:
            edge_matrix[i, :len(edges)] = edges
            valid_text[i] = True
    spans = np.asarray(data.token_char_span.tolist(), dtype=np.int64)
    starts, ends = spans[:, 0], spans[:, 1]
    valid_rows = valid_text[text_codes] & (starts >= 0) & (ends > starts)
    crosses = np.zeros(len(data), dtype=np.int16)
    start_edge = np.zeros(len(data), dtype=bool)
    end_edge = np.zeros(len(data), dtype=bool)
    for j in range(max_edges):
        edge = edge_matrix[text_codes, j]
        present = edge >= 0
        crosses += ((starts < edge) & (ends > edge) & present).astype(np.int16)
        start_edge |= (starts == edge) & present
        end_edge |= (ends == edge) & present
    mecab_class = np.full(len(data), "UNALIGNED", dtype=object)
    mecab_class[valid_rows & (crosses > 0)] = "CROSS"
    mecab_class[valid_rows & (crosses == 0) & start_edge & end_edge] = "EXACT"
    mecab_class[valid_rows & (crosses == 0) & ~(start_edge & end_edge)] = "WITHIN_SPLIT"
    data["mecab_class"] = mecab_class
    fit_data = data.loc[data.mecab_class.isin(["CROSS", "WITHIN_SPLIT"])].copy()
    fit_data["alignment_class"] = fit_data.mecab_class
    meta = {
        "analyzer": "mecab-ko",
        "package_version": importlib.metadata.version("mecab-ko"),
        "dictionary_version": importlib.metadata.version("mecab-ko-dic"),
        "unique_eojeol_texts": int(len(texts)),
        "concatenation_exact_unique_eojeols": int(valid_texts),
        "strict_rows": int(len(data)),
        "classified_rows": int(fit_data.shape[0]),
        "mecab_cross_rows": int(fit_data.mecab_class.eq("CROSS").sum()),
        "mecab_within_rows": int(fit_data.mecab_class.eq("WITHIN_SPLIT").sum()),
        "not_in_model_rows": int((~data.mecab_class.isin(["CROSS", "WITHIN_SPLIT"])).sum()),
        "label_definition": "MeCab-ko morpheme boundaries are cumulative character offsets from parseToNode surfaces. CROSS spans contain at least one internal MeCab boundary; exact endpoint matches are excluded; remaining valid token spans are WITHIN_SPLIT.",
    }
    return fit_data, meta


def create_human_sheet(frame: pd.DataFrame, output: Path) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    aligned = frame.loc[
        frame.alignment_join.eq("both")
        & frame.alignment_status.eq("ALIGNED")
        & truthy(frame.visible_candidate)
        & truthy(frame.roundtrip_exact)
        & truthy(frame.eojeol_matched)
        & frame.alignment_class.isin(["EXACT", "WITHIN_SPLIT", "CROSS"])
        & frame.rejected.notna()
    ].copy()
    aligned["outcome"] = np.where(aligned.rejected.astype(bool), "rejected", "accepted")
    selected: list[pd.DataFrame] = []
    stratum_counts = []
    for cls in ["EXACT", "WITHIN_SPLIT", "CROSS"]:
        for outcome in ["accepted", "rejected"]:
            group = aligned.loc[aligned.alignment_class.eq(cls) & aligned.outcome.eq(outcome)]
            if cls == "CROSS":
                n2p = group.loc[group.coarse_boundary.eq("NOMINAL_TO_PARTICLE")]
                other = group.loc[~group.coarse_boundary.eq("NOMINAL_TO_PARTICLE")]
                n2p_n = min(35, len(n2p))
                other_n = min(HUMAN_PER_STRATUM - n2p_n, len(other))
                sample = pd.concat([
                    n2p.sample(n=n2p_n, random_state=int(rng.integers(0, 2**31 - 1))) if n2p_n else n2p.head(0),
                    other.sample(n=other_n, random_state=int(rng.integers(0, 2**31 - 1))) if other_n else other.head(0),
                ])
                if len(sample) < HUMAN_PER_STRATUM:
                    remainder = group.loc[~group.index.isin(sample.index)]
                    add_n = min(HUMAN_PER_STRATUM - len(sample), len(remainder))
                    if add_n:
                        sample = pd.concat([sample, remainder.sample(n=add_n, random_state=int(rng.integers(0, 2**31 - 1)))])
            else:
                sample = group.sample(n=min(HUMAN_PER_STRATUM, len(group)), random_state=int(rng.integers(0, 2**31 - 1)))
            if len(sample) < HUMAN_PER_STRATUM:
                raise ValueError(f"Insufficient support for human validation stratum {cls}/{outcome}: {len(sample)}")
            sample = sample.sample(frac=1, random_state=int(rng.integers(0, 2**31 - 1))).copy()
            sample["stratum_class"] = cls
            sample["stratum_outcome"] = outcome
            selected.append(sample)
            stratum_counts.append({
                "class": cls, "outcome": outcome, "population_n": int(len(group)),
                "sample_n": int(len(sample)),
                "sample_n2p": int(sample.coarse_boundary.eq("NOMINAL_TO_PARTICLE").sum()) if cls == "CROSS" else 0,
            })
    sample = pd.concat(selected, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    def as_json(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "[]"
        return json.dumps(np.asarray(value, dtype=object).tolist(), ensure_ascii=False, default=str)

    sheet = pd.DataFrame({
        "annotation_id": [f"KOR-{i + 1:03d}" for i in range(len(sample))],
        "doc_id": sample.doc_id.astype(str),
        "generated_token_position": sample.position.astype(int),
        "target_token_id": sample.target_token_id.astype(int),
        "token_surface": sample.token_raw_text.astype(str),
        "eojeol": sample.eojeol_text.astype(str),
        "token_span_start": sample.token_start.astype(int),
        "token_span_end": sample.token_end.astype(int),
        "kiwi_morpheme_surfaces": sample.morpheme_surfaces.map(as_json),
        "kiwi_morpheme_pos": sample.morpheme_pos_tags.map(as_json),
        "kiwi_morpheme_spans": sample.morpheme_char_spans.map(as_json),
        "kiwi_class": sample.alignment_class.astype(str),
        "kiwi_boundary": sample.fine_boundary.fillna("").astype(str),
        "coarse_boundary": sample.coarse_boundary.fillna("").astype(str),
        "stratum_class": sample.stratum_class,
        "stratum_outcome": sample.stratum_outcome,
        "annotator_class_correct": "",
        "annotator_boundary_correct": "",
        "annotator_notes": "",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(output, index=False)
    return {
        "path": str(output.relative_to(ROOT)),
        "rows": int(len(sheet)),
        "strata": stratum_counts,
        "status": "PENDING_HUMAN_ANNOTATION",
        "sample_seed": SEED,
        "annotation_columns": ["annotator_class_correct", "annotator_boundary_correct", "annotator_notes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pseudo-reps", type=int, default=PSEUDO_REPS)
    args = parser.parse_args()
    config = load_config(args.config)
    root = output_root(config, args.output_dir)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    print("RQ4_LOAD_Q2_ALL", flush=True)
    all_frame, join_meta = load_pair_frame(root, "Q2", config)
    print(f"RQ4_Q2_ALL rows={len(all_frame):,} docs={all_frame.doc_id.nunique():,}", flush=True)
    primary = eligible_cross_within(all_frame)
    rows: list[dict[str, Any]] = []
    model_meta: dict[str, Any] = {}

    baseline_source = pd.read_csv(out / "rq1_table2b_cross_within.csv")
    baseline_source = baseline_source.loc[baseline_source.pair_id.eq("Q2")].iloc[0]
    baseline = {
        "variant": "Q2_all_baseline", "status": "REUSED_TABLE2B",
        "ame_pp": 100.0 * baseline_source.AME_cross_minus_within,
        "se_pp": 100.0 * baseline_source.clustered_SE,
        "ci_low_pp": 100.0 * baseline_source.CI_low_95,
        "ci_high_pp": 100.0 * baseline_source.CI_high_95,
        "p_value": baseline_source.p_value, "n_rows": baseline_source.n_rows,
        "n_docs": baseline_source.n_docs, "n_cross": baseline_source.n_cross,
        "n_within": baseline_source.n_within,
        "cross_rejection_rate": baseline_source.cross_rejection_rate,
        "within_rejection_rate": baseline_source.within_rejection_rate,
    }
    rows.append(baseline)

    strict = strict_rows(all_frame)
    print(f"RQ4_STRICT support rows={len(strict):,}", flush=True)
    if len(strict) == len(primary):
        strict_result = dict(baseline)
        strict_result.update({
            "variant": "strict_roundtrip_and_complete_eojeol",
            "status": "IDENTICAL_SAMPLE_REUSED_TABLE2B",
        })
        strict_meta = {"same_rows_as_primary": True, "rows": int(len(strict)), "fit_reused": "rq1_table2b_cross_within.csv"}
    else:
        strict_result, strict_meta = fit_row("strict_roundtrip_and_complete_eojeol", strict)
    rows.append(strict_result)
    model_meta["strict_roundtrip_and_complete_eojeol"] = strict_meta

    single = primary.loc[
        primary.alignment_class.ne("CROSS")
        | pd.to_numeric(primary.crossed_boundary_count, errors="coerce").eq(1)
    ].copy()
    single_result, single_meta = fit_row("single_boundary_cross_only", single)
    rows.append(single_result)
    model_meta["single_boundary_cross_only"] = single_meta

    type_rows = []
    for n in [10, 25, 50, 100]:
        reduced, remove_meta = top_type_removal(primary, n, label=f"top_{n}_cross_token_types_per_fine_boundary_removed")
        fit_result, raw_model = fit_row(remove_meta["variant"], reduced)
        fit_result.update({k: v for k, v in remove_meta.items() if k != "variant"})
        rows.append(fit_result)
        type_rows.append({**fit_result, **{k: v for k, v in remove_meta.items() if k != "variant"}})
        model_meta[remove_meta["variant"]] = raw_model
        del reduced, raw_model
        gc.collect()

    print("RQ4_LOAD_Q2_COMMON", flush=True)
    common_frame, common_meta = load_pair_frame(root, "Q2", config, split_override="common")
    common_rows = eligible_cross_within(common_frame)
    common_result, common_fit_meta = fit_row("Q2_common20k", common_rows)
    rows.append(common_result)
    model_meta["Q2_common20k"] = {"join": common_meta, "fit": common_fit_meta}
    del common_frame, common_rows
    gc.collect()

    freq_path = ROOT / "runs/20260926T184145Z_pilot1000/e1_token_frequency_cache.parquet"
    freq_meta_path = freq_path.with_name("e1_token_frequency_cache_metadata.json")
    frequency = pd.read_parquet(freq_path, columns=["token_id", "token_count", "log_token_count", "token_rank_percentile"])
    frequency = frequency.rename(columns={
        "token_id": "target_token_id", "log_token_count": "external_log_token_count",
        "token_rank_percentile": "external_rank_percentile",
    })
    freq_rows = primary.merge(frequency, on="target_token_id", how="left", validate="many_to_one")
    unmatched_frequency = int(freq_rows.external_log_token_count.isna().sum())
    if unmatched_frequency:
        raise ValueError(f"Wikipedia frequency cache misses {unmatched_frequency} Q2 proposal rows")
    freq_rows["external_freq_decile"] = pd.qcut(
        freq_rows.external_rank_percentile, 10, labels=False, duplicates="drop"
    ).astype(str)
    spline_formula = GLOBAL_FORMULA + " + bs(external_log_token_count, df=4, degree=3, include_intercept=False)"
    frequency_spline, frequency_spline_meta = fit_row("Wikipedia_frequency_spline", freq_rows, formula=spline_formula)
    rows.append(frequency_spline)
    decile_formula = GLOBAL_FORMULA + " + C(external_freq_decile)"
    frequency_decile, frequency_decile_meta = fit_row("Wikipedia_frequency_decile", freq_rows, formula=decile_formula)
    rows.append(frequency_decile)
    freq_meta = json.loads(freq_meta_path.read_text(encoding="utf-8"))
    model_meta["frequency_source"] = {
        "path": str(freq_path.relative_to(ROOT)), "metadata": freq_meta,
        "interpretation": "Out-of-domain Wikipedia frequency proxy for Q2 FineWeb2 data; the cache overlaps the separate W source domain and is not the planned disjoint FineWeb2 frequency slice.",
        "unmatched_rows": unmatched_frequency,
        "spline_fit": frequency_spline_meta, "decile_fit": frequency_decile_meta,
    }
    del freq_rows, frequency
    gc.collect()

    print("RQ4_MECAB_START", flush=True)
    mecab_data, mecab_meta = mecab_relabel(all_frame)
    mecab_result, mecab_fit_meta = fit_row("MeCab_ko_strict_alignment", mecab_data)
    rows.append(mecab_result)
    mecab_meta["fit"] = mecab_fit_meta
    model_meta["mecab"] = mecab_meta
    del mecab_data
    gc.collect()

    pseudo, pseudo_meta = pseudo_boundary_analysis(all_frame, reps=args.pseudo_reps)
    pseudo.to_csv(out / "rq4_pseudo_boundary_draws.csv", index=False)
    model_meta["pseudo_boundary"] = pseudo_meta

    human_meta = create_human_sheet(all_frame, out / "rq4_human_validation_300.csv")
    model_meta["human_validation"] = human_meta

    # Proposal-side morphology must project target-context draft proposal blocks.
    # Keep this row explicitly pending: rebuilding millions of full-context
    # candidate blocks is not a local-label substitution and is not conflated
    # with the target-side analysis above.
    rows.append({
        "variant": "proposal_side_draft_candidate_labels",
        "status": "PENDING_PROPOSAL_BLOCK_PROJECTION",
        "ame_pp": np.nan, "se_pp": np.nan, "ci_low_pp": np.nan, "ci_high_pp": np.nan,
        "p_value": np.nan, "n_rows": np.nan, "n_docs": np.nan, "n_cross": np.nan, "n_within": np.nan,
        "cross_rejection_rate": np.nan, "within_rejection_rate": np.nan,
    })
    model_meta["proposal_side"] = {
        "status": "PENDING_PROPOSAL_BLOCK_PROJECTION",
        "reason": "The available Q2 artifacts store teacher-forced candidate argmaxes; proposal-side labels require projecting each eligible candidate block through the tokenizer and Kiwi on its committed context. This is a separate, substantially larger projection than target-side alignment and has not been run.",
    }

    table = pd.DataFrame(rows)
    table.to_csv(out / "rq4_table4_robustness.csv", index=False)
    pd.DataFrame(type_rows).to_csv(out / "rq4_top_type_removal.csv", index=False)
    metadata = {
        "pair": "Q2", "split": "all / 40,000 FineWeb2 prompts", "seed": SEED,
        "primary_population": "eligible k=4 proposals with visible ALIGNED Kiwi class CROSS or WITHIN_SPLIT",
        "model": GLOBAL_FORMULA,
        "robustness_variants": [
            "strict complete-eojeol exact-roundtrip subset",
            "single-boundary CROSS subset",
            "top 10/25/50/100 CROSS token IDs removed per fine boundary from both classes",
            "Q2 common 20k split",
            "external Wikipedia frequency spline and decile controls",
            "MeCab-ko boundary labels on the strict aligned subset",
        ],
        "frequency_source": freq_meta,
        "frequency_domain_caveat": "Frequency values come from the prior 1,000-article Korean Wikipedia cache, not the planned disjoint FineWeb2 frequency slice; this is an external-domain proxy, while W uses Wikipedia.",
        "pseudo_boundary": pseudo_meta,
        "human_validation": human_meta,
        "proposal_side": model_meta["proposal_side"],
        "meCab_versions": {"mecab-ko": mecab_meta.get("package_version"), "mecab-ko-dic": mecab_meta.get("dictionary_version")},
        "join": join_meta,
        "outputs": [
            "rq4_table4_robustness.csv", "rq4_top_type_removal.csv", "rq4_pseudo_boundary_draws.csv",
            "rq4_human_validation_300.csv", "rq4_metadata.json",
        ],
        "model_details": model_meta,
        "notes": "All outcome comparisons are observational saved-continuation replay analyses, scoped to the V0 shared-context gate. Human annotation is pending; MeCab is an automatic analyzer comparison, not human validation.",
    }
    atomic_json(out / "rq4_metadata.json", metadata)
    print(f"RQ4_COMPLETE rows={len(table)} pseudo_draws={len(pseudo)} human_sheet={human_meta['rows']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
