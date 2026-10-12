#!/usr/bin/env python3
"""RQ5 greedy-SD rejection cost and counterfactual accepted-length analysis."""

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

from analyze_naacl_cpu import read_shards
from src.naacl_experiment import atomic_json


ALIGN_COLUMNS = [
    "doc_id", "generated_token_position", "alignment_class", "alignment_status",
    "visible_candidate", "coarse_boundary", "nearest_coarse_boundary",
    "crossed_boundary_count",
]
BLOCK_SIZES = [1, 2, 4, 8, 16]
BOOTSTRAP_REPS = 500
SEED = 20261012


def load_full_target_path(root: Path) -> pd.DataFrame:
    target_dir = root / "teacher_forced" / "generated" / "qwen4" / "qwen4" / "all"
    draft_dir = root / "teacher_forced" / "generated" / "qwen4" / "qwen06" / "all"
    target = read_shards(target_dir, ["doc_id", "position", "target_token_id"])
    draft = read_shards(draft_dir, ["doc_id", "position", "argmax_id"])
    target["doc_id"] = target.doc_id.astype(str)
    draft["doc_id"] = draft.doc_id.astype(str)
    target["position"] = target.position.astype("int64")
    draft["position"] = draft.position.astype("int64")
    full = target.merge(draft, on=["doc_id", "position"], how="inner", validate="one_to_one")
    align_dir = root / "aligned" / "kiwi" / "qwen4" / "all"
    alignment = read_shards(align_dir, ALIGN_COLUMNS)
    alignment["doc_id"] = alignment.doc_id.astype(str)
    alignment["position"] = alignment.generated_token_position.astype("int64")
    alignment = alignment.drop(columns=["generated_token_position"])
    full = full.merge(alignment, on=["doc_id", "position"], how="left", validate="one_to_one", indicator="alignment_join")
    full["rejected"] = full.argmax_id.astype("int64").ne(full.target_token_id.astype("int64"))
    aligned_visible = (
        full.alignment_join.eq("both")
        & full.alignment_status.eq("ALIGNED")
        & full.visible_candidate.fillna(False).astype(bool)
    )
    cross = aligned_visible & full.alignment_class.eq("CROSS")
    single_cross = cross & full.crossed_boundary_count.eq(1)
    multi_cross = cross & full.crossed_boundary_count.fillna(0).gt(1)
    within = aligned_visible & full.alignment_class.eq("WITHIN_SPLIT")
    full["class_group"] = np.select(
        [single_cross, multi_cross, within],
        ["CROSS", "MULTI_CROSS", "WITHIN"],
        default="OTHER_OR_UNALIGNED",
    )
    full["boundary"] = full.coarse_boundary.where(single_cross, full.nearest_coarse_boundary).astype("string")
    full.loc[multi_cross, "boundary"] = "MULTI_BOUNDARY"
    full["boundary"] = full.boundary.fillna("OTHER_OR_UNALIGNED").astype(str)
    full = full.sort_values(["doc_id", "position"], kind="stable").reset_index(drop=True)
    return full


def eligible_event_rows(root: Path, block_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = root / "simulated" / "Q2" / "all" / f"events_k{block_size}.parquet"
    events = pd.read_parquet(path, columns=[
        "doc_id", "position", "block_id", "slot", "eligible", "rejected", "invalidated_after",
    ])
    events["doc_id"] = events.doc_id.astype(str)
    events["position"] = events.position.astype("int64")
    eligible = events.loc[events.eligible.fillna(False).astype(bool)].copy()
    rejected = eligible.loc[eligible.rejected.fillna(False).astype(bool)].copy()
    suffix_counts = events.loc[events.invalidated_after.fillna(False).astype(bool)].groupby(
        ["doc_id", "block_id"], sort=False
    ).size().rename("invalidated_suffix_tokens").reset_index()
    rejected = rejected.merge(suffix_counts, on=["doc_id", "block_id"], how="left", validate="many_to_one")
    rejected["invalidated_suffix_tokens"] = rejected.invalidated_suffix_tokens.fillna(0).astype(int)
    rejected["wasted_draft_tokens"] = 1 + rejected.invalidated_suffix_tokens
    return eligible, rejected


def with_alignment(rows: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    align = full[[
        "doc_id", "position", "alignment_join", "alignment_status", "visible_candidate",
        "alignment_class", "class_group", "boundary",
    ]]
    return rows.merge(align, on=["doc_id", "position"], how="left", validate="one_to_one")


def expected_calls(prob_reject: np.ndarray, block_size: int) -> float:
    n = len(prob_reject)
    calls = np.zeros(n + 1, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        length = min(block_size, n - i)
        survival = 1.0
        value = 1.0
        for slot in range(length):
            p = float(prob_reject[i + slot])
            value += survival * p * calls[i + slot + 1]
            survival *= 1.0 - p
        # A fully accepted k-token block emits one bonus target token and skips
        # its verification position, except when the continuation has ended.
        if length == block_size and i + block_size < n:
            value += survival * calls[i + block_size + 1]
        calls[i] = value
    return float(calls[0])


def counterfactual_calls_by_document(
    full: pd.DataFrame,
    block_size: int,
    within_rates: dict[str, float],
    pooled_within_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_arrays = []
    token_counts = []
    doc_ids = []
    for doc_id, group in full.groupby("doc_id", sort=False):
        p = group.rejected.to_numpy(dtype=np.float64)
        mask = group.class_group.eq("CROSS").to_numpy()
        boundary = group.boundary.to_numpy(dtype=object)
        for i in np.flatnonzero(mask):
            p[i] = within_rates.get(str(boundary[i]), pooled_within_rate)
        group_arrays.append(expected_calls(p, block_size))
        token_counts.append(len(group))
        doc_ids.append(str(doc_id))
    return np.asarray(doc_ids, dtype=object), np.asarray(token_counts, dtype=np.int64), np.asarray(group_arrays, dtype=np.float64)


def bootstrap_ratio(token_counts: np.ndarray, calls: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(token_counts)
    draws = rng.integers(0, n, size=(BOOTSTRAP_REPS, n), endpoint=False)
    token_sums = token_counts[draws].sum(axis=1)
    call_sums = calls[draws].sum(axis=1)
    ratios = token_sums / call_sums
    low, high = np.quantile(ratios, [0.025, 0.975])
    return float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/naacl_experiments")
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "results/naacl_experiments/analysis")
    args = parser.parse_args()
    root, out = args.output_root, args.analysis_dir
    out.mkdir(parents=True, exist_ok=True)

    print("LOAD_Q2_TARGET_PATH", flush=True)
    full = load_full_target_path(root)
    print(f"TARGET_PATH rows={len(full):,} docs={full.doc_id.nunique():,}", flush=True)
    summaries = pd.read_csv(root / "simulated" / "Q2" / "all" / "summary_by_k.csv")
    summary_lookup = {int(row["Block size"]): row for _, row in summaries.iterrows()}
    family_rows = []
    counterfactual_rows = []
    metadata_rates: dict[str, Any] = {}
    rng = np.random.default_rng(SEED)

    for k in BLOCK_SIZES:
        print(f"COST_BLOCK_START k={k}", flush=True)
        eligible, rejected = eligible_event_rows(root, k)
        rejected = with_alignment(rejected, full)
        total_rejections = int(len(rejected))
        total_waste = int(rejected.wasted_draft_tokens.sum())
        rejected_group = rejected.groupby("boundary", dropna=False).agg(
            n_rejections=("position", "size"), wasted_draft_tokens=("wasted_draft_tokens", "sum")
        ).reset_index()
        rejected_group["block_size"] = k
        rejected_group["share_all_rejections"] = rejected_group.n_rejections / total_rejections if total_rejections else np.nan
        rejected_group["share_all_wasted_draft_tokens"] = rejected_group.wasted_draft_tokens / total_waste if total_waste else np.nan
        family_rows.extend(rejected_group.to_dict("records"))

        eligible = with_alignment(eligible, full)
        within = eligible.loc[
            eligible.alignment_join.eq("both")
            & eligible.alignment_status.eq("ALIGNED")
            & eligible.visible_candidate.fillna(False).astype(bool)
            & eligible.class_group.eq("WITHIN")
        ]
        pooled_rate = float(within.rejected.fillna(False).astype(bool).mean()) if len(within) else float("nan")
        rates = within.groupby("boundary").rejected.mean().to_dict()
        rates = {str(key): float(value) for key, value in rates.items() if pd.notna(value)}
        metadata_rates[str(k)] = {"within_rejection_rate_pooled": pooled_rate, "within_rejection_rate_by_boundary": rates}

        target_calls = int(summary_lookup[k]["Target calls"])
        target_tokens = int(summary_lookup[k]["Target tokens"])
        actual_mean = target_tokens / target_calls
        doc_ids, doc_tokens, doc_calls = counterfactual_calls_by_document(full, k, rates, pooled_rate)
        cf_mean = float(doc_tokens.sum() / doc_calls.sum())
        ci_low, ci_high = bootstrap_ratio(doc_tokens, doc_calls, rng)
        counterfactual_rows.append({
            "block_size": k,
            "target_tokens": target_tokens,
            "observed_target_calls": target_calls,
            "observed_mean_tokens_per_target_call": actual_mean,
            "counterfactual_expected_target_calls": float(doc_calls.sum()),
            "counterfactual_mean_tokens_per_target_call": cf_mean,
            "counterfactual_ci_low_95": ci_low,
            "counterfactual_ci_high_95": ci_high,
            "counterfactual_gain_tokens_per_call": cf_mean - actual_mean,
            "within_rate_pooled": pooled_rate,
            "n_docs": int(len(doc_ids)),
            "bootstrap_reps": BOOTSTRAP_REPS,
        })
        print(
            f"COST_BLOCK_DONE k={k} rejections={total_rejections:,} wasted={total_waste:,} "
            f"actual_mean={actual_mean:.5f} cf_mean={cf_mean:.5f} ci=[{ci_low:.5f},{ci_high:.5f}]",
            flush=True,
        )
        del eligible, rejected, within, doc_ids, doc_tokens, doc_calls
        gc.collect()

    family = pd.DataFrame(family_rows)
    family.to_csv(out / "rq5_cost_decomposition.csv", index=False)
    cf = pd.DataFrame(counterfactual_rows)
    cf.to_csv(out / "rq5_counterfactual_accepted_length.csv", index=False)
    metadata = {
        "pair": "Q2",
        "split": "all / 40,000 prompts",
        "outcome_scope": "offline greedy-SD replay on the saved Qwen4 target continuation; V0 shared-context scope caveat applies",
        "rejection_cost": "One rejected proposal plus the invalidated draft suffix work in its k-sized block are charged to the boundary family at the rejection position.",
        "counterfactual": "For each k, estimate within rejection probability by coarse boundary from eligible aligned visible WITHIN proposals. Replace each CROSS position's rejection indicator with an independent Bernoulli draw at that boundary's within rate; keep all other positions' observed mismatch indicators. Dynamic programming computes expected target calls per continuation, including correction and bonus-token transitions. Mean accepted length is total target tokens / expected target calls.",
        "bootstrap": f"{BOOTSTRAP_REPS} document-cluster resamples of expected token/call ratios; within-rate estimates are held fixed in each resample.",
        "random_seed": SEED,
        "within_rates": metadata_rates,
        "outputs": ["rq5_cost_decomposition.csv", "rq5_counterfactual_accepted_length.csv"],
    }
    atomic_json(out / "rq5_metadata.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
