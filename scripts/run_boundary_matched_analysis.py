#!/usr/bin/env python3
"""Prompt/rank-matched analysis of Korean N→P proposal rejection.

This is a controlled observational reanalysis of actual greedy-SD proposals.
It compares valid one-boundary N→P CROSS proposals with WITHIN_SPLIT proposals
assigned to the nearest N→P boundary, matching within the same workload, pair,
prompt, proposal slot, and (primary analysis) target fragmentation bin.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/proposal_side_boundary_validation/proposal_boundary_rows_all.parquet"
OUT = ROOT / "runs/proposal_boundary_matched_analysis"
SEED = 20261001
BOOTSTRAP_REPS = 5000

DISTANCE_COLS = [
    "draft_entropy", "target_entropy", "generation_position",
    "candidate_token_char_length", "candidate_eojeol_char_length", "log_candidate_token_count",
]
PRIMARY_KEYS = ["workload", "pair", "prompt_id", "proposal_slot", "fragmentation_bin"]
SENSITIVITY_KEYS = ["workload", "pair", "prompt_id", "proposal_slot"]
LENGTH_MATCH_KEYS = ["workload", "pair", "prompt_id", "proposal_slot", "candidate_token_char_length"]


def eligible_rows(source: Path) -> pd.DataFrame:
    df = pd.read_parquet(source)
    if "fragmentation_bin" not in df:
        df["fragmentation_bin"] = np.nan
    if "trace_source" in df:
        has_valid_fragmentation = df.fragmentation_bin.notna() | df.trace_source.eq("expansion")
    else:
        has_valid_fragmentation = df.fragmentation_bin.notna()
    valid = df.loc[
        df.projection_status.eq("VALID")
        & df.proposal_morph_class.isin(["CROSS_MORPHEME", "WITHIN_SPLIT"])
        & df.proposal_morph_environment.eq("NOMINAL_TO_PARTICLE")
        & has_valid_fragmentation
    ].copy()
    is_cross = valid.proposal_morph_class.eq("CROSS_MORPHEME")
    valid["condition"] = np.where(is_cross, "NP_CROSS", "WITHIN_SPLIT")
    valid = valid.loc[~is_cross | valid.num_boundaries_crossed.eq(1)].copy()
    valid["fragmentation_bin"] = valid.fragmentation_bin.astype("string")
    valid = valid.dropna(subset=DISTANCE_COLS).reset_index(drop=True)
    valid["source_row_id"] = valid.index.astype(int)
    return valid


def scale_features(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    values = df[DISTANCE_COLS].astype(float)
    scales = values.std(ddof=0).replace(0, 1.0).to_dict()
    scales = {k: float(v) if np.isfinite(v) and v > 0 else 1.0 for k, v in scales.items()}
    z = values.to_numpy(dtype=float) / np.array([scales[c] for c in DISTANCE_COLS])
    return z, scales


def match_one_to_one(df: pd.DataFrame, keys: list[str], scaled: np.ndarray,
                     include_fragment_penalty: bool = False) -> pd.DataFrame:
    """Minimum-cost 1:1 matching without replacement inside exact strata."""
    index_to_position = {idx: i for i, idx in enumerate(df.index)}
    output: list[dict] = []
    match_id = 0
    grouped = df.groupby(keys, sort=True, dropna=False, observed=True)
    for _, group in grouped:
        treated = group.loc[group.condition.eq("NP_CROSS")]
        controls = group.loc[group.condition.eq("WITHIN_SPLIT")]
        if treated.empty or controls.empty:
            continue
        tpos = np.array([index_to_position[i] for i in treated.index])
        cpos = np.array([index_to_position[i] for i in controls.index])
        delta = scaled[tpos, None, :] - scaled[None, cpos, :]
        cost = np.sqrt(np.square(delta).mean(axis=2))
        if include_fragment_penalty and treated.fragmentation_bin.notna().all() and controls.fragmentation_bin.notna().all():
            tfrag = treated.fragmentation_bin.astype(str).to_numpy()[:, None]
            cfrag = controls.fragmentation_bin.astype(str).to_numpy()[None, :]
            cost = cost + (tfrag != cfrag) * 1.0
        ti, ci = linear_sum_assignment(cost)
        for i, j in zip(ti, ci):
            tr = treated.iloc[int(i)]
            cr = controls.iloc[int(j)]
            output.append({
                "match_id": match_id,
                "workload": str(tr.workload), "pair": str(tr.pair),
                "prompt_id": int(tr.prompt_id), "proposal_slot": int(tr.proposal_slot),
                "cross_source_row_id": int(tr.source_row_id),
                "within_source_row_id": int(cr.source_row_id),
                "cross_rejected": int(tr.sd_rejected), "within_rejected": int(cr.sd_rejected),
                "risk_difference": int(tr.sd_rejected) - int(cr.sd_rejected),
                "match_distance": float(cost[int(i), int(j)]),
                "cross_fragmentation_bin": None if pd.isna(tr.fragmentation_bin) else str(tr.fragmentation_bin),
                "within_fragmentation_bin": None if pd.isna(cr.fragmentation_bin) else str(cr.fragmentation_bin),
            })
            match_id += 1
    return pd.DataFrame(output)


def bootstrap_mean(frame: pd.DataFrame, reps: int = BOOTSTRAP_REPS) -> tuple[float, float, float]:
    if frame.empty:
        return float("nan"), float("nan"), float("nan")
    prompt = frame.groupby("prompt_id", sort=True).risk_difference.agg(["sum", "count"])
    sums = prompt["sum"].to_numpy(dtype=float)
    counts = prompt["count"].to_numpy(dtype=float)
    point = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(SEED)
    pick = rng.integers(0, len(prompt), size=(reps, len(prompt)))
    boot = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return point, float(lo), float(hi)


def summarize_matches(matches: pd.DataFrame, analysis_name: str) -> pd.DataFrame:
    rows = []
    for workload in ("WIKIPEDIA", "FLORES"):
        for pair in ("P1", "P2", "P1+P2"):
            sub = matches.loc[matches.workload.eq(workload)]
            if pair != "P1+P2":
                sub = sub.loc[sub.pair.eq(pair)]
            if sub.empty:
                continue
            estimate, low, high = bootstrap_mean(sub)
            cross_reject = float(sub.cross_rejected.mean())
            within_reject = float(sub.within_rejected.mean())
            rows.append({
                "analysis": analysis_name, "workload": workload, "pair": pair,
                "n_matched": int(len(sub)), "n_prompts": int(sub.prompt_id.nunique()),
                "cross_reject_rate": cross_reject, "within_reject_rate": within_reject,
                "risk_difference": estimate, "ci_low": low, "ci_high": high,
            })
    return pd.DataFrame(rows)


def balance_table(source: pd.DataFrame, matches: pd.DataFrame, analysis_name: str) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(columns=["analysis", "workload", "covariate", "cross_mean", "within_mean", "smd", "abs_smd"])
    left = source.set_index("source_row_id", drop=False)
    cross = left.loc[matches.cross_source_row_id].copy()
    within = left.loc[matches.within_source_row_id].copy()
    rows = []
    for workload in ("WIKIPEDIA", "FLORES"):
        t = cross.loc[cross.workload.eq(workload)]
        c = within.loc[within.workload.eq(workload)]
        if t.empty:
            continue
        reference = source.loc[source.workload.eq(workload), DISTANCE_COLS].astype(float)
        scales = reference.std(ddof=0).replace(0, 1.0)
        for col in DISTANCE_COLS:
            smd = float((t[col].mean() - c[col].mean()) / scales[col])
            rows.append({"analysis": analysis_name, "workload": workload, "covariate": col,
                         "cross_mean": float(t[col].mean()), "within_mean": float(c[col].mean()),
                         "smd": smd, "abs_smd": abs(smd)})
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}"
        return str(v)
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines.extend("| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    df = eligible_rows(args.source)
    scaled, scales = scale_features(df)
    primary = match_one_to_one(df, LENGTH_MATCH_KEYS, scaled, include_fragment_penalty=True)
    strict_source = df.loc[df.fragmentation_bin.notna()].copy()
    strict_scaled, _ = scale_features(strict_source)
    strict_fragment = match_one_to_one(strict_source, PRIMARY_KEYS, strict_scaled)
    sensitivity = match_one_to_one(df, SENSITIVITY_KEYS, scaled, include_fragment_penalty=True)
    primary["analysis"] = "PRIMARY_SAME_LENGTH_SLOT"
    strict_fragment["analysis"] = "STRICT_FRAGMENT_BIN"
    sensitivity["analysis"] = "SENSITIVITY_SLOT_ONLY"
    all_matches = pd.concat([primary, strict_fragment, sensitivity], ignore_index=True)
    all_matches.to_csv(OUT / "matched_pairs.csv", index=False)
    summary = pd.concat([
        summarize_matches(primary, "PRIMARY_SAME_LENGTH_SLOT"),
        summarize_matches(strict_fragment, "STRICT_FRAGMENT_BIN"),
        summarize_matches(sensitivity, "SENSITIVITY_SLOT_ONLY"),
    ], ignore_index=True)
    summary.to_csv(OUT / "matched_summary.csv", index=False)
    balance = pd.concat([
        balance_table(df, primary, "PRIMARY_SAME_LENGTH_SLOT"),
        balance_table(df, strict_fragment, "STRICT_FRAGMENT_BIN"),
        balance_table(df, sensitivity, "SENSITIVITY_SLOT_ONLY"),
    ], ignore_index=True)
    balance.to_csv(OUT / "covariate_balance.csv", index=False)
    audit = {
        "source_file": str(args.source),
        "eligible_cross": int(df.condition.eq("NP_CROSS").sum()),
        "eligible_within_split": int(df.condition.eq("WITHIN_SPLIT").sum()),
        "primary_matched": int(len(primary)),
        "primary_cross_matched": int(primary.cross_source_row_id.nunique()),
        "primary_control_reuse": int(len(primary) - primary.within_source_row_id.nunique()),
        "primary_covariate_max_abs_smd_by_workload": balance.loc[balance.analysis.eq("PRIMARY_SAME_LENGTH_SLOT")].groupby("workload").abs_smd.max().to_dict(),
        "strict_fragment_matched": int(len(strict_fragment)),
        "sensitivity_matched": int(len(sensitivity)),
        "sensitivity_control_reuse": int(len(sensitivity) - sensitivity.within_source_row_id.nunique()),
        "keys_primary": LENGTH_MATCH_KEYS,
        "keys_strict_fragment_sensitivity": PRIMARY_KEYS,
        "keys_sensitivity": SENSITIVITY_KEYS,
        "distance_covariates": DISTANCE_COLS,
        "fragmentation_available_for_primary_rows": int(df.fragmentation_bin.notna().sum()),
        "scale_sd": scales,
        "bootstrap_prompt_cluster_reps": BOOTSTRAP_REPS,
        "seed": SEED,
    }
    (OUT / "matching_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    primary_summary = summary.loc[summary.analysis.eq("PRIMARY_SAME_LENGTH_SLOT")]
    primary_pooled = primary_summary.loc[primary_summary.pair.eq("P1+P2")].set_index("workload")
    primary_balance = balance.loc[balance.analysis.eq("PRIMARY_SAME_LENGTH_SLOT")]
    max_smd = primary_balance.groupby("workload").abs_smd.max().to_dict()
    ci_above_zero = bool((primary_pooled.ci_low > 0).all()) if not primary_pooled.empty else False
    pair_directions_positive = bool(primary_summary.loc[primary_summary.pair.isin(["P1", "P2"]), "risk_difference"].gt(0).all())
    balance_passes = bool(primary_balance.abs_smd.le(0.10).all())
    if set(("WIKIPEDIA", "FLORES")).issubset(primary_pooled.index):
        wiki = primary_pooled.loc["WIKIPEDIA"]
        flores = primary_pooled.loc["FLORES"]
        result_text = (
            f"The primary same-length/same-slot match estimates {100*wiki.risk_difference:+.1f} pp "
            f"(95% CI {100*wiki.ci_low:+.1f} to {100*wiki.ci_high:+.1f}) on Wiki and "
            f"{100*flores.risk_difference:+.1f} pp (95% CI {100*flores.ci_low:+.1f} to {100*flores.ci_high:+.1f}) "
            f"on FLORES, from {int(wiki.n_matched)} and {int(flores.n_matched)} matched proposal pairs. "
            f"Maximum primary absolute SMDs are {max_smd.get('WIKIPEDIA', float('nan')):.2f} on Wiki and "
            f"{max_smd.get('FLORES', float('nan')):.2f} on FLORES. "
            + ("The prespecified conditional-association criterion is met." if ci_above_zero and pair_directions_positive and balance_passes else
               "The prespecified criterion is not met: the primary CIs include zero and covariate balance remains inadequate.")
        )
    else:
        result_text = "The primary dataset-level summary is unavailable; no decision can be made."
    report = [
        "# Prompt- and rank-matched N→P proposal analysis", "",
        "## Experiment card", "",
        "- **Question:** Among actual Korean draft proposals near nominal→particle structure, does an N→P boundary-crossing proposal have a higher target-rejection rate than a proposal that stays within one morpheme?",
        "- **Hypothesis:** N→P CROSS proposals retain higher rejection after matching on prompt, model pair, exact proposal slot and exact candidate-token character length, then minimizing observed difficulty/position/frequency differences.",
        "- **Baseline / variant:** WITHIN_SPLIT proposal near an N→P boundary / valid one-boundary N→P CROSS proposal.",
        "- **Primary metric:** Matched difference in rejection probability (CROSS minus WITHIN_SPLIT), with prompt-cluster bootstrap 95% CI.",
        "- **Decision rule:** Direction is consistent with the hypothesis only if both dataset CIs are above zero, both P1/P2 point estimates are positive within each dataset, and every primary covariate has absolute SMD ≤ 0.10. This is evidence of a robust conditional association, not a morphology-only causal effect.",
        "- **Data / split:** Saved actual greedy SD proposals for Wiki and English→Korean FLORES, P1/P2; proposals after the first rejection are already excluded by the validated source population.",
        "- **Cheapest rung / budget:** CPU reanalysis only; no new generation or GPU inference.",
        "- **Exploratory-only:** The slot-only sensitivity match and any claim that morphology itself causes rejection.", "",
        "## Matching design", "",
        f"Primary matches are 1:1 without replacement within workload × model pair × prompt × proposal slot × exact candidate-token character length. The Hungarian minimum-distance assignment uses standardized draft/target entropy, generation position, eojeol character length, and candidate-token frequency; a target-fragmentation mismatch penalty is applied only when fragmentation labels are available. A strict-fragment sensitivity uses only rows with observed fragmentation and matches it exactly; a broader sensitivity omits exact length and fragmentation matching. Outcomes are not used in matching. Confidence intervals resample prompts as clusters. Source: `{args.source}`.", "",
        f"Eligible rows: {audit['eligible_cross']:,} CROSS and {audit['eligible_within_split']:,} WITHIN_SPLIT. Primary matched pairs: {audit['primary_matched']:,}; control reuse: {audit['primary_control_reuse']}. Strict-fragment matched pairs: {audit['strict_fragment_matched']:,}. Broader sensitivity matched pairs: {audit['sensitivity_matched']:,}; control reuse: {audit['sensitivity_control_reuse']}.", "",
        "## Result", "", result_text, "",
        "## Matched rejection estimates", "", markdown_table(summary), "",
        "## Primary covariate balance", "", markdown_table(balance.loc[balance.analysis.eq("PRIMARY_SAME_LENGTH_SLOT")]), "",
        "## Interpretation limits", "",
        "Matching controls source prompt, model pair, proposal slot, candidate character length, and the recorded difficulty/position covariates. Target fragmentation is available only for the original traces. Candidate token identity and its linguistic context remain different between matched proposals, and unmeasured features may remain. Therefore this analysis tests replication of a conditional association; it cannot establish that crossing a morpheme boundary itself causes rejection. The separate proposal-suppression intervention estimates a policy effect and does not identify this token-level causal effect.", "",
    ]
    (OUT / "matched_analysis_report.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
