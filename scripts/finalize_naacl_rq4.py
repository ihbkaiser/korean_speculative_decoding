#!/usr/bin/env python3
"""Finish RQ4 artifacts from completed GLM log plus pseudo/human analyses."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.stats import norm

import analyze_naacl_rq4 as rq4
from analyze_naacl_cpu import GLOBAL_FORMULA, eligible_cross_within, load_pair_frame
from src.naacl_experiment import atomic_json, load_config, output_root


START_RE = re.compile(r"MODEL_START (.+?) rows=([0-9,]+) docs=([0-9,]+)")
DONE_RE = re.compile(r"MODEL_DONE (.+?) status=(\S+) converged=(\S+) AME=([-+0-9.eE]+) raw_AME=([-+0-9.eE]+) SE=([-+0-9.eE]+) warnings=(.*)")


def parse_fit_log(path: Path) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        start = START_RE.search(line)
        if start:
            found.setdefault(start.group(1), {}).update({
                "n_rows": int(start.group(2).replace(",", "")),
                "n_docs": int(start.group(3).replace(",", "")),
            })
        done = DONE_RE.search(line)
        if done:
            found.setdefault(done.group(1), {}).update({
                "status": done.group(2), "converged": done.group(3) == "True",
                "ame": float(done.group(4)), "raw_ame": float(done.group(5)),
                "se": float(done.group(6)), "warnings": done.group(7),
            })
    return found


def complete_case(rows: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    required = [
        "rejected", "alignment_class", "doc_id", "slot", "eojeol_token_count",
        "position", "eojeol_char_len", "draft_entropy", "target_entropy", "draft_margin", "target_margin",
    ] + (extra or [])
    data = rows.loc[:, required].copy()
    numeric = data.select_dtypes(include=[np.number]).columns
    data.loc[:, numeric] = data.loc[:, numeric].replace([np.inf, -np.inf], np.nan)
    return data.dropna(subset=required)


def row_from_log(
    label: str,
    rows: pd.DataFrame,
    fits: dict[str, dict[str, Any]],
    *,
    extra: list[str] | None = None,
    label_override: str | None = None,
) -> dict[str, Any]:
    metrics = fits[label]
    data = complete_case(rows, extra)
    if len(data) != metrics["n_rows"]:
        raise AssertionError(f"{label} complete-case row mismatch: {len(data)} != {metrics['n_rows']}")
    cross = data.alignment_class.eq("CROSS")
    rejected = data.rejected.astype(int)
    ame, se = metrics["ame"], metrics["se"]
    z = ame / se if se > 0 else np.nan
    return {
        "variant": label_override or label,
        "status": metrics["status"],
        "ame_pp": 100.0 * ame,
        "se_pp": 100.0 * se,
        "ci_low_pp": 100.0 * (ame - 1.96 * se),
        "ci_high_pp": 100.0 * (ame + 1.96 * se),
        "p_value": float(2.0 * norm.sf(abs(z))) if np.isfinite(z) else np.nan,
        "n_rows": int(len(data)), "n_docs": int(data.doc_id.nunique()),
        "n_cross": int(cross.sum()), "n_within": int((~cross).sum()),
        "cross_rejection_rate": float(rejected[cross].mean()),
        "within_rejection_rate": float(rejected[~cross].mean()),
        "fit_warnings": metrics.get("warnings", ""),
    }


def baseline_row(out: Path) -> dict[str, Any]:
    baseline = pd.read_csv(out / "rq1_table2b_cross_within.csv")
    row = baseline.loc[baseline.pair_id.eq("Q2")].iloc[0]
    return {
        "variant": "Q2_all_baseline", "status": "REUSED_TABLE2B",
        "ame_pp": 100.0 * row.AME_cross_minus_within,
        "se_pp": 100.0 * row.clustered_SE,
        "ci_low_pp": 100.0 * row.CI_low_95,
        "ci_high_pp": 100.0 * row.CI_high_95,
        "p_value": row.p_value, "n_rows": int(row.n_rows), "n_docs": int(row.n_docs),
        "n_cross": int(row.n_cross), "n_within": int(row.n_within),
        "cross_rejection_rate": row.cross_rejection_rate,
        "within_rejection_rate": row.within_rejection_rate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fit-log", type=Path, default=ROOT / "logs/naacl_experiments/20261012/CPU-RQ4-final.log")
    parser.add_argument("--pseudo-reps", type=int, default=rq4.PSEUDO_REPS)
    args = parser.parse_args()
    config = load_config(args.config)
    root = output_root(config, args.output_dir)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    fits = parse_fit_log(args.fit_log)

    print("RQ4_FINALIZE_LOAD_ALL", flush=True)
    all_frame, join_meta = load_pair_frame(root, "Q2", config)
    primary = eligible_cross_within(all_frame)
    baseline = baseline_row(out)
    table_rows = [baseline]

    strict = rq4.strict_rows(all_frame)
    if len(strict) == len(primary):
        strict_row = dict(baseline)
        strict_row.update({"variant": "strict_roundtrip_and_complete_eojeol", "status": "IDENTICAL_SAMPLE_REUSED_TABLE2B"})
    else:
        raise RuntimeError("Strict-alignment filter changed after the logged fit run; rebuild rather than reuse")
    table_rows.append(strict_row)

    single = primary.loc[
        primary.alignment_class.ne("CROSS")
        | pd.to_numeric(primary.crossed_boundary_count, errors="coerce").eq(1)
    ].copy()
    single_row = row_from_log("single_boundary_cross_only", single, fits)
    table_rows.append(single_row)

    top_rows = []
    for n in [10, 25, 50, 100]:
        label = f"top_{n}_cross_token_types_per_fine_boundary_removed"
        reduced, removal_meta = rq4.top_type_removal(primary, n, label=label)
        result = row_from_log(label, reduced, fits)
        result.update({k: v for k, v in removal_meta.items() if k != "variant"})
        table_rows.append(result)
        top_rows.append(result.copy())
        del reduced

    print("RQ4_FINALIZE_LOAD_COMMON", flush=True)
    common_frame, common_meta = load_pair_frame(root, "Q2", config, split_override="common")
    common = eligible_cross_within(common_frame)
    common_row = row_from_log("Q2_common20k", common, fits)
    table_rows.append(common_row)
    del common_frame, common

    freq_path = ROOT / "runs/20260926T184145Z_pilot1000/e1_token_frequency_cache.parquet"
    freq_meta_path = freq_path.with_name("e1_token_frequency_cache_metadata.json")
    frequency = pd.read_parquet(freq_path, columns=["token_id", "log_token_count", "token_rank_percentile"])
    frequency = frequency.rename(columns={
        "token_id": "target_token_id", "log_token_count": "external_log_token_count",
        "token_rank_percentile": "external_rank_percentile",
    })
    freq_rows = primary.merge(frequency, on="target_token_id", how="left", validate="many_to_one")
    if freq_rows.external_log_token_count.isna().any():
        raise ValueError("Frequency cache contains unmatched Q2 token IDs during finalization")
    freq_rows["external_freq_decile"] = pd.qcut(
        freq_rows.external_rank_percentile, 10, labels=False, duplicates="drop"
    ).astype(str)
    table_rows.append(row_from_log(
        "Wikipedia_frequency_spline", freq_rows, fits,
        extra=["external_log_token_count"],
    ))
    table_rows.append(row_from_log(
        "Wikipedia_frequency_decile", freq_rows, fits,
        extra=["external_freq_decile"],
    ))
    freq_meta = json.loads(freq_meta_path.read_text(encoding="utf-8"))
    del freq_rows, frequency

    fineweb_frequency_path = out / "fineweb2_disjoint_qwen4_token_frequency.parquet"
    fineweb_frequency_metadata_path = out / "fineweb2_disjoint_frequency_metadata.json"
    fineweb_frequency_meta: dict[str, Any] | None = None
    fineweb_frequency_fits: dict[str, Any] = {}
    if fineweb_frequency_path.exists() and fineweb_frequency_metadata_path.exists():
        print("RQ4_FINALIZE_FINEWEB2_FREQUENCY", flush=True)
        fineweb_frequency = pd.read_parquet(
            fineweb_frequency_path,
            columns=["token_id", "log_token_count", "token_rank_percentile"],
        ).rename(columns={
            "token_id": "target_token_id",
            "log_token_count": "fineweb_log_token_count",
            "token_rank_percentile": "fineweb_rank_percentile",
        })
        fineweb_rows = primary.merge(
            fineweb_frequency, on="target_token_id", how="left", validate="many_to_one",
        )
        fineweb_unmatched = int(fineweb_rows.fineweb_log_token_count.isna().sum())
        if fineweb_unmatched:
            raise ValueError(f"FineWeb2 frequency cache misses {fineweb_unmatched} Q2 proposal rows")
        fineweb_rows["fineweb_freq_decile"] = pd.qcut(
            fineweb_rows.fineweb_rank_percentile, 10, labels=False, duplicates="drop"
        ).astype(str)
        fineweb_spline_formula = GLOBAL_FORMULA + " + bs(fineweb_log_token_count, df=4, degree=3, include_intercept=False)"
        fineweb_spline_row, fineweb_spline_fit = rq4.fit_row(
            "FineWeb2_disjoint_frequency_spline", fineweb_rows,
            formula=fineweb_spline_formula,
        )
        table_rows.append(fineweb_spline_row)
        fineweb_decile_formula = GLOBAL_FORMULA + " + C(fineweb_freq_decile)"
        fineweb_decile_row, fineweb_decile_fit = rq4.fit_row(
            "FineWeb2_disjoint_frequency_decile", fineweb_rows,
            formula=fineweb_decile_formula,
        )
        table_rows.append(fineweb_decile_row)
        fineweb_frequency_meta = json.loads(fineweb_frequency_metadata_path.read_text(encoding="utf-8"))
        attempt_logs = [
            ROOT / "logs/naacl_experiments/20261012/CPU-RQ4-fineweb2-frequency.log",
            ROOT / "logs/naacl_experiments/20261012/CPU-RQ4-fineweb2-frequency-retry.log",
        ]
        validation_log = ROOT / "logs/naacl_experiments/20261012/CPU-RQ4-fineweb2-frequency-validation.log"
        fineweb_frequency_meta["execution_observation"] = {
            "attempt_logs": [str(path.relative_to(ROOT)) for path in attempt_logs if path.exists()],
            "success_marker_printed_before_shutdown_abort": all(
                path.exists()
                and "FINEWEB2_FREQUENCY_COMPLETE docs=40,000 tokens=43,497,725" in path.read_text(encoding="utf-8")
                and "Aborted" in path.read_text(encoding="utf-8")
                for path in attempt_logs
            ),
            "artifact_validation_log": str(validation_log.relative_to(ROOT)),
            "artifact_validation_passed": validation_log.exists() and "FREQUENCY_OUTPUT_VALIDATION_COMPLETE" in validation_log.read_text(encoding="utf-8"),
        }
        fineweb_frequency_fits = {
            "unmatched_rows": fineweb_unmatched,
            "spline_fit": fineweb_spline_fit,
            "decile_fit": fineweb_decile_fit,
        }
        del fineweb_rows, fineweb_frequency

    print("RQ4_FINALIZE_MECAB_LABELS", flush=True)
    mecab_rows, mecab_meta = rq4.mecab_relabel(all_frame)
    mecab_result = row_from_log("MeCab_ko_strict_alignment", mecab_rows, fits)
    table_rows.append(mecab_result)

    print("RQ4_FINALIZE_PSEUDO_BOUNDARY", flush=True)
    pseudo, pseudo_meta = rq4.pseudo_boundary_analysis(all_frame, reps=args.pseudo_reps)
    pseudo.to_csv(out / "rq4_pseudo_boundary_draws.csv", index=False)

    human_meta = rq4.create_human_sheet(all_frame, out / "rq4_human_validation_300.csv")
    proposal_sample_path = out / "rq4_proposal_side_sample_summary.json"
    proposal_sample = json.loads(proposal_sample_path.read_text(encoding="utf-8")) if proposal_sample_path.exists() else None
    proposal_full_path = out / "rq4_proposal_side_full40k_summary.json"
    proposal_full = json.loads(proposal_full_path.read_text(encoding="utf-8")) if proposal_full_path.exists() else None

    def proposal_table_row(result: dict[str, Any], variant: str, status: str) -> dict[str, Any]:
        fit_result = result.get("ame_fit") or {}
        if not fit_result:
            return {
                "variant": variant, "status": status,
                "ame_pp": np.nan, "se_pp": np.nan, "ci_low_pp": np.nan, "ci_high_pp": np.nan,
                "p_value": np.nan, "n_rows": np.nan, "n_docs": np.nan,
                "n_cross": np.nan, "n_within": np.nan,
                "cross_rejection_rate": np.nan, "within_rejection_rate": np.nan,
            }
        return {
            "variant": variant, "status": status,
            "ame_pp": 100.0 * float(fit_result["AME_cross_minus_within"]),
            "se_pp": 100.0 * float(fit_result["clustered_SE"]),
            "ci_low_pp": 100.0 * float(fit_result["CI_low_95"]),
            "ci_high_pp": 100.0 * float(fit_result["CI_high_95"]),
            "p_value": float(fit_result["p_value"]),
            "n_rows": int(fit_result["n_rows"]), "n_docs": int(fit_result["n_docs"]),
            "n_cross": int(fit_result["n_cross"]), "n_within": int(fit_result["n_within"]),
            "cross_rejection_rate": float(fit_result["cross_rejection_rate"]),
            "within_rejection_rate": float(fit_result["within_rejection_rate"]),
        }

    if proposal_sample is not None:
        table_rows.append({
            **proposal_table_row(
                proposal_sample, "proposal_side_random_1k_common_sample", "EXPLORATORY_SAMPLE_COMPLETE",
            ),
        })
    full_projection_complete = bool(
        proposal_full is not None
        and proposal_full.get("projection_error_blocks") == 0
        and (proposal_full.get("ame_fit") or {}).get("status") == "COMPLETE"
    )
    table_rows.append(proposal_table_row(
        proposal_full or {},
        "proposal_side_full40k_candidate_labels",
        "COMPLETE" if full_projection_complete else (
            "PARTIAL_PROJECTION_ERRORS" if proposal_full is not None else "PENDING_PROPOSAL_BLOCK_PROJECTION"
        ),
    ))

    table = pd.DataFrame(table_rows)
    table.to_csv(out / "rq4_table4_robustness.csv", index=False)
    pd.DataFrame(top_rows).to_csv(out / "rq4_top_type_removal.csv", index=False)
    mecab_meta["fit"] = fits["MeCab_ko_strict_alignment"]
    metadata = {
        "pair": "Q2", "split": "all / 40,000 FineWeb2 prompts", "seed": rq4.SEED,
        "primary_population": "eligible k=4 proposals with visible ALIGNED Kiwi class CROSS or WITHIN_SPLIT",
        "model": GLOBAL_FORMULA,
        "fit_log": str(args.fit_log.relative_to(ROOT)),
        "fit_log_records": fits,
        "strict_alignment": {"same_rows_as_primary": True, "fit_reused": "rq1_table2b_cross_within.csv", "rows": len(strict)},
        "common20k_join": common_meta,
        "frequency_source": freq_meta,
        "frequency_domain_caveat": (
            "The Wikipedia frequencies are retained as an out-of-domain proxy. A second set of variants uses token counts from the pinned FineWeb2 kor_Hang train slice with source indices after the frozen pool's scan window and normalized-text deduplication against all 40k prompts."
            if fineweb_frequency_meta is not None else
            "Frequency values come from the prior 1,000-article Korean Wikipedia cache, not the planned disjoint FineWeb2 frequency slice; the cache source is Wikipedia while Q2 is FineWeb2."
        ),
        "fineweb2_disjoint_frequency": {
            "metadata": fineweb_frequency_meta,
            "fits": fineweb_frequency_fits,
            "status": "COMPLETE" if fineweb_frequency_meta is not None else "PENDING",
        },
        "mecab": mecab_meta,
        "pseudo_boundary": pseudo_meta,
        "human_validation": human_meta,
        "proposal_side": {
            "status": "COMPLETE" if full_projection_complete else (
                "PARTIAL_FULL40K_PROJECTION" if proposal_full is not None else (
                    "PARTIAL_RANDOM_1K_COMPLETE_FULL40K_PENDING" if proposal_sample is not None else "PENDING_PROPOSAL_BLOCK_PROJECTION"
                )
            ),
            "sample_result": proposal_sample,
            "full40k_result": proposal_full,
            "full40k_status": "COMPLETE" if full_projection_complete else (
                "PARTIAL_PROJECTION_ERRORS" if proposal_full is not None else "PENDING_PROPOSAL_BLOCK_PROJECTION"
            ),
            "reason": "The 1k common-pool run is exploratory. The full result uses the frozen Q2 all-pool 40k prompts and projects each eligible teacher-forced draft argmax on its saved target-prefix context. Projection errors, if any, are reported explicitly.",
        },
        "join": join_meta,
        "outputs": [
            "rq4_table4_robustness.csv", "rq4_top_type_removal.csv", "rq4_pseudo_boundary_draws.csv",
            "rq4_human_validation_300.csv", "rq4_metadata.json",
            "rq4_proposal_side_sample_rows.parquet", "rq4_proposal_side_sample_summary.json",
            "rq4_proposal_side_full40k_rows.parquet", "rq4_proposal_side_full40k_summary.json",
            "fineweb2_disjoint_frequency_documents.parquet",
            "fineweb2_disjoint_qwen4_token_frequency.parquet",
            "fineweb2_disjoint_frequency_metadata.json",
        ],
        "notes": "Saved-continuation replay analysis within the V0 shared-context scope. Human annotation is pending; MeCab is an automatic analyzer comparison, not human validation. The log parser recovers converged fit AMEs and clustered SEs from completed model runs; support and observed rates are recomputed from the fit populations.",
    }
    atomic_json(out / "rq4_metadata.json", metadata)
    print(f"RQ4_FINALIZE_COMPLETE models={len(table)} pseudo={len(pseudo)} human={human_meta['rows']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
