#!/usr/bin/env python3
"""Matched-population comparison of aggregate and directional morphology features.

Computes 1 - boundary F1 on the exact candidate eojeol in the saved proposal-side
projection parquet, then compares four cluster-robust logistic specifications
on the same eligible proposal events for P1/P2 and Wikipedia/FLORES.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import statsmodels
from transformers import AutoTokenizer
from kiwipiepy import Kiwi

from scripts import analyze_h3 as h3
from src.e2_model_pairs import MODELS
from src.morphology import analyze_eojeol


SOURCE = ROOT / "runs/proposal_side_boundary_validation"
OUT = SOURCE / "aggregate_vs_directional"
EXPECTED_TOKENIZER_BACKEND = "41e00eccf531cffc2e562d38bdd879d41e5044ea279af5b73c6a32aabcc8fe04"
EXPECTED_VOCAB_SIZE = 151669
MODEL_ORDER = ["M0_CONTROLS", "M1_CONTROLS_PLUS_AGGREGATE", "M2_CONTROLS_PLUS_DIRECTION", "M3_CONTROLS_PLUS_BOTH"]
CONTROL_TERMS = [
    "C(fragmentation_bin, Treatment(reference='{frag_ref}'))",
    "C(morph_environment, Treatment(reference='{env_ref}'))",
    *h3.STRUCTURAL,
    *h3.ENTROPY,
    "bs(log_token_count, df=4, degree=3, include_intercept=False)",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tokenizer(allow_hub: bool, tokenizer_dir: Path | None) -> tuple[Any, dict[str, Any]]:
    model = MODELS["0.6B"]
    if tokenizer_dir is not None:
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model["id"], revision=model["revision"], use_fast=True,
            local_files_only=not allow_hub,
        )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("The pinned fast tokenizer is required to reproduce boundary offsets")
    backend_sha = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode("utf-8")).hexdigest()
    if backend_sha != EXPECTED_TOKENIZER_BACKEND or len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise RuntimeError(
            "Pinned tokenizer identity mismatch: "
            f"sha256={backend_sha}, vocab={len(tokenizer)}; expected "
            f"sha256={EXPECTED_TOKENIZER_BACKEND}, vocab={EXPECTED_VOCAB_SIZE}"
        )
    return tokenizer, {
        "model_id": model["id"], "revision": model["revision"],
        "tokenizer_class": tokenizer.__class__.__name__, "vocab_size": len(tokenizer),
        "backend_sha256": backend_sha,
        "source": str(tokenizer_dir) if tokenizer_dir is not None else "pinned Hugging Face revision",
    }


def load_or_compute_scores(tokenizer: Any, kiwi: Any, recompute: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_path = OUT / "eojeol_misalignment_cache.parquet"
    if cache_path.exists() and not recompute:
        cache = pd.read_parquet(cache_path)
        required = {"candidate_eojeol", "boundary_f1", "misalignment", "morpheme_count", "llm_token_count"}
        if not required.issubset(cache.columns):
            raise RuntimeError(f"Invalid score cache schema: missing {sorted(required-set(cache.columns))}")
        return cache, {"cache_reused": True, "unique_eojeols_scored": int(len(cache))}

    surfaces: set[str] = set()
    for workload in ("WIKIPEDIA", "FLORES"):
        for pair in ("P1", "P2"):
            frame = pd.read_parquet(SOURCE / f"proposal_boundary_rows_{workload.lower()}_{pair.lower()}.parquet")
            surfaces.update(frame.candidate_eojeol.dropna().astype(str).tolist())
    rows: list[dict[str, Any]] = []
    ordered = sorted(surfaces)
    for index, surface in enumerate(ordered, start=1):
        metrics = analyze_eojeol(surface, kiwi, tokenizer)
        rows.append({
            "candidate_eojeol": surface,
            "boundary_f1": float(metrics["boundary_f1"]),
            "misalignment": float(metrics["misalignment"]),
            "morpheme_count": int(metrics["morpheme_count"]),
            "llm_token_count": int(metrics["llm_token_count"]),
        })
        if index % 2500 == 0:
            print(f"Scored {index:,}/{len(ordered):,} unique candidate-eojeol strings", flush=True)
    cache = pd.DataFrame(rows)
    if not cache.misalignment.between(0.0, 1.0).all():
        raise AssertionError("Computed misalignment escaped [0, 1]")
    OUT.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(cache_path, index=False)
    return cache, {"cache_reused": False, "unique_eojeols_scored": int(len(cache))}


def eligible_population(frame: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    data = frame.merge(scores, on="candidate_eojeol", how="left", validate="many_to_one")
    single_boundary_cross = data.proposal_morph_class.eq("CROSS_MORPHEME") & data.num_boundaries_crossed.eq(1)
    keep = (
        data.projection_status.eq("VALID")
        & (data.proposal_morph_class.eq("WITHIN_SPLIT") | single_boundary_cross)
        & data.fragmentation_bin.astype(str).isin(h3.FRAGMENTATION_ORDER)
    )
    data = data.loc[keep].copy()
    data = data.rename(columns={
        "proposal_morph_class": "morph_class",
        "proposal_morph_environment": "morph_environment",
        "log_candidate_token_count": "log_token_count",
        "candidate_token_char_length": "token_char_length",
        "candidate_eojeol_char_length": "eojeol_char_length",
    })
    data["morph_environment"] = data.morph_environment.where(data.morph_environment.isin(h3.ENVIRONMENTS), "OTHER")
    data["morph_class"] = pd.Categorical(data.morph_class.astype(str), categories=["WITHIN_SPLIT", "CROSS_MORPHEME"])
    data, env_ref = h3._categories(data, "morph_environment", "LEXICAL_TO_LEXICAL")
    data, frag_ref = h3._categories(data, "fragmentation_bin", "2")
    required = [
        "sd_rejected", "prompt_id", "misalignment", "log_token_count", "fragmentation_bin",
        "morph_environment", "morph_class", *h3.STRUCTURAL, *h3.ENTROPY,
    ]
    data = data.dropna(subset=required).copy()
    data.attrs["environment_reference"] = env_ref
    data.attrs["fragmentation_reference"] = frag_ref
    return data


def formula_for(model_name: str, env_ref: str, frag_ref: str) -> str:
    controls = [term.format(env_ref=env_ref, frag_ref=frag_ref) for term in CONTROL_TERMS]
    if model_name == "M1_CONTROLS_PLUS_AGGREGATE":
        controls.append("misalignment")
    elif model_name == "M2_CONTROLS_PLUS_DIRECTION":
        controls.append("C(morph_class, Treatment(reference='WITHIN_SPLIT'))")
    elif model_name == "M3_CONTROLS_PLUS_BOTH":
        controls.extend(["misalignment", "C(morph_class, Treatment(reference='WITHIN_SPLIT'))"])
    return "sd_rejected ~ " + " + ".join(controls)


def coefficient_summary(fitted: Any, term: str) -> dict[str, float]:
    if fitted is None or not bool(getattr(fitted, "converged", False)) or term not in fitted.params.index:
        return {"estimate": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    ci = fitted.conf_int().loc[term]
    return {
        "estimate": float(fitted.params[term]), "se": float(fitted.bse[term]),
        "ci_low": float(ci.iloc[0]), "ci_high": float(ci.iloc[1]),
        "p_value": float(fitted.pvalues[term]),
    }


def directional_ame(fitted: Any, data: pd.DataFrame) -> dict[str, float]:
    if fitted is None or not bool(getattr(fitted, "converged", False)):
        return {"cross_probability": np.nan, "split_probability": np.nan, "ame": np.nan,
                "ame_se": np.nan, "ame_ci_low": np.nan, "ame_ci_high": np.nan, "ame_p_value": np.nan}
    p_cross, grad_cross = h3.predict_grad(fitted, data, "CROSS_MORPHEME")
    p_split, grad_split = h3.predict_grad(fitted, data, "WITHIN_SPLIT")
    ame = p_cross - p_split
    inference = h3.inference_from_gradient(fitted, ame, grad_cross - grad_split)
    return {
        "cross_probability": p_cross, "split_probability": p_split, "ame": ame,
        "ame_se": inference["se"], "ame_ci_low": inference["ci_low"],
        "ame_ci_high": inference["ci_high"], "ame_p_value": inference["p_value"],
    }


def fit_cell(workload: str, pair: str, data: pd.DataFrame) -> list[dict[str, Any]]:
    env_ref = str(data.attrs["environment_reference"])
    frag_ref = str(data.attrs["fragmentation_reference"])
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        formula = formula_for(model_name, env_ref, frag_ref)
        fitted, diag = h3.fit_clustered(data, formula, {
            "morph_environment": env_ref, "fragmentation_bin": frag_ref,
        })
        record: dict[str, Any] = {
            "workload": workload, "pair": pair, "model": model_name,
            "n_rows": int(diag.get("n_tokens") or 0), "n_prompts": int(diag.get("n_prompts") or 0),
            "rejection_rate": float(data.sd_rejected.mean()),
            "n_cross": int(data.morph_class.astype(str).eq("CROSS_MORPHEME").sum()),
            "n_split": int(data.morph_class.astype(str).eq("WITHIN_SPLIT").sum()),
            "converged": bool(diag.get("converged", False)), "rank": diag.get("rank"),
            "n_columns": diag.get("n_columns"), "fit_error": diag.get("fit_error"),
            "formula": formula,
        }
        if fitted is not None and bool(getattr(fitted, "converged", False)):
            record["log_likelihood"] = float(fitted.llf)
            record["aic"] = float(fitted.aic)
            record["mcfadden_pseudo_r2"] = float(1.0 - fitted.llf / fitted.llnull) if np.isfinite(fitted.llnull) and abs(fitted.llnull) > 1e-12 else np.nan
            misalignment_coef = coefficient_summary(fitted, "misalignment")
            class_term = next((name for name in fitted.params.index if "C(morph_class" in name and "CROSS_MORPHEME" in name), None)
            direction_coef = coefficient_summary(fitted, class_term) if class_term else {}
            direction_effect = directional_ame(fitted, data) if class_term else {}
            record.update({
                "misalignment_beta": misalignment_coef.get("estimate"),
                "misalignment_ci_low": misalignment_coef.get("ci_low"),
                "misalignment_ci_high": misalignment_coef.get("ci_high"),
                "misalignment_p": misalignment_coef.get("p_value"),
                "direction_log_odds": direction_coef.get("estimate"),
                "direction_p": direction_coef.get("p_value"),
                "adjusted_p_cross": direction_effect.get("cross_probability"),
                "adjusted_p_split": direction_effect.get("split_probability"),
                "direction_ame_pp": 100.0 * direction_effect.get("ame", np.nan),
                "direction_ame_ci_low_pp": 100.0 * direction_effect.get("ame_ci_low", np.nan),
                "direction_ame_ci_high_pp": 100.0 * direction_effect.get("ame_ci_high", np.nan),
                "direction_ame_p": direction_effect.get("ame_p_value"),
            })
        rows.append(record)
    return rows


def main() -> None:
    global SOURCE, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-hub", action="store_true", help="download the pinned tokenizer if it is not cached")
    parser.add_argument("--tokenizer-dir", type=Path, help="local tokenizer snapshot; identity is checked against the pinned backend SHA")
    parser.add_argument("--source-dir", type=Path, default=SOURCE, help="directory containing the four saved candidate-side parquet files")
    parser.add_argument("--output-dir", type=Path, default=OUT, help="directory for score cache, model tables, and report")
    parser.add_argument("--recompute-scores", action="store_true", help="ignore and overwrite the eojeol score cache")
    args = parser.parse_args()

    SOURCE = args.source_dir.resolve()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer, tok_meta = load_tokenizer(args.allow_hub, args.tokenizer_dir)
    kiwi_version = importlib.metadata.version("kiwipiepy")
    if kiwi_version != "0.24.0":
        raise RuntimeError(f"Expected Kiwi 0.24.0 from the projection run, got {kiwi_version}")
    kiwi = Kiwi()
    scores, score_meta = load_or_compute_scores(tokenizer, kiwi, args.recompute_scores)

    result_rows: list[dict[str, Any]] = []
    population_rows: list[pd.DataFrame] = []
    input_hashes: dict[str, str] = {}
    for workload in ("WIKIPEDIA", "FLORES"):
        for pair in ("P1", "P2"):
            path = SOURCE / f"proposal_boundary_rows_{workload.lower()}_{pair.lower()}.parquet"
            input_hashes[path.name] = sha256_file(path)
            frame = pd.read_parquet(path)
            data = eligible_population(frame, scores)
            if data.empty or data.morph_class.nunique() != 2 or data.sd_rejected.nunique() != 2:
                raise RuntimeError(f"Not estimable: {workload}/{pair} eligible rows={len(data)}")
            print(
                f"Fitting {workload}/{pair}: n={len(data):,}, prompts={data.prompt_id.nunique():,}, "
                f"cross={int(data.morph_class.astype(str).eq('CROSS_MORPHEME').sum()):,}, "
                f"split={int(data.morph_class.astype(str).eq('WITHIN_SPLIT').sum()):,}", flush=True,
            )
            result_rows.extend(fit_cell(workload, pair, data))
            data["workload"] = workload
            data["pair"] = pair
            population_rows.append(data)

    results = pd.DataFrame(result_rows)
    # Holm correction is reported separately for the four planned aggregate-score
    # tests and four directional AME tests; raw estimates remain in the table.
    from statsmodels.stats.multitest import multipletests
    for source_col, output_col in (("misalignment_p", "misalignment_p_holm"), ("direction_ame_p", "direction_ame_p_holm")):
        selected = results.model.eq("M1_CONTROLS_PLUS_AGGREGATE") if source_col == "misalignment_p" else results.model.eq("M3_CONTROLS_PLUS_BOTH")
        pvals = results.loc[selected, source_col]
        valid = pvals.notna()
        results.loc[pvals.index[valid], output_col] = multipletests(pvals[valid].astype(float).tolist(), method="holm")[1]

    population = pd.concat(population_rows, ignore_index=True)
    population.to_parquet(OUT / "candidate_population_with_misalignment.parquet", index=False)
    results.to_csv(OUT / "model_comparison.csv", index=False)

    audit = []
    for workload in ("WIKIPEDIA", "FLORES"):
        for pair in ("P1", "P2"):
            full = pd.read_parquet(SOURCE / f"proposal_boundary_rows_{workload.lower()}_{pair.lower()}.parquet")
            selected = population.loc[population.workload.eq(workload) & population.pair.eq(pair)]
            audit.append({
                "workload": workload, "pair": pair, "all_saved_rows": len(full),
                "projection_invalid_rows": int((~full.projection_status.eq("VALID")).sum()),
                "eligible_rows": len(selected), "eligible_prompts": int(selected.prompt_id.nunique()),
                "n_cross_single_boundary": int(selected.morph_class.astype(str).eq("CROSS_MORPHEME").sum()),
                "n_within_split": int(selected.morph_class.astype(str).eq("WITHIN_SPLIT").sum()),
                "rejection_rate": float(selected.sd_rejected.mean()),
                "unique_candidate_eojeols": int(selected.candidate_eojeol.nunique()),
            })
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(OUT / "population_audit.csv", index=False)

    meta = {
        "analysis": "4.2.1 matched-population aggregate alignment vs directional morphology",
        "unit": "observed proposal event; invalidated suffixes excluded in source projection",
        "population": "VALID projection; WITHIN_SPLIT or exactly-one-boundary CROSS_MORPHEME; fragmentation bins 2 through 8+; common complete-case rows for all models",
        "aggregate_feature": "1 - src.morphology.analyze_eojeol(candidate_eojeol, Kiwi, pinned fast tokenizer).boundary_f1",
        "directional_feature": "proposal_morph_class: CROSS_MORPHEME vs WITHIN_SPLIT",
        "models": MODEL_ORDER,
        "controls": [*h3.STRUCTURAL, *h3.ENTROPY, "fragmentation_bin", "morph_environment", "four-df spline of log_candidate_token_count"],
        "inference": "logit with prompt-clustered sandwich covariance; directional average marginal contrast standardized over the eligible cell population; Holm correction across four workload-pair cells for each primary feature family",
        "tokenizer": tok_meta,
        "kiwipiepy_version": kiwi_version,
        "statsmodels_version": statsmodels.__version__,
        "score_cache": score_meta,
        "input_sha256": input_hashes,
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = [
        "# 4.2.1: Aggregate alignment versus directional mismatch", "",
        "This is a matched-population offline reanalysis of saved P1/P2 speculative-decoding proposals. No model inference is performed.",
        "Misalignment is computed on each candidate's eojeol using the pinned Qwen3-0.6B fast tokenizer and Kiwi 0.24.0. The eligible outcome population contains valid actual proposals, including accepted proposals and the first rejection, and excludes invalidated suffix proposals.",
        "All four specifications use the same complete-case rows within each workload-pair cell. Standard errors are clustered by prompt. The reported directional AME is CROSS minus WITHIN_SPLIT in adjusted rejection probability.", "",
        "## Population audit", "", h3.md_table(audit_df, floatfmt=".4f"), "",
        "## Model comparison", "",
        h3.md_table(results[["workload", "pair", "model", "n_rows", "n_prompts", "rejection_rate", "aic", "mcfadden_pseudo_r2", "misalignment_beta", "misalignment_ci_low", "misalignment_ci_high", "misalignment_p", "misalignment_p_holm", "direction_ame_pp", "direction_ame_ci_low_pp", "direction_ame_ci_high_pp", "direction_ame_p", "direction_ame_p_holm", "converged"]], floatfmt=".4f"), "",
        "## Interpretation", "",
        "Treat this as associational evidence. Compare M1 with M0 to assess the scalar feature, M2 with M0 to assess direction, and M3 with M1/M2 to assess whether each feature contributes conditional information. Report results per cell and do not pool Wikipedia and FLORES as if they were the same workload.", "",
    ]
    (OUT / "report.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Wrote 4.2.1 results to {OUT}", flush=True)


if __name__ == "__main__":
    main()
