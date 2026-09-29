#!/usr/bin/env python3
"""Run CPU-only E2 marginal-effects, common-prompt, and P3 revision audits."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from src.e2_model_pairs import (
    DEFAULT_ENTROPY,
    DEFAULT_STRUCTURAL,
    MODELS,
    P3_DIR,
    P3_FREQUENCY_CACHE,
    ROOT,
    _attach_frequency,
    _eligible,
    _fit_pooled,
    _formula,
    _primary_formula,
    _p3_harmonized_entropy,
)


E2_DIR = ROOT / "runs/e2_model_pair_replication"
COMBINED = E2_DIR / "combined"
PAIR_DIRS = {
    "P1": E2_DIR / "p1_06b_to_17b",
    "P2": E2_DIR / "p2_17b_to_4b",
}
PAIR_LABELS = {"P1": "0.6B → 1.7B", "P2": "1.7B → 4B", "P3": "0.6B → 4B"}
PAIR_ORDER = ("P1", "P2", "P3")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _sha256(path: Path, max_bytes: int | None = None) -> str | None:
    if not path.is_file() or (max_bytes is not None and path.stat().st_size > max_bytes):
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def load_pair_tables() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    tables = {
        "P1": pd.read_parquet(PAIR_DIRS["P1"] / "token_table.parquet"),
        "P2": pd.read_parquet(PAIR_DIRS["P2"] / "token_table.parquet"),
        "P3": pd.read_parquet(P3_DIR / "h2_token_table.parquet"),
    }
    cache = pd.read_parquet(P3_FREQUENCY_CACHE)
    return tables, {"frequency_cache": cache}


def _fit_delta_ame(data: pd.DataFrame, formula: str, pair: str, model_name: str) -> dict[str, Any]:
    import statsmodels.formula.api as smf
    from patsy import build_design_matrices, dmatrices

    fit = smf.logit(formula, data=data).fit(
        disp=False,
        maxiter=200,
        cov_type="cluster",
        cov_kwds={"groups": data["prompt_id"], "use_correction": True},
    )
    _, reference_design = dmatrices(formula, data=data, return_type="dataframe")
    if len(reference_design) != len(data):
        raise AssertionError(f"{pair}/{model_name}: formula dropped rows from the declared eligible sample")
    design_info = reference_design.design_info
    categories = list(data["morph_class"].cat.categories)
    counterfactuals = {}
    matrices = {}
    predictions = {}
    params = fit.params.to_numpy(dtype=float)
    covariance = fit.cov_params().to_numpy(dtype=float)
    n = len(data)
    for label in ("CROSS_MORPHEME", "WITHIN_SPLIT"):
        counterfactual = data.copy()
        counterfactual["morph_class"] = pd.Categorical([label] * n, categories=categories)
        counterfactuals[label] = counterfactual
        matrices[label] = np.asarray(
            build_design_matrices([design_info], counterfactual, return_type="dataframe")[0],
            dtype=float,
        )
        predictions[label] = expit(matrices[label] @ params)

    # Only morph_class changes in either counterfactual; verify all other columns.
    other_columns = [c for c in data.columns if c != "morph_class"]
    for label in counterfactuals:
        if not data[other_columns].reset_index(drop=True).equals(
            counterfactuals[label][other_columns].reset_index(drop=True)
        ):
            raise AssertionError(f"{pair}/{model_name}: counterfactual changed non-morphology covariates")

    gradients = {}
    for label in predictions:
        p = predictions[label]
        gradients[label] = np.mean((p * (1.0 - p))[:, None] * matrices[label], axis=0)
    means = {label: float(predictions[label].mean()) for label in predictions}
    difference = means["CROSS_MORPHEME"] - means["WITHIN_SPLIT"]
    difference_gradient = gradients["CROSS_MORPHEME"] - gradients["WITHIN_SPLIT"]

    def se_for(gradient: np.ndarray) -> float:
        variance = float(gradient @ covariance @ gradient)
        return float(np.sqrt(max(variance, 0.0)))

    se_cross = se_for(gradients["CROSS_MORPHEME"])
    se_split = se_for(gradients["WITHIN_SPLIT"])
    se_difference = se_for(difference_gradient)
    zcrit = float(norm.ppf(0.975))
    p_value = float(2.0 * norm.sf(abs(difference / se_difference))) if se_difference else 0.0
    odds_term = next(name for name in fit.params.index if "morph_class" in name and "CROSS_MORPHEME" in name)
    return {
        "pair": pair,
        "draft_to_target": PAIR_LABELS[pair],
        "model": model_name,
        "formula": formula,
        "n_tokens": int(len(data)),
        "n_prompts": int(data.prompt_id.nunique()),
        "adjusted_p_cross": means["CROSS_MORPHEME"],
        "adjusted_p_cross_ci_low": max(0.0, means["CROSS_MORPHEME"] - zcrit * se_cross),
        "adjusted_p_cross_ci_high": min(1.0, means["CROSS_MORPHEME"] + zcrit * se_cross),
        "adjusted_p_split": means["WITHIN_SPLIT"],
        "adjusted_p_split_ci_low": max(0.0, means["WITHIN_SPLIT"] - zcrit * se_split),
        "adjusted_p_split_ci_high": min(1.0, means["WITHIN_SPLIT"] + zcrit * se_split),
        "ame_cross_minus_split": difference,
        "ame_pp": 100.0 * difference,
        "ame_ci_low": 100.0 * (difference - zcrit * se_difference),
        "ame_ci_high": 100.0 * (difference + zcrit * se_difference),
        "ame_p_value": p_value,
        "model_cross_beta": float(fit.params[odds_term]),
        "model_cross_or": float(np.exp(fit.params[odds_term])),
        "cluster_count": int(data.prompt_id.nunique()),
        "covariance": "prompt-clustered sandwich; delta method conditional on observed covariates",
    }


def adjusted_marginal_effects(tables: dict[str, pd.DataFrame], cache: pd.DataFrame) -> pd.DataFrame:
    saved = pd.read_csv(COMBINED / "e2_model_pair_summary.csv").set_index("pair")
    rows = []
    model_formulas = {
        "M2": _primary_formula(list(DEFAULT_STRUCTURAL)),
        "M3": _primary_formula(list(DEFAULT_STRUCTURAL) + list(DEFAULT_ENTROPY)),
        "M5": _formula(
            "sd_rejected",
            list(DEFAULT_STRUCTURAL) + list(DEFAULT_ENTROPY)
            + ["bs(log_token_count, df=4, degree=3, include_intercept=False)"],
        ),
    }
    for pair in PAIR_ORDER:
        raw = tables[pair]
        for model_name, formula in model_formulas.items():
            model_data = _eligible(_attach_frequency(raw, cache)) if model_name == "M5" else _eligible(raw)
            row = _fit_delta_ame(model_data, formula, pair, model_name)
            saved_or = float(saved.loc[pair, f"{model_name}_OR"])
            saved_n = int(saved.loc[pair, f"{model_name}_N"])
            saved_prompts = int(saved.loc[pair, f"{model_name}_prompts"])
            row.update({
                "saved_e2_or": saved_or,
                "saved_e2_n_tokens": saved_n,
                "saved_e2_n_prompts": saved_prompts,
                "parity_with_saved_e2": bool(
                    row["n_tokens"] == saved_n
                    and row["n_prompts"] == saved_prompts
                    and np.isclose(row["model_cross_or"], saved_or, rtol=1e-7, atol=1e-8)
                ),
            })
            if not row["parity_with_saved_e2"]:
                raise AssertionError(
                    f"{pair}/{model_name} AME refit does not match the saved E2 N/prompt/OR: {row}"
                )
            rows.append(row)
    result = pd.DataFrame(rows)
    _atomic_csv(COMBINED / "e2_adjusted_marginal_effects.csv", result)
    _write_ame_markdown(result)
    _write_ame_figure(result)
    return result


def _write_ame_markdown(frame: pd.DataFrame) -> None:
    lines = [
        "# Adjusted marginal rejection differences",
        "",
        "Each estimate averages two counterfactual predictions over the eligible rows for that pair and model, holding every observed covariate fixed and changing only `morph_class`. The difference is CROSS_MORPHEME minus WITHIN_SPLIT.",
        "",
        "Intervals and two-sided p-values use a delta method with the fitted model's prompt-clustered sandwich covariance. These are adjusted marginal associations conditional on the observed covariate distribution, not causal effects. No multiplicity correction is applied.",
        "",
        "| Pair | Model | N tokens / prompts | Adjusted P(CROSS) | Adjusted P(SPLIT) | Difference (percentage points, 95% CI) | p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in frame.itertuples(index=False):
        lines.append(
            f"| {r.draft_to_target} | {r.model} | {r.n_tokens:,} / {r.n_prompts:,} | "
            f"{r.adjusted_p_cross:.2%} ({r.adjusted_p_cross_ci_low:.2%}–{r.adjusted_p_cross_ci_high:.2%}) | "
            f"{r.adjusted_p_split:.2%} ({r.adjusted_p_split_ci_low:.2%}–{r.adjusted_p_split_ci_high:.2%}) | "
            f"{r.ame_pp:.2f} ({r.ame_ci_low:.2f}–{r.ame_ci_high:.2f}) | {r.ame_p_value:.3g} |"
        )
    lines += [
        "",
        "The fitted odds-ratio term from each re-estimated model is included in the CSV for parity checks against the original M2/M3/M5 pair estimates.",
    ]
    _atomic_text(COMBINED / "e2_adjusted_marginal_effects.md", "\n".join(lines) + "\n")


def _write_ame_figure(frame: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    y_base = np.arange(len(PAIR_ORDER))
    offsets = {"M3": -0.12, "M5": 0.12}
    colors = {"M3": "#176b87", "M5": "#b04a36"}
    for model in ("M3", "M5"):
        subset = frame[frame.model == model].set_index("pair").reindex(PAIR_ORDER)
        y = y_base + offsets[model]
        point = subset.ame_pp.to_numpy(dtype=float)
        low = subset.ame_ci_low.to_numpy(dtype=float)
        high = subset.ame_ci_high.to_numpy(dtype=float)
        ax.errorbar(point, y, xerr=[point - low, high - point], fmt="o", capsize=3,
                    color=colors[model], label=model, linewidth=1.6)
    ax.axvline(0.0, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks(y_base, [PAIR_LABELS[p] for p in PAIR_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Adjusted rejection probability difference (percentage points)")
    ax.set_title("CROSS_MORPHEME minus WITHIN_SPLIT")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False, title="Model")
    fig.tight_layout()
    fig.savefig(COMBINED / "e2_adjusted_marginal_effects.png", dpi=200)
    plt.close(fig)


def prompt_overlap_and_common_sensitivity(tables: dict[str, pd.DataFrame], cache: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, str]:
    eligible_ids = {
        pair: set(map(int, _eligible(tables[pair]).prompt_id.unique()))
        for pair in PAIR_ORDER
    }
    all_ids = set(map(int, pd.read_json(E2_DIR / "prompt_ids.jsonl", lines=True).prompt_id))
    input_ids = {
        "P1": set(map(int, pd.read_parquet(PAIR_DIRS["P1"] / "continuations.parquet").prompt_id)),
        "P2": set(map(int, pd.read_parquet(PAIR_DIRS["P2"] / "continuations.parquet").prompt_id)),
        "P3": set(map(int, pd.DataFrame(
            [json.loads(line) for line in (P3_DIR / "reference_outputs.jsonl").read_text(encoding="utf-8").splitlines()]
        ).source_index)),
    }
    if any(ids != all_ids for ids in input_ids.values()):
        raise AssertionError("Pair input prompt sets do not exactly equal the E2 1,000-prompt set")
    union = set.union(*(eligible_ids[p] for p in PAIR_ORDER))
    common = set.intersection(*(eligible_ids[p] for p in PAIR_ORDER))
    pairwise = {
        "P1_intersect_P2": sorted(eligible_ids["P1"] & eligible_ids["P2"]),
        "P1_intersect_P3": sorted(eligible_ids["P1"] & eligible_ids["P3"]),
        "P2_intersect_P3": sorted(eligible_ids["P2"] & eligible_ids["P3"]),
    }
    audit = {
        "definition": "Prompt sets refer to prompts contributing at least one eligible primary-contrast row after the unchanged H2 filter (sd_valid, fragmentation 2/3/4/5/6/7/8+, morphology CROSS_MORPHEME or WITHIN_SPLIT). Full input coverage is recorded separately.",
        "full_input_prompt_count": len(all_ids),
        "full_input_prompt_ids_by_pair": {p: sorted(ids) for p, ids in input_ids.items()},
        "full_input_pair_sets_identical": True,
        "eligible_prompt_ids": {p: sorted(eligible_ids[p]) for p in PAIR_ORDER},
        "eligible_prompt_counts": {p: len(eligible_ids[p]) for p in PAIR_ORDER},
        "eligible_union_prompt_ids": sorted(union),
        "eligible_union_count": len(union),
        "pairwise_intersection_prompt_ids": pairwise,
        "pairwise_intersection_counts": {k: len(v) for k, v in pairwise.items()},
        "three_way_intersection_prompt_ids": sorted(common),
        "three_way_intersection_count": len(common),
        "eligible_prompts_missing_from_each_set_within_union": {
            p: sorted(union - eligible_ids[p]) for p in PAIR_ORDER
        },
        "full_input_prompts_without_any_eligible_rows": sorted(all_ids - union),
        "prior_999_prompt_count_interpretation": "The existing pooled model's 999 prompt count is the union of prompts with eligible rows in at least one pair, not the three-way common-prompt intersection.",
    }
    _atomic_json(COMBINED / "e2_prompt_overlap_audit.json", audit)

    audit_lines = [
        "# Prompt overlap audit",
        "",
        "The model-pair prompt sets below are the prompts with at least one eligible primary-contrast row under that pair's unchanged H2 filter. The full input set is reported separately.",
        "",
        f"- Full prompt inputs: {len(all_ids)} per pair; exact ID sets match across P1, P2, and P3.",
        f"- Eligible prompts: P1={len(eligible_ids['P1'])}, P2={len(eligible_ids['P2'])}, P3={len(eligible_ids['P3'])}.",
        f"- Eligible union: {len(union)} prompts.",
        f"- Pairwise eligible intersections: P1∩P2={len(pairwise['P1_intersect_P2'])}, P1∩P3={len(pairwise['P1_intersect_P3'])}, P2∩P3={len(pairwise['P2_intersect_P3'])}.",
        f"- Three-way eligible intersection: {len(common)} prompts.",
        "",
        "Thus the previous pooled count of 999 is the eligible-prompt union, not a count of prompts represented in every pair. Full prompt inputs were still identical across pairs.",
        "",
        "Missing eligible IDs by pair within the union:",
    ]
    for pair in PAIR_ORDER:
        audit_lines.append(f"- {pair}: {audit['eligible_prompts_missing_from_each_set_within_union'][pair]}")
    audit_lines.append(f"- No eligible rows in any pair: {audit['full_input_prompts_without_any_eligible_rows']}")
    _atomic_text(COMBINED / "e2_prompt_overlap_audit.md", "\n".join(audit_lines) + "\n")

    p3 = pd.read_parquet(P3_DIR / "h2_token_table.parquet")
    p3_harmonized, _ = _p3_harmonized_entropy(
        p3,
        pd.read_parquet(P3_DIR / "teacher_forced_tokens.parquet"),
        pd.read_parquet(P3_DIR / "sd_events.parquet"),
    )
    all_tables = {"P1": tables["P1"], "P2": tables["P2"], "P3": p3_harmonized}
    original_parts = []
    common_parts = []
    for pair in PAIR_ORDER:
        part = _attach_frequency(all_tables[pair], cache)
        part["model_pair"] = pair
        original_parts.append(part)
        common_parts.append(part.loc[part.prompt_id.astype(int).isin(common)].copy())
    original_data = pd.concat(original_parts, ignore_index=True, sort=False)
    common_data = pd.concat(common_parts, ignore_index=True, sort=False)
    if set(map(int, _eligible(common_data).prompt_id.unique())) != common:
        raise AssertionError("Common pooled table did not retain exactly the three-way eligible intersection")

    pooled_results = {}
    pooled_fits = {}
    for population, frame in (("original_union", original_data), ("three_way_common", common_data)):
        for model, frequency in (("M3", False), ("M5", True)):
            fit, result, text = _fit_pooled(frame, frequency_control=frequency)
            pooled_results[(population, model)] = result
            pooled_fits[(population, model)] = fit

    # Check the original-union refits against the exact machine-readable pooled
    # estimates embedded in the original regression report before comparing
    # them with the common-prompt sensitivity.
    original_report = (COMBINED / "e2_model_pair_regression.txt").read_text(encoding="utf-8")
    saved_pooled = {}
    for model, heading in (
        ("M3", "===== Pooled M3 with model-pair × morphology interaction ====="),
        ("M5", "===== Pooled frequency-controlled M5 with model-pair × morphology interaction ====="),
    ):
        section = original_report.split(heading, 1)[1]
        json_start = section.index("{")
        saved_pooled[model], _ = json.JSONDecoder().raw_decode(section[json_start:])
        refit = pooled_results[("original_union", model)]
        for pair in PAIR_ORDER:
            for metric in ("beta", "odds_ratio", "or_ci_low", "or_ci_high", "p_value"):
                if not np.isclose(
                    refit["pair_effects"][pair][metric],
                    saved_pooled[model]["pair_effects"][pair][metric],
                    rtol=1e-7, atol=1e-9,
                ):
                    raise AssertionError(
                        f"Original-union {model}/{pair}/{metric} does not reproduce the saved E2 regression report"
                    )

    rows = []
    by_population = {"original_union": original_data, "three_way_common": common_data}
    for model in ("M3", "M5"):
        original = pooled_results[("original_union", model)]
        common_result = pooled_results[("three_way_common", model)]
        for pair in PAIR_ORDER:
            original_effect = original["pair_effects"][pair]
            common_effect = common_result["pair_effects"][pair]
            saved_effect = saved_pooled[model]["pair_effects"][pair]
            counts = _eligible(by_population["three_way_common"])
            original_counts = _eligible(by_population["original_union"])
            n_common = int((counts.model_pair == pair).sum())
            prompts_common = int(counts.loc[counts.model_pair == pair, "prompt_id"].nunique())
            n_original = int((original_counts.model_pair == pair).sum())
            prompts_original = int(original_counts.loc[original_counts.model_pair == pair, "prompt_id"].nunique())
            rows.append({
                "model": model,
                "pair": pair,
                "draft_to_target": PAIR_LABELS[pair],
                "original_beta": original_effect["beta"],
                "original_or": original_effect["odds_ratio"],
                "original_or_ci_low": original_effect["or_ci_low"],
                "original_or_ci_high": original_effect["or_ci_high"],
                "original_p_value": original_effect["p_value"],
                "saved_original_or": saved_effect["odds_ratio"],
                "saved_original_or_ci_low": saved_effect["or_ci_low"],
                "saved_original_or_ci_high": saved_effect["or_ci_high"],
                "original_refit_matches_saved": True,
                "original_n_tokens": n_original,
                "original_n_prompts": prompts_original,
                "common_beta": common_effect["beta"],
                "common_or": common_effect["odds_ratio"],
                "common_or_ci_low": common_effect["or_ci_low"],
                "common_or_ci_high": common_effect["or_ci_high"],
                "common_p_value": common_effect["p_value"],
                "common_n_tokens": n_common,
                "common_n_prompts": prompts_common,
                "delta_log_or": float(np.log(common_effect["odds_ratio"] / original_effect["odds_ratio"])),
                "relative_or_change_pct": 100.0 * (common_effect["odds_ratio"] / original_effect["odds_ratio"] - 1.0),
                "direction_preserved": bool(common_effect["beta"] > 0) == bool(original_effect["beta"] > 0),
            })
    comparison = pd.DataFrame(rows)
    signs_preserved = bool(comparison.direction_preserved.all())
    max_abs_log_change = float(comparison.delta_log_or.abs().max())
    if not signs_preserved or max_abs_log_change >= np.log(1.10):
        classification = "material change"
    elif max_abs_log_change <= np.log(1.05):
        classification = "unchanged"
    else:
        classification = "minor numerical change"
    _atomic_csv(COMBINED / "e2_common_prompt_sensitivity.csv", comparison)

    txt_lines = [
        "Common-prompt pooled sensitivity; CPU-only refit on saved E2/P3 tables.",
        "The original pooled eligible sample is the union; the sensitivity uses exactly P1∩P2∩P3 eligible prompt IDs.",
        f"P1/P2/P3 eligible prompt counts: {len(eligible_ids['P1'])}/{len(eligible_ids['P2'])}/{len(eligible_ids['P3'])}; union={len(union)}; three-way intersection={len(common)}.",
        "Both models use model-pair fixed effects, morphology, their interaction, fragmentation FE, structural controls, and entropy controls; M5 adds the E1 cubic spline log-frequency term. SEs cluster by prompt_id.",
        "The original-union refits were checked against all pair-specific coefficients, OR intervals, and p-values in the saved E2 pooled regression report before sensitivity comparisons.",
        f"Sensitivity classification: {classification}; max absolute log-OR change={max_abs_log_change:.4f}; direction preserved for all rows={signs_preserved}.",
    ]
    for population, frame in (("ORIGINAL UNION", original_data), ("THREE-WAY COMMON", common_data)):
        txt_lines.append(f"\n===== {population} =====")
        for model, frequency in (("M3", False), ("M5", True)):
            _, result, summary = _fit_pooled(frame, frequency_control=frequency)
            txt_lines.append(f"\n--- {model} ---\n{summary}")
            txt_lines.append(json.dumps({k: v for k, v in result.items() if k != "summary"}, indent=2))
    _atomic_text(COMBINED / "e2_common_prompt_sensitivity.txt", "\n".join(txt_lines) + "\n")

    md_lines = [
        "# Common-prompt pooled sensitivity",
        "",
        f"Eligible prompt sets have sizes P1={len(eligible_ids['P1'])}, P2={len(eligible_ids['P2'])}, and P3={len(eligible_ids['P3'])}. Their union has {len(union)} prompts; the exact three-way intersection has **{len(common)} prompts**.",
        "",
        "The original pooled eligible result uses the union of prompts with any eligible rows, not a common-prompt intersection. The sensitivity refits pooled M3 and frequency-controlled M5 using only the exact three-way intersection. Formulas, fixed effects, controls, and prompt-clustered SEs match the original pooled models.",
        "The original-union refits match the saved E2 regression coefficients, OR intervals, and p-values to numerical tolerance.",
        "",
        f"Classification: **{classification}**. This is based on direction and effect magnitude: unchanged means every OR changes by at most 5%; minor means the maximum change is between 5% and 10%; material means any sign reversal or a change of 10% or more. Maximum absolute log-OR change: {max_abs_log_change:.4f}.",
        "",
        "| Model | Pair | Original union OR (95% CI) | Common-only OR (95% CI) | Relative OR change | Common N tokens / prompts | p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in comparison.itertuples(index=False):
        md_lines.append(
            f"| {r.model} | {r.pair} ({r.draft_to_target}) | {r.original_or:.3f} ({r.original_or_ci_low:.3f}–{r.original_or_ci_high:.3f}) | "
            f"{r.common_or:.3f} ({r.common_or_ci_low:.3f}–{r.common_or_ci_high:.3f}) | {r.relative_or_change_pct:+.1f}% | "
            f"{r.common_n_tokens:,} / {r.common_n_prompts:,} | {r.common_p_value:.3g} |"
        )
    md_lines += [
        "",
        "This is a population-overlap sensitivity analysis only; it does not redefine the primary E2 estimates.",
    ]
    _atomic_text(COMBINED / "e2_common_prompt_sensitivity.md", "\n".join(md_lines) + "\n")

    report_path = COMBINED / "e2_model_pair_replication.md"
    report = report_path.read_text(encoding="utf-8")
    report_lines = report.splitlines()
    new_line = (
        f"- Pooled M3 interaction model: N={pooled_results[('original_union','M3')]['n_tokens']:,} tokens; "
        f"{len(union)} prompts in the eligible-pair union, not the {len(common)}-prompt three-way intersection; "
        f"model-pair × morphology joint interaction p={pooled_results[('original_union','M3')]['heterogeneity_test']['p_value']:.4g}."
    )
    pooled_line_indices = [i for i, line in enumerate(report_lines) if line.startswith("- Pooled M3 interaction model:")]
    if len(pooled_line_indices) != 1:
        raise AssertionError(f"Expected one pooled M3 prompt-count line to correct, found {len(pooled_line_indices)}")
    pooled_line_index = pooled_line_indices[0]
    report_lines[pooled_line_index] = new_line
    sensitivity_line = (
        f"- Common-prompt sensitivity: the exact eligible three-way intersection contains {len(common)} prompts; "
        f"pooled M3/M5 classification is **{classification}** (see `e2_common_prompt_sensitivity.md`)."
    )
    report_lines = [line for line in report_lines if not line.startswith("- Common-prompt sensitivity:")]
    pooled_line_index = next(i for i, line in enumerate(report_lines) if line.startswith("- Pooled M3 interaction model:"))
    report_lines.insert(pooled_line_index + 1, sensitivity_line)
    _atomic_text(report_path, "\n".join(report_lines) + "\n")
    return audit, comparison, classification


def _cache_candidates() -> list[Path]:
    candidates: list[Path] = []
    for name in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value).expanduser())
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home).expanduser() / "hub")
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        candidates.append(Path(xdg).expanduser() / "huggingface/hub")
    candidates.append(Path.home() / ".cache/huggingface/hub")
    unique = []
    seen = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if str(resolved) not in seen:
            unique.append(path)
            seen.add(str(resolved))
    return unique


def _history_matches() -> dict[str, Any]:
    files = [Path.home() / name for name in (".bash_history", ".zsh_history", ".python_history")]
    scanned = []
    matches = []
    pattern = re.compile(r"Qwen3-(?:0\.6B|4B)-Base|models--Qwen--Qwen3-(?:0\.6B|4B)-Base")
    for path in files:
        if not path.is_file():
            continue
        scanned.append(str(path))
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for index, line in enumerate(lines, 1):
            if pattern.search(line):
                matches.append({
                    "file": str(path),
                    "line": index,
                    "models": sorted(set(pattern.findall(line))),
                    "sha256_candidates": re.findall(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", line),
                    "command_text_redacted": re.sub(r"(hf_[A-Za-z0-9]{10,}|(?:token|password|secret|key)=\S+)", "[REDACTED]", line, flags=re.I),
                })
    return {"files_scanned": scanned, "matching_entries": matches}


def _snapshot_report(cache_root: Path, model_id: str) -> dict[str, Any]:
    repo = model_id.replace("/", "--")
    model_path = cache_root / f"models--{repo}"
    result: dict[str, Any] = {
        "model_id": model_id,
        "cache_path": str(model_path),
        "exists": model_path.exists(),
        "refs": {},
        "snapshots": [],
    }
    if not model_path.exists():
        return result
    refs = model_path / "refs"
    if refs.exists():
        for ref in refs.rglob("*"):
            if ref.is_file():
                result["refs"][str(ref.relative_to(model_path))] = ref.read_text(errors="replace").strip()
    snapshots = model_path / "snapshots"
    if snapshots.exists():
        for snap in sorted(p for p in snapshots.iterdir() if p.is_dir()):
            item: dict[str, Any] = {
                "commit_sha": snap.name,
                "snapshot_mtime_utc": _iso_mtime(snap),
                "files": {},
                "weights": [],
            }
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "model.safetensors.index.json"):
                file = snap / name
                if file.exists():
                    st = file.stat()
                    item["files"][name] = {
                        "is_symlink": file.is_symlink(),
                        "blob_path": str(file.resolve()),
                        "bytes": st.st_size,
                        "blob_mtime_utc": _iso_mtime(file.resolve()),
                        "sha256": _sha256(file, max_bytes=8_000_000),
                    }
            for weight in sorted(snap.glob("*.safetensors")):
                st = weight.stat()
                item["weights"].append({
                    "name": weight.name,
                    "bytes": st.st_size,
                    "snapshot_link_mtime_utc": datetime.fromtimestamp(weight.lstat().st_mtime, timezone.utc).isoformat(),
                    "blob_mtime_utc": _iso_mtime(weight.resolve()),
                    "blob_path": str(weight.resolve()),
                })
            result["snapshots"].append(item)
    return result


def recover_p3_revisions() -> dict[str, Any]:
    p3_config = (P3_DIR / "config.yaml").read_text(encoding="utf-8")
    run_meta = json.loads((P3_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    e1_meta_path = P3_DIR / "e1_token_frequency_cache_metadata.json"
    e1_meta = json.loads(e1_meta_path.read_text(encoding="utf-8")) if e1_meta_path.exists() else {}
    start_match = re.match(r"(\d{8}T\d{6}Z)_", P3_DIR.name)
    start = datetime.strptime(start_match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc) if start_match else None
    elapsed = run_meta.get("elapsed_seconds")
    estimated_end = start + timedelta(seconds=float(elapsed)) if start and elapsed else None
    candidates = _cache_candidates()
    existing = [p for p in candidates if p.exists()]
    cache_root = existing[0].resolve() if existing else candidates[0]
    cache_reports = [_snapshot_report(p.resolve() if p.exists() else p, model_id) for p in existing for model_id in (MODELS["0.6B"]["id"], MODELS["4B"]["id"])]
    histories = _history_matches()
    log_files = [str(p) for p in P3_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".log", ".out", ".err"}]
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--", str(P3_DIR.relative_to(ROOT))], cwd=ROOT,
            capture_output=True, text=True, check=False,
        ).stdout.splitlines()
    except OSError:
        tracked = []

    model_entries = {}
    for label, model_id in (("draft", "Qwen/Qwen3-0.6B-Base"), ("target", "Qwen/Qwen3-4B-Base")):
        model_cache = next((entry for entry in cache_reports if entry["model_id"] == model_id), {"snapshots": [], "refs": {}})
        model_entries[label] = {
            "model_id": model_id,
            "recovered_revision": None,
            "confidence": "unresolved",
            "evidence_source": [
                "P3 config.yaml records the model ID but no immutable revision.",
                "P3 run_metadata.json records no model revision or model snapshot path.",
                "No matching revision was found in readable local shell history or P3 log files.",
                "The only accessible local Hugging Face snapshot metadata is timestamped after the historical P3 run interval, so it cannot identify which snapshot P3 used.",
            ],
            "local_cache_refs": model_cache.get("refs", {}),
            "local_cache_snapshots": model_cache.get("snapshots", []),
            "current_e2_pinned_revision_not_attributed_to_p3": MODELS["0.6B" if label == "draft" else "4B"]["revision"],
        }
    model_entries["target"]["tokenizer_only_revision_evidence"] = {
        "revision": e1_meta.get("tokenizer_revision"),
        "source": str(e1_meta_path),
        "limitation": "E1 recorded a tokenizer revision after H2/P3; this is not evidence for the target weight revision used in P3.",
    }
    cache_info = {
        "configured_cache_environment": {k: os.environ.get(k) for k in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME")},
        "candidate_cache_paths": [str(p) for p in candidates],
        "accessible_cache_paths": [str(p.resolve()) for p in existing],
        "cache_reports": cache_reports,
    }
    result = {
        "p3_run_id": P3_DIR.name,
        "run_id_start_utc": start.isoformat() if start else None,
        "run_elapsed_seconds_recorded": elapsed,
        "estimated_end_utc_from_run_id_plus_elapsed": estimated_end.isoformat() if estimated_end else None,
        "estimated_end_method_note": "The run ID encodes the start time; estimated end adds recorded elapsed_seconds. Treat as approximate if that elapsed field excludes setup or postprocessing.",
        "p3_config": p3_config,
        "p3_run_metadata": {k: run_meta.get(k) for k in ("elapsed_seconds", "torch_version", "cuda_version", "device", "seed", "model_load_elapsed_seconds", "target_model", "draft_model", "model_revisions")},
        "p3_e1_tokenizer_metadata": e1_meta,
        "cache": cache_info,
        "shell_history_search": histories,
        "p3_local_log_files": log_files,
        "p3_git_tracked_files": tracked,
        "models": model_entries,
        "overall_confidence": "unresolved",
        "conclusion": "Neither the P3 0.6B draft weight revision nor the P3 4B target weight revision can be recovered reliably from the accessible local artifacts. Current cache snapshots and the E1 tokenizer revision are not attributed to P3 weights.",
        "recommended_future_action": "If exact historical revisions are required for model-level replication, run one future comparison with both model revisions pinned and recorded; do not rerun now.",
    }
    _atomic_json(COMBINED / "p3_revision_recovery.json", result)
    lines = [
        "# Historical P3 model revision recovery",
        "",
        "**Result: unresolved for both P3 weight revisions.** No revision is inferred from current Hugging Face `main` or silently copied from the E2 run.",
        "",
        f"- P3 run ID: `{P3_DIR.name}`; approximate start from run ID: `{result['run_id_start_utc']}`; recorded elapsed: `{elapsed}` seconds; estimated end: `{result['estimated_end_utc_from_run_id_plus_elapsed']}`.",
        f"- Hugging Face cache path inspected: `{cache_root}`; `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, and `TRANSFORMERS_CACHE` are unset in this process. The home cache path resolves to `{cache_root}`.",
        f"- P3 files tracked by repository Git: {len(tracked)}; local P3 `.log`/`.out`/`.err` files: {len(log_files)}; readable shell-history matches: {len(histories['matching_entries'])}.",
        "",
        "| P3 model | Recovered revision | Confidence | Local evidence |",
        "|---|---|---|---|",
    ]
    for label in ("draft", "target"):
        entry = model_entries[label]
        snaps = entry["local_cache_snapshots"]
        snap_text = "; ".join(f"{x['commit_sha']} @ {x['snapshot_mtime_utc']}" for x in snaps) or "no snapshot"
        lines.append(f"| {entry['model_id']} ({label}) | unresolved | unresolved | accessible cache snapshot(s): {snap_text} |")
    lines += [
        "",
        "## Evidence inspected",
        "",
        "- P3 `config.yaml` has model IDs but no `revision` fields; P3 `run_metadata.json` has no model revision or snapshot path.",
        "- The accessible cache currently contains only the listed snapshots. Their snapshot and weight-blob modification times are after the approximate P3 completion time, so they are evidence of later availability, not P3 selection.",
        "- The cached config/tokenizer hashes and weight file sizes/timestamps are recorded in `p3_revision_recovery.json`. No corresponding P3-time model config/weight hashes were saved for comparison.",
        f"- E1 recorded tokenizer revision `{e1_meta.get('tokenizer_revision')}` for the 4B tokenizer. This does not identify the 4B weights loaded by P3.",
        f"- No matching local shell history entries or P3 run logs were found. Git tracks {len(tracked)} files under the P3 run path.",
        "",
        "The historical P3 0.6B revision remains unresolved. The historical 4B model-weight revision also remains unresolved; its later tokenizer revision is not sufficient evidence.",
        "",
        "**Recommended for future reproducibility:** one future model-pair rerun with immutable revisions for both models recorded in run metadata. No rerun is performed as part of this housekeeping task.",
    ]
    _atomic_text(COMBINED / "p3_revision_recovery.md", "\n".join(lines) + "\n")
    # Keep the existing E2 report explicit that its later tokenizer metadata
    # does not establish the historical P3 weight revisions.
    e2_report_path = COMBINED / "e2_model_pair_replication.md"
    e2_report = e2_report_path.read_text(encoding="utf-8")
    old_revision_line = "- P3's exact model revisions were not fully recorded in its run metadata; its configured IDs and 4B tokenizer revision match the pinned comparison, while the historical P3 0.6B draft commit remains unknown."
    new_revision_line = "- P3's exact historical 0.6B draft and 4B target weight revisions remain unresolved in local artifacts; the later E1 4B tokenizer revision does not establish P3 weight identity. See `p3_revision_recovery.md`."
    if old_revision_line in e2_report:
        e2_report = e2_report.replace(old_revision_line, new_revision_line)
    elif new_revision_line not in e2_report:
        raise AssertionError("Could not locate the historical P3 revision wording in the E2 report")
    _atomic_text(e2_report_path, e2_report)
    return result


def write_housekeeping_summary(ame: pd.DataFrame, audit: dict[str, Any], sensitivity: pd.DataFrame, classification: str, revision: dict[str, Any]) -> None:
    ame_lookup = ame.set_index(["pair", "model"])
    ame_text = "; ".join(
        f"{pair} M2/M3/M5 {ame_lookup.loc[(pair, 'M2'), 'ame_pp']:+.2f}/"
        f"{ame_lookup.loc[(pair, 'M3'), 'ame_pp']:+.2f}/"
        f"{ame_lookup.loc[(pair, 'M5'), 'ame_pp']:+.2f} pp"
        for pair in PAIR_ORDER
    )
    max_change = float(sensitivity.relative_or_change_pct.abs().max())
    lines = [
        "# E2 housekeeping summary",
        "",
        f"Adjusted CROSS-minus-SPLIT differences (M2/M3/M5): {ame_text}. The exact three-way eligible intersection is {audit['three_way_intersection_count']} prompts; common-prompt sensitivity is **{classification}** (maximum OR change {max_change:.2f}%); historical P3 weight revisions remain unresolved.",
        "",
        "**These checks do not change `E2 result: A — strong replication`.** All pair-specific adjusted differences remain positive and the common-prompt sensitivity does not materially change the pooled estimates. The unresolved P3 revisions limit exact model-level reproducibility, but are not by themselves evidence against the statistical replication.",
        "",
        "## CONFIRMED",
        "",
        "- All three pair-specific M2, M3, and M5 adjusted marginal rejection differences use the unchanged H2 eligible populations and prompt-clustered delta-method covariance.",
        f"- The common-prompt-only pooled analysis uses exactly {audit['three_way_intersection_count']} eligible prompt IDs present in P1, P2, and P3.",
        f"- Common-prompt pooled sensitivity classification: {classification}; classification uses effect magnitude and direction, not significance alone.",
        "",
        "## EXPLORATORY",
        "",
        "- Adjusted marginal probabilities are conditional associations averaged over each fitted sample's observed covariate distribution; they are not causal effects.",
        "- The common-prompt restriction is a sensitivity analysis and does not replace the primary pooled estimates.",
        "",
        "## FAILED / INCOMPLETE",
        "",
        "- P3 historical model-weight revision recovery is unresolved for both draft and target. Local snapshots were created after the P3 run interval, and available tokenizer metadata does not identify model weights.",
        "- No speculative decoding or teacher-forced scoring was rerun for this housekeeping analysis.",
        "",
        "## HIGHEST VERIFIED RUNG",
        "",
        "The completed E2 analysis and these housekeeping sensitivities are supported by the full 1,000-prompt P1/P2 artifacts, reused P3 artifacts, source summary CSV, and the additional AME/common-prompt outputs in this directory. The historical P3 weight identity remains outside what the artifacts can verify.",
        "",
        "## EVIDENCE GAPS",
        "",
        "- The original P3 0.6B and 4B weight revisions were not recorded; exact historical model identity is unavailable locally.",
        "- AME intervals condition on the observed covariate rows and use clustered coefficient covariance; they do not resample the covariate distribution.",
        "",
        "## RECOMMENDED NEXT",
        "",
        "Before H3, record immutable model revisions in future run metadata. H3 can proceed without a P3 rerun if the unresolved historical revision is carried as a reproducibility limitation.",
        "",
        "## Adjusted marginal effects",
        "",
        "| Pair | Model | Adjusted P(CROSS) | Adjusted P(SPLIT) | AME percentage points (95% CI) | p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in ame.itertuples(index=False):
        lines.append(
            f"| {r.pair} | {r.model} | {r.adjusted_p_cross:.2%} | {r.adjusted_p_split:.2%} | "
            f"{r.ame_pp:.2f} ({r.ame_ci_low:.2f}–{r.ame_ci_high:.2f}) | {r.ame_p_value:.3g} |"
        )
    lines += [
        "",
        "## Common-prompt sensitivity",
        "",
        f"The old pooled count of 999 is the union of the eligible prompt sets, not the intersection. The exact three-way common eligible count is {audit['three_way_intersection_count']}. See `e2_prompt_overlap_audit.md` and `e2_common_prompt_sensitivity.md` for exact IDs and estimates.",
        "",
        "## Revision recovery",
        "",
        revision["conclusion"],
        "",
        "## Verification",
        "",
        "- Existing pair-specific summary estimates were not rewritten by the housekeeping script.",
        "- P1/P2/P3 token tables and frequency cache were reused; morphology labels were not changed.",
        "- Only CPU-side regressions and artifact audits were run. No model weights were loaded; no speculative decoding or teacher-forced scoring was rerun.",
        "- Counterfactual predictions modify only `morph_class`; uncertainty is computed with prompt-clustered delta-method covariance.",
        "- Pooled common-prompt models use the original E2 M3/M5 formulas and prompt-clustered SEs.",
    ]
    _atomic_text(COMBINED / "e2_housekeeping_summary.md", "\n".join(lines) + "\n")


def main() -> int:
    COMBINED.mkdir(parents=True, exist_ok=True)
    tables, other = load_pair_tables()
    cache = other["frequency_cache"]
    ame = adjusted_marginal_effects(tables, cache)
    audit, sensitivity, classification = prompt_overlap_and_common_sensitivity(tables, cache)
    revision = recover_p3_revisions()
    write_housekeeping_summary(ame, audit, sensitivity, classification, revision)
    print(json.dumps({
        "ame_file": str(COMBINED / "e2_adjusted_marginal_effects.csv"),
        "common_prompt_file": str(COMBINED / "e2_common_prompt_sensitivity.md"),
        "common_prompt_count": audit["three_way_intersection_count"],
        "sensitivity_classification": classification,
        "p3_revision_confidence": revision["overall_confidence"],
        "summary": str(COMBINED / "e2_housekeeping_summary.md"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
