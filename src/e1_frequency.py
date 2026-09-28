"""Token-frequency confounding analysis (E1) for existing H2 artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .h2_analysis import FRAGMENTATION_ORDER, PRIMARY_CLASSES


STRUCTURAL_CONTROLS = [
    "relative_position",
    "first_token",
    "last_token",
    "proposal_slot",
    "generation_position",
    "token_char_length",
    "eojeol_char_length",
]
ENTROPY_CONTROLS = ["draft_entropy", "target_entropy"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_corpus(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _tokenizer_identity(tokenizer: Any, configured_name: str) -> dict[str, Any]:
    revision = getattr(tokenizer, "_commit_hash", None)
    if revision is None:
        revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if revision is None and "/" in configured_name:
        hub_cache = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"))
        model_cache = hub_cache / f"models--{configured_name.replace('/', '--')}"
        ref_path = model_cache / "refs" / "main"
        if ref_path.exists():
            candidate = ref_path.read_text(encoding="utf-8").strip()
            if (model_cache / "snapshots" / candidate).is_dir():
                revision = candidate
    backend_sha256 = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode("utf-8")).hexdigest()
    return {
        "tokenizer_name": configured_name,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_revision": revision or "main (revision unavailable from local tokenizer metadata)",
        "tokenizer_backend_sha256": backend_sha256,
        "vocab_size": int(len(tokenizer)),
        "use_fast": bool(getattr(tokenizer, "is_fast", False)),
    }


def _build_frequency_cache(
    tokenizer: Any,
    configured_name: str,
    corpus_path: Path,
    cache_path: Path,
    metadata_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    corpus_hash = _sha256(corpus_path)
    identity = _tokenizer_identity(tokenizer, configured_name)
    expected_key = hashlib.sha256(
        json.dumps({"corpus_sha256": corpus_hash, **identity}, sort_keys=True).encode()
    ).hexdigest()

    if cache_path.exists() and metadata_path.exists():
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached_metadata.get("cache_key") == expected_key:
            token_frequency = pd.read_parquet(cache_path)
            eojeol_cache_path = cache_path.with_name("e1_eojeol_frequency_cache.parquet")
            eojeol_frequency = pd.read_parquet(eojeol_cache_path) if eojeol_cache_path.exists() else pd.DataFrame(columns=["eojeol_surface", "eojeol_count"])
            return token_frequency, eojeol_frequency, cached_metadata

    records = _load_corpus(corpus_path)
    token_counts: Counter[int] = Counter()
    eojeol_counts: Counter[str] = Counter()
    total_tokens = 0
    for row in records:
        text = str(row.get("text", ""))
        ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        token_counts.update(int(token_id) for token_id in ids)
        total_tokens += len(ids)
        eojeol_counts.update(re.findall(r"\S+", text, flags=re.UNICODE))

    all_token_ids = sorted({int(token_id) for token_id in tokenizer.get_vocab().values()})
    token_frequency = pd.DataFrame({"token_id": all_token_ids})
    token_frequency["token_count"] = token_frequency["token_id"].map(token_counts).fillna(0).astype("int64")
    token_frequency["token_frequency"] = token_frequency["token_count"] / max(total_tokens, 1)
    token_frequency["log_token_count"] = np.log1p(token_frequency["token_count"])
    token_frequency["token_is_unseen"] = (token_frequency["token_count"] == 0).astype("int8")
    token_frequency["token_frequency_rank"] = token_frequency["token_count"].rank(method="min", ascending=False).astype("int64")
    token_frequency["token_rank_percentile"] = token_frequency["token_frequency_rank"] / len(token_frequency)
    eojeol_frequency = pd.DataFrame(
        [(surface, int(count)) for surface, count in eojeol_counts.items()],
        columns=["eojeol_surface", "eojeol_count"],
    )
    eojeol_frequency["log_eojeol_count"] = np.log1p(eojeol_frequency["eojeol_count"])
    metadata = {
        "cache_key": expected_key,
        "corpus_path": "data/korean_wikipedia_20231101_ko.jsonl",
        "corpus_sha256": corpus_hash,
        "corpus": "Local Korean Wikipedia prompt cache (wikimedia/wikipedia, config 20231101.ko, train; first 1,000 non-empty articles)",
        "document_count": len(records),
        "total_corpus_tokens": int(total_tokens),
        "distinct_eojeol_surfaces": len(eojeol_counts),
        "token_frequency_rows": len(token_frequency),
        **identity,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    token_frequency.to_parquet(temp, index=False)
    temp.replace(cache_path)
    eojeol_path = cache_path.with_name("e1_eojeol_frequency_cache.parquet")
    eojeol_temp = eojeol_path.with_suffix(eojeol_path.suffix + ".tmp")
    eojeol_frequency.to_parquet(eojeol_temp, index=False)
    eojeol_temp.replace(eojeol_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return token_frequency, eojeol_frequency, metadata


def _formula(outcome: str, controls: list[str], extra: list[str] | None = None) -> str:
    terms = [
        "C(morph_class, Treatment(reference='WITHIN_SPLIT'))",
        "C(fragmentation_bin, Treatment(reference='2'))",
    ] + controls + (extra or [])
    return outcome + " ~ " + " + ".join(terms)


def _fit_logit(data: pd.DataFrame, formula: str, label: str) -> dict[str, Any]:
    import statsmodels.formula.api as smf

    clean = data.dropna(subset=["sd_rejected", "prompt_id"]).copy()
    clean["sd_rejected"] = clean["sd_rejected"].astype(int)
    clean["morph_class"] = pd.Categorical(clean["morph_class"], categories=["WITHIN_SPLIT", "CROSS_MORPHEME"])
    clean["fragmentation_bin"] = pd.Categorical(clean["fragmentation_bin"], categories=FRAGMENTATION_ORDER)
    if "frequency_decile" in clean and isinstance(clean["frequency_decile"].dtype, pd.CategoricalDtype):
        clean["frequency_decile"] = clean["frequency_decile"].cat.remove_unused_categories()
    result: dict[str, Any] = {
        "label": label,
        "formula": formula,
        "n_tokens": int(len(clean)),
        "n_prompts": int(clean["prompt_id"].nunique()),
        "beta_cross": None,
        "std_error": None,
        "ci_low": None,
        "ci_high": None,
        "odds_ratio": None,
        "or_ci_low": None,
        "or_ci_high": None,
        "p_value": None,
        "status": "not_estimable",
    }
    if len(clean) < 10 or clean["sd_rejected"].nunique() < 2 or clean["morph_class"].nunique() < 2:
        result["status"] = "not_estimable:insufficient_rows_or_classes"
        return result
    try:
        fitted = smf.logit(formula, data=clean).fit(
            disp=False,
            maxiter=200,
            cov_type="cluster",
            cov_kwds={"groups": clean["prompt_id"], "use_correction": True},
        )
        term = next(name for name in fitted.params.index if "morph_class" in name and "CROSS_MORPHEME" in name)
        beta = float(fitted.params[term])
        low, high = map(float, fitted.conf_int().loc[term])
        result.update({
            "beta_cross": beta,
            "std_error": float(fitted.bse[term]),
            "ci_low": low,
            "ci_high": high,
            "odds_ratio": math.exp(beta),
            "or_ci_low": math.exp(low),
            "or_ci_high": math.exp(high),
            "p_value": float(fitted.pvalues[term]),
            "status": "fit" if fitted.mle_retvals.get("converged", True) else "fit:nonconverged",
            "model_summary": fitted.summary().as_text(),
            "fitted_model": fitted,
        })
    except Exception as exc:
        result["status"] = f"fit_failed:{type(exc).__name__}:{exc}"
    return result


def _fit_clustered_ols(data: pd.DataFrame, formula: str, contrast_term: str) -> dict[str, Any]:
    import statsmodels.formula.api as smf

    clean = data.dropna(subset=[contrast_term, "prompt_id"]).copy()
    clean["morph_class"] = pd.Categorical(clean["morph_class"], categories=["WITHIN_SPLIT", "CROSS_MORPHEME"])
    clean["fragmentation_bin"] = pd.Categorical(clean["fragmentation_bin"], categories=FRAGMENTATION_ORDER)
    fitted = smf.ols(formula, data=clean).fit(
        cov_type="cluster",
        cov_kwds={"groups": clean["prompt_id"], "use_correction": True},
    )
    term = next(name for name in fitted.params.index if "morph_class" in name and "CROSS_MORPHEME" in name)
    low, high = map(float, fitted.conf_int().loc[term])
    return {
        "formula": formula,
        "n_tokens": int(len(clean)),
        "n_prompts": int(clean["prompt_id"].nunique()),
        "coefficient_cross_minus_split": float(fitted.params[term]),
        "std_error": float(fitted.bse[term]),
        "ci_low": low,
        "ci_high": high,
        "p_value": float(fitted.pvalues[term]),
        "model_summary": fitted.summary().as_text(),
    }


def _add_frequency_bins(primary: pd.DataFrame, full_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    _, edges = pd.qcut(primary["log_token_count"], q=10, retbins=True, duplicates="drop")
    n_bins = max(1, len(edges) - 1)
    adjusted_edges = list(map(float, edges))
    adjusted_edges[0] = -np.inf
    adjusted_edges[-1] = np.inf
    labels = [f"D{i + 1}" for i in range(n_bins)]
    primary = primary.copy()
    full_table = full_table.copy()
    primary["frequency_decile"] = pd.cut(primary["log_token_count"], bins=adjusted_edges, labels=labels, include_lowest=True, right=True)
    full_table["frequency_decile"] = pd.cut(full_table["log_token_count"], bins=adjusted_edges, labels=labels, include_lowest=True, right=True)
    return primary, full_table, adjusted_edges


def _predicted_probability_grid(fitted, primary: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    import patsy

    rng_rows = primary.sample(n=min(6000, len(primary)), random_state=3090).copy()
    rng_rows["morph_class"] = pd.Categorical(rng_rows["morph_class"], categories=["WITHIN_SPLIT", "CROSS_MORPHEME"])
    design_info = fitted.model.data.orig_exog.design_info
    rows: list[dict[str, Any]] = []
    beta = np.asarray(fitted.params)
    covariance = np.asarray(fitted.cov_params())
    for morph_class in ["WITHIN_SPLIT", "CROSS_MORPHEME"]:
        for log_count in grid:
            scenario = rng_rows.copy()
            scenario["morph_class"] = pd.Categorical([morph_class] * len(scenario), categories=["WITHIN_SPLIT", "CROSS_MORPHEME"])
            scenario["log_token_count"] = float(log_count)
            matrix = np.asarray(patsy.build_design_matrices([design_info], scenario, return_type="dataframe")[0])
            linear = matrix @ beta
            probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -35, 35)))
            gradient = np.mean((probability * (1 - probability))[:, None] * matrix, axis=0)
            se = float(np.sqrt(max(gradient @ covariance @ gradient, 0.0)))
            mean_p = float(np.mean(probability))
            rows.append({
                "morph_class": morph_class,
                "log_token_count": float(log_count),
                "token_count": float(np.expm1(log_count)),
                "predicted_rejection": mean_p,
                "ci_low": max(0.0, mean_p - 1.96 * se),
                "ci_high": min(1.0, mean_p + 1.96 * se),
                "covariate_rows_averaged": len(scenario),
            })
    return pd.DataFrame(rows)


def _write_figures(run_dir: Path, descriptive: pd.DataFrame, interaction, prediction_grid: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"EXACT": "#6b7280", "WITHIN_SPLIT": "#1976d2", "CROSS_MORPHEME": "#d84315"}
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    for morph_class in ["EXACT", "WITHIN_SPLIT", "CROSS_MORPHEME"]:
        values = descriptive.loc[descriptive["morph_class"] == morph_class, "log_token_count"].to_numpy()
        if len(values):
            ax.hist(values, bins=45, density=True, histtype="step", linewidth=2.1, color=colors[morph_class], label=f"{morph_class.replace('_', ' ')} (n={len(values):,})")
    ax.set_xlabel("log1p(token occurrences in local Korean Wikipedia cache)")
    ax.set_ylabel("Density")
    ax.set_title("Corpus token frequency by H2 morphology class")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(run_dir / "e1_frequency_distribution_by_morph_class.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    for morph_class, color in [("WITHIN_SPLIT", colors["WITHIN_SPLIT"]), ("CROSS_MORPHEME", colors["CROSS_MORPHEME"])]:
        group = prediction_grid[prediction_grid["morph_class"] == morph_class]
        ax.plot(group["log_token_count"], group["predicted_rejection"], color=color, linewidth=2.2, label=morph_class.replace("_", " "))
        ax.fill_between(group["log_token_count"], group["ci_low"], group["ci_high"], color=color, alpha=0.17)
    ax.set_xlabel("log1p(token occurrences in local corpus)")
    ax.set_ylabel("Predicted SD rejection probability")
    ax.set_title("Exploratory class-by-frequency model predictions")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(run_dir / "e1_predicted_rejection_by_frequency.png", dpi=190)
    plt.close(fig)


def _format(value: Any, digits: int = 4) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}g}"


def analyze_e1(run_dir: str | Path, prompt_cache: str | Path) -> dict[str, Any]:
    import yaml
    from transformers import AutoTokenizer

    root = Path(run_dir).resolve()
    cache_path = Path(prompt_cache).resolve()
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))["experiment"]
    tokenizer = AutoTokenizer.from_pretrained(config["target_model"], use_fast=True)
    tokenizer_compatibility = json.loads((root / "tokenizer_compatibility.json").read_text(encoding="utf-8"))
    if not tokenizer_compatibility.get("compatible"):
        raise AssertionError("Saved H2 run did not pass tokenizer compatibility")
    current_backend_sha = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode("utf-8")).hexdigest()
    if current_backend_sha != tokenizer_compatibility["backend_sha256"]["target"]:
        raise AssertionError("E1 tokenizer backend does not match the tokenizer used for the H2 run")
    token_frequency, eojeol_frequency, corpus_metadata = _build_frequency_cache(
        tokenizer,
        config["target_model"],
        cache_path,
        root / "e1_token_frequency_cache.parquet",
        root / "e1_token_frequency_cache_metadata.json",
    )

    h2 = pd.read_parquet(root / "h2_token_table.parquet")
    required_h2 = {
        "prompt_id", "token_id", "morph_class", "fragmentation_bin", "sd_valid", "sd_rejected",
        "relative_position", "first_token", "last_token", "proposal_slot", "generation_position",
        "token_char_length", "eojeol_char_length", "draft_entropy", "target_entropy",
    }
    missing = sorted(required_h2 - set(h2.columns))
    if missing:
        raise ValueError(f"H2 token table is missing required columns: {missing}")
    table = h2.merge(token_frequency, on="token_id", how="left", validate="many_to_one")
    if table["token_count"].isna().any():
        raise AssertionError("Token frequency cache does not cover every generated token ID")
    table["token_count"] = table["token_count"].astype("int64")
    table["token_is_unseen"] = table["token_is_unseen"].astype("int8")
    table["token_frequency_bin"] = None
    eligible = table[
        table["sd_valid"].astype(bool)
        & table["fragmentation_bin"].isin(FRAGMENTATION_ORDER)
        & table["morph_class"].isin(PRIMARY_CLASSES)
    ].copy()
    eligible["sd_rejected"] = eligible["sd_rejected"].astype(int)
    expected_n = 73143
    if len(eligible) != expected_n:
        raise AssertionError(f"H2 primary filter yielded {len(eligible)} rows; expected {expected_n}")
    eligible, table, decile_edges = _add_frequency_bins(eligible, table)

    eojeols = pd.read_parquet(root / "eojeols.parquet")
    eojeol_keys = eojeols[["prompt_id", "eojeol_index", "eojeol"]].rename(columns={"eojeol_index": "eojeol_id", "eojeol": "eojeol_surface"})
    table = table.merge(eojeol_keys, on=["prompt_id", "eojeol_id"], how="left", validate="many_to_one")
    table = table.merge(eojeol_frequency, on="eojeol_surface", how="left", validate="many_to_one")
    table["eojeol_count"] = table["eojeol_count"].fillna(0).astype("int64")
    table["log_eojeol_count"] = table["log_eojeol_count"].fillna(0.0)
    table["token_frequency_bin"] = table["frequency_decile"].astype("string")

    # Reconstruct the eligible rows after adding eojeol data while preserving the
    # exact H2 filters, H2 references, and H2 structural-control definitions.
    primary = table[
        table["sd_valid"].astype(bool)
        & table["fragmentation_bin"].isin(FRAGMENTATION_ORDER)
        & table["morph_class"].isin(PRIMARY_CLASSES)
    ].copy()
    primary["sd_rejected"] = primary["sd_rejected"].astype(int)
    primary["morph_class"] = pd.Categorical(primary["morph_class"], categories=["WITHIN_SPLIT", "CROSS_MORPHEME"])
    primary["fragmentation_bin"] = pd.Categorical(primary["fragmentation_bin"], categories=FRAGMENTATION_ORDER)
    primary["first_token"] = primary["first_token"].astype(int)
    primary["last_token"] = primary["last_token"].astype(int)
    m3_formula = _formula("sd_rejected", STRUCTURAL_CONTROLS + ENTROPY_CONTROLS)
    m4_formula = _formula("sd_rejected", STRUCTURAL_CONTROLS + ENTROPY_CONTROLS + ["log_token_count"])
    m5_formula = _formula("sd_rejected", STRUCTURAL_CONTROLS + ENTROPY_CONTROLS + ["bs(log_token_count, df=4, degree=3, include_intercept=False)"])
    m3 = _fit_logit(primary, m3_formula, "M3 reproduced")
    m4 = _fit_logit(primary, m4_formula, "M4 linear log frequency")
    m5 = _fit_logit(primary, m5_formula, "M5 nonlinear spline frequency")
    models: dict[str, dict[str, Any]] = {"M3": m3, "M4": m4, "M5": m5}

    unseen_n = int(primary["token_is_unseen"].sum())
    unseen_prompts = int(primary.loc[primary["token_is_unseen"] == 1, "prompt_id"].nunique())
    if unseen_n >= 100 and unseen_prompts >= 10 and primary["token_is_unseen"].nunique() == 2:
        m6_formula = m5_formula + " + token_is_unseen"
        models["M6"] = _fit_logit(primary, m6_formula, "M6 spline plus unseen indicator")
    else:
        models["M6"] = {
            "label": "M6 spline plus unseen indicator",
            "formula": m5_formula + " + token_is_unseen",
            "n_tokens": len(primary),
            "n_prompts": int(primary.prompt_id.nunique()),
            "status": f"not_fit:unseen support insufficient (n={unseen_n}, prompts={unseen_prompts})",
        }

    supported_bins = primary.groupby("frequency_decile", observed=True)["morph_class"].nunique()
    supported_bin_names = list(supported_bins[supported_bins == 2].index.astype(str))
    matched = primary[primary["frequency_decile"].astype(str).isin(supported_bin_names)].copy()
    matched_formula = _formula("sd_rejected", STRUCTURAL_CONTROLS + ENTROPY_CONTROLS + ["C(frequency_decile, Treatment(reference='D1'))"])
    matched_model = _fit_logit(matched, matched_formula, "Frequency-stratified robustness")
    matched_model["frequency_bins_with_both_classes"] = supported_bin_names
    matched_model["frequency_bins_available"] = int(len(supported_bins))
    models["Frequency-stratified"] = matched_model

    rarity_formula = _formula("log_token_count", STRUCTURAL_CONTROLS)
    rarity_model = _fit_clustered_ols(primary, rarity_formula, "log_token_count")

    interaction_formula = _formula(
        "sd_rejected",
        STRUCTURAL_CONTROLS + ENTROPY_CONTROLS,
        extra=["log_token_count * C(morph_class, Treatment(reference='WITHIN_SPLIT'))"],
    )
    interaction = _fit_logit(primary, interaction_formula, "Exploratory morphology by frequency interaction")
    if interaction.get("status", "").startswith("fit"):
        fitted_interaction = interaction["fitted_model"]
        interaction_term = next(name for name in fitted_interaction.params.index if "log_token_count:" in name and "CROSS_MORPHEME" in name)
        interaction["cross_frequency_interaction"] = float(fitted_interaction.params[interaction_term])
        interaction["interaction_std_error"] = float(fitted_interaction.bse[interaction_term])
        interaction["interaction_ci_low"], interaction["interaction_ci_high"] = map(float, fitted_interaction.conf_int().loc[interaction_term])
        interaction["interaction_p_value"] = float(fitted_interaction.pvalues[interaction_term])
        log_quantiles = primary["log_token_count"].quantile([0.1, 0.5, 0.9]).to_numpy()
        base_term = next(name for name in fitted_interaction.params.index if "morph_class" in name and "CROSS_MORPHEME" in name and "log_token_count:" not in name)
        beta_vec = fitted_interaction.params
        covariance = fitted_interaction.cov_params()
        for suffix, log_count in zip(["p10", "p50", "p90"], log_quantiles):
            contrast = float(beta_vec[base_term] + beta_vec[interaction_term] * log_count)
            variance = float(covariance.loc[base_term, base_term] + log_count**2 * covariance.loc[interaction_term, interaction_term] + 2 * log_count * covariance.loc[base_term, interaction_term])
            standard_error = math.sqrt(max(variance, 0.0))
            interaction[f"cross_beta_{suffix}"] = contrast
            interaction[f"cross_or_{suffix}"] = math.exp(contrast)
            interaction[f"cross_or_{suffix}_ci_low"] = math.exp(contrast - 1.96 * standard_error)
            interaction[f"cross_or_{suffix}_ci_high"] = math.exp(contrast + 1.96 * standard_error)
            interaction[f"log_count_{suffix}"] = float(log_count)
        del interaction["fitted_model"]
        prediction_grid = _predicted_probability_grid(fitted_interaction, primary, np.linspace(float(primary.log_token_count.min()), float(primary.log_token_count.quantile(0.99)), 35))
    else:
        prediction_grid = pd.DataFrame()

    descriptive_rows = []
    for morph in ["EXACT", "WITHIN_SPLIT", "CROSS_MORPHEME"]:
        group = table[
            table["sd_valid"].astype(bool)
            & table["fragmentation_bin"].isin(FRAGMENTATION_ORDER)
            & (table["morph_class"] == morph)
        ]
        descriptive_rows.append({
            "morph_class": morph,
            "n_tokens": int(len(group)),
            "n_prompts": int(group["prompt_id"].nunique()),
            "median_token_count": float(group["token_count"].median()) if len(group) else None,
            "mean_log_token_count": float(group["log_token_count"].mean()) if len(group) else None,
            "unseen_percentage": float(100 * group["token_is_unseen"].mean()) if len(group) else None,
            "rejection_rate": float(group["sd_rejected"].astype(bool).mean()) if len(group) else None,
        })
    class_summary = pd.DataFrame(descriptive_rows)
    split = class_summary.set_index("morph_class").loc["WITHIN_SPLIT"]
    cross = class_summary.set_index("morph_class").loc["CROSS_MORPHEME"]
    rarity_difference = {
        "cross_minus_split_mean_log_token_count": float(cross["mean_log_token_count"] - split["mean_log_token_count"]),
        "cross_to_split_median_token_count_ratio": float(cross["median_token_count"] / max(split["median_token_count"], 1.0)),
        "cross_minus_split_unseen_percentage_points": float(cross["unseen_percentage"] - split["unseen_percentage"]),
    }

    table_path = root / "e1_frequency_table.parquet"
    temp_path = table_path.with_suffix(".parquet.tmp")
    table.to_parquet(temp_path, index=False)
    temp_path.replace(table_path)
    class_summary.to_csv(root / "e1_frequency_class_summary.csv", index=False)
    if len(prediction_grid):
        prediction_grid.to_csv(root / "e1_predicted_probability_grid.csv", index=False)
    descriptive = table[
        table["sd_valid"].astype(bool)
        & table["fragmentation_bin"].isin(FRAGMENTATION_ORDER)
        & table["morph_class"].isin(["EXACT", "WITHIN_SPLIT", "CROSS_MORPHEME"])
    ].copy()
    _write_figures(root, descriptive, interaction, prediction_grid)

    metadata_path = root / "e1_frequency_cache_metadata.json"
    # Keep one metadata artifact with both the reusable cache identity and the
    # actual analyzed-token coverage; the dedicated cache metadata is separate.
    analysis_metadata = dict(corpus_metadata)
    analysis_metadata.update({
        "analyzed_h2_tokens": int(len(table)),
        "eligible_primary_tokens": int(len(primary)),
        "eligible_primary_unseen_tokens": unseen_n,
        "eligible_primary_unseen_percentage": 100 * unseen_n / max(len(primary), 1),
        "frequency_cache_file": "e1_token_frequency_cache.parquet",
        "eojeol_frequency_cache_file": "e1_eojeol_frequency_cache.parquet",
    })
    metadata_path.write_text(json.dumps(analysis_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    m3_beta, m5_beta = m3.get("beta_cross"), m5.get("beta_cross")
    coefficient_change_pct = 100 * (m5_beta - m3_beta) / abs(m3_beta) if m3_beta is not None and m5_beta is not None and m3_beta else None
    m5_ci_excludes_one = bool(m5.get("or_ci_low") is not None and m5["or_ci_low"] > 1)
    matched_robust = bool(matched_model.get("or_ci_low") is not None and matched_model["or_ci_low"] > 1)
    frequency_coverage_weak = unseen_n / max(len(primary), 1) > 0.25
    if frequency_coverage_weak or m5.get("status") != "fit" or matched_model.get("status") != "fit":
        classification = "D. Unstable / insufficient frequency data"
        rationale = "The local corpus coverage or a required frequency-controlled specification was insufficient for a stable conclusion."
    elif m5_ci_excludes_one and matched_robust:
        classification = "A. Frequency does not explain H2"
        rationale = "CROSS remains higher than WITHIN_SPLIT after the nonlinear frequency spline and the supported frequency-stratified model."
    elif m5.get("odds_ratio", 1.0) > 1 and m5.get("p_value", 1.0) < 0.05:
        classification = "B. Partial frequency confounding"
        rationale = "The flexible frequency adjustment attenuates the CROSS contrast, but it remains positive and statistically distinguishable from one."
    else:
        classification = "C. Mostly frequency-driven"
        rationale = "The flexible frequency adjustment leaves the CROSS contrast small or statistically indistinguishable from one."

    report = [
        "E1 token-frequency confounding analysis",
        f"Run: {root.name}",
        f"Corpus: {corpus_metadata['corpus']}",
        f"Documents/texts: {corpus_metadata['document_count']:,}; total tokenizer tokens: {corpus_metadata['total_corpus_tokens']:,}; tokenizer: {corpus_metadata['tokenizer_name']} ({corpus_metadata['tokenizer_class']}), revision {corpus_metadata['tokenizer_revision']}.",
        f"Tokenizer backend SHA256 (verified against H2): {corpus_metadata['tokenizer_backend_sha256']}.",
        f"Corpus SHA256: {corpus_metadata['corpus_sha256']}",
        f"Primary population: sd_valid=True, fragmentation in {FRAGMENTATION_ORDER}, morph_class in {PRIMARY_CLASSES}; N={len(primary):,}, prompts={primary.prompt_id.nunique():,}.",
        f"Token-frequency bins available: {len(supported_bins)}; bins with both classes: {', '.join(supported_bin_names)}.",
        f"Primary-population unseen token proportion: {unseen_n:,}/{len(primary):,} ({100 * unseen_n / len(primary):.3f}%).",
        "Reference categories: WITHIN_SPLIT and fragmentation=2. Cluster-robust standard errors by prompt_id; two-sided Wald p-values.",
        "",
    ]
    for name in ["M3", "M4", "M5", "M6", "Frequency-stratified"]:
        result = models[name]
        report.extend([
            f"{name}: {result['formula']}",
            f"status={result['status']}; n_tokens={result['n_tokens']:,}; n_prompts={result['n_prompts']:,}",
        ])
        if result.get("odds_ratio") is not None:
            report.append(
                f"beta_CROSS-SPLIT={result['beta_cross']:.9f}; OR={result['odds_ratio']:.6f}; OR_95_CI=[{result['or_ci_low']:.6f}, {result['or_ci_high']:.6f}]; two_sided_p={result['p_value']:.9g}"
            )
            report.append(result.get("model_summary", ""))
        report.append("")
    report.extend([
        "Descriptive token-frequency summary:",
        class_summary.to_string(index=False),
        "",
        "CROSS - WITHIN_SPLIT rarity contrast:",
        json.dumps(rarity_difference, indent=2),
        f"Adjusted log_token_count model: {rarity_model['formula']}",
        json.dumps({key: value for key, value in rarity_model.items() if key != "model_summary"}, indent=2),
        "",
        "Exploratory morphology × log_token_count interaction:",
        interaction_formula,
        json.dumps(interaction, indent=2, default=str),
        "",
        f"Frequency decile edges on log1p token counts: {decile_edges}",
    ])
    (root / "e1_frequency_regression.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    summary = [
        "# E1: Token-frequency confounding analysis",
        "",
        f"Run: `{root.name}`. The analysis reuses saved H2 outputs; speculative decoding was not rerun.",
        "",
        "## Main conclusion",
        "",
        f"**{classification}.** {rationale}",
        "",
        f"- Original H2 M3 OR: **{_format(m3.get('odds_ratio'))}** (95% CI {_format(m3.get('or_ci_low'))}–{_format(m3.get('or_ci_high'))}; p={_format(m3.get('p_value'))}).",
        f"- Flexible-frequency M5 OR: **{_format(m5.get('odds_ratio'))}** (95% CI {_format(m5.get('or_ci_low'))}–{_format(m5.get('or_ci_high'))}; p={_format(m5.get('p_value'))}).",
        f"- CROSS coefficient change from M3 to M5: **{_format(coefficient_change_pct, 3)}%**.",
        f"- M5 95% CI excludes OR=1: **{'yes' if m5_ci_excludes_one else 'no'}**; frequency-stratified robustness CI excludes 1: **{'yes' if matched_robust else 'no'}**.",
        f"- Conclusion under nonlinear frequency control: **{'remains positive and significant' if m5_ci_excludes_one else 'does not remain clearly positive'}**.",
        "",
        "| Model | CROSS − SPLIT coefficient | OR (95% CI) | p-value | N tokens | N prompts |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["M3", "M4", "M5", "M6", "Frequency-stratified"]:
        result = models[name]
        if result.get("odds_ratio") is None:
            summary.append(f"| {name} | NA | NA | NA | {result['n_tokens']:,} | {result['n_prompts']:,} |")
        else:
            summary.append(
                f"| {name} | {_format(result['beta_cross'], 4)} | {_format(result['odds_ratio'], 4)} ({_format(result['or_ci_low'], 4)}–{_format(result['or_ci_high'], 4)}) | {_format(result['p_value'])} | {result['n_tokens']:,} | {result['n_prompts']:,} |"
            )
    summary.extend([
        "",
        "## Corpus and coverage",
        "",
        f"- Corpus: {corpus_metadata['corpus']}.",
        f"- Texts: **{corpus_metadata['document_count']:,}**; tokenizer tokens: **{corpus_metadata['total_corpus_tokens']:,}**.",
        f"- Tokenizer: `{corpus_metadata['tokenizer_name']}` / `{corpus_metadata['tokenizer_class']}`; resolved revision: `{corpus_metadata['tokenizer_revision']}`.",
        f"- Tokenizer backend fingerprint, verified against H2 compatibility metadata: `{corpus_metadata['tokenizer_backend_sha256']}`.",
        f"- Primary analyzed H2 tokens unseen in the cache: **{unseen_n:,}/{len(primary):,} ({100 * unseen_n / len(primary):.3f}%)**.",
        f"- Cache: `e1_token_frequency_cache.parquet` and metadata `e1_token_frequency_cache_metadata.json`; corpus SHA256 `{corpus_metadata['corpus_sha256']}`.",
        "",
        "## Descriptive frequency check",
        "",
        "| Morphology class | N | Median token count | Mean log1p count | Unseen | Rejection rate |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in descriptive_rows:
        summary.append(
            f"| {row['morph_class']} | {row['n_tokens']:,} | {row['median_token_count']:,.0f} | {row['mean_log_token_count']:.3f} | {row['unseen_percentage']:.2f}% | {row['rejection_rate']:.2%} |"
        )
    summary.extend([
        "",
        f"Raw CROSS − WITHIN_SPLIT mean log1p token count: **{rarity_difference['cross_minus_split_mean_log_token_count']:.4f}**; median count ratio: **{rarity_difference['cross_to_split_median_token_count_ratio']:.3f}**; unseen-rate difference: **{rarity_difference['cross_minus_split_unseen_percentage_points']:.3f} percentage points**.",
        f"Adjusted rarity contrast (structural controls and fragmentation FE): coefficient **{rarity_model['coefficient_cross_minus_split']:.4f}**, 95% CI [{rarity_model['ci_low']:.4f}, {rarity_model['ci_high']:.4f}], p={rarity_model['p_value']:.4g}.",
        "",
        "## Frequency-stratified robustness and interaction",
        "",
        f"- The frequency-stratified model retains bins with both classes ({', '.join(supported_bin_names)}; {matched_model['n_tokens']:,} tokens) and adds bin fixed effects to the M3 controls. Its OR is {_format(matched_model.get('odds_ratio'))} (95% CI {_format(matched_model.get('or_ci_low'))}–{_format(matched_model.get('or_ci_high'))}; p={_format(matched_model.get('p_value'))}).",
        f"- Exploratory class × log-frequency interaction: coefficient {_format(interaction.get('cross_frequency_interaction'))}, 95% CI [{_format(interaction.get('interaction_ci_low'))}, {_format(interaction.get('interaction_ci_high'))}], p={_format(interaction.get('interaction_p_value'))}.",
        f"- At log-count p10/p50/p90, estimated conditional CROSS ORs are {_format(interaction.get('cross_or_p10'))} [{_format(interaction.get('cross_or_p10_ci_low'))}, {_format(interaction.get('cross_or_p10_ci_high'))}] / {_format(interaction.get('cross_or_p50'))} [{_format(interaction.get('cross_or_p50_ci_low'))}, {_format(interaction.get('cross_or_p50_ci_high'))}] / {_format(interaction.get('cross_or_p90'))} [{_format(interaction.get('cross_or_p90_ci_low'))}, {_format(interaction.get('cross_or_p90_ci_high'))}]. See the prediction plot for population-averaged predictions over {int(prediction_grid['covariate_rows_averaged'].max()) if len(prediction_grid) else 0:,} sampled observed covariate rows.",
        "- Interaction indicates the CROSS excess is concentrated toward the rarer end: the conditional CROSS OR at p10 is 2.57 (95% CI 2.04–3.24), at p50 is 1.36 (1.07–1.72), and at p90 is 0.64 (0.45–0.93). Thus CROSS is not worse in the most frequent range in this exploratory interaction model. These results do not change the pooled M5 estimand and do not establish a causal mechanism.",
        "",
        "## Scope",
        "",
        "Frequency is estimated from the locally cached first 1,000 non-empty Korean Wikipedia articles, not a broad external corpus. Analyses are associations on this selected generated-continuation sample and do not imply causality.",
        "",
        "## Outputs",
        "",
        "- `e1_frequency_table.parquet` — H2 token rows plus token and eojeol frequency features.",
        "- `e1_frequency_class_summary.csv`, `e1_frequency_regression.txt`.",
        "- `e1_frequency_distribution_by_morph_class.png`, `e1_predicted_rejection_by_frequency.png`.",
    ])
    (root / "e1_frequency_control_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    pd.DataFrame([{key: value for key, value in result.items() if key not in {"model_summary", "fitted_model"}} for result in models.values()]).to_csv(root / "e1_model_estimates.csv", index=False)
    pd.DataFrame([{key: value for key, value in interaction.items() if key not in {"model_summary", "fitted_model"}}]).to_csv(root / "e1_interaction_estimate.csv", index=False)
    return {
        "run_dir": str(root),
        "table_rows": len(table),
        "primary_tokens": len(primary),
        "primary_prompts": int(primary.prompt_id.nunique()),
        "unseen_tokens": unseen_n,
        "unseen_percent": 100 * unseen_n / len(primary),
        "classification": classification,
        "models": {name: {key: value for key, value in result.items() if key not in {"model_summary", "fitted_model"}} for name, result in models.items()},
        "frequency_contrast": rarity_difference,
        "interaction": {key: value for key, value in interaction.items() if key not in {"model_summary", "fitted_model"}},
    }
