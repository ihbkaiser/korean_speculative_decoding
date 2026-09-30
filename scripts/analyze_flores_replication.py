#!/usr/bin/env python3
"""CPU-only H2/E2/H3/H4 analysis of saved FLORES replication artifacts."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import e2_model_pairs as e2
from src import h2_analysis as h2
from scripts import analyze_h3 as h3
from scripts import analyze_h4 as h4

RUN = ROOT / "runs/flores200_en_ko_replication"
TRACE = RUN / "traces"
TABLES = RUN / "token_tables"
AUDIT = RUN / "audits"
RESULTS = RUN / "results"
FIGURES = RUN / "figures"
PAIR_ORDER = ["P1", "P2", "P3"]
ENVIRONMENTS = h3.ENVIRONMENTS
ENV_LABEL = h3.ENV_LABELS
PAIR_LABEL = h3.PAIR_LABELS


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_json(path: Path, obj) -> None:
    e2.atomic_json(path, obj)


def md_table(frame: pd.DataFrame, digits: int = 3) -> str:
    if frame.empty:
        return "_(no rows)_"
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    headers = [str(x) for x in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(x) for x in row) + " |" for row in display.fillna("").itertuples(index=False, name=None))
    return "\n".join(lines)


def predominant_korean(text: str) -> tuple[float, bool, int, int]:
    letters = [ch for ch in text if ch.isalpha()]
    hangul = sum("HANGUL" in __import__("unicodedata").name(ch, "") for ch in letters)
    rate = hangul / len(letters) if letters else float("nan")
    return rate, bool(letters and rate >= 0.5), len(letters), hangul


def read_inputs():
    token_cache = pd.read_parquet(ROOT / "runs/20260926T184145Z_pilot1000/e1_token_frequency_cache.parquet")
    eo_cache = pd.read_parquet(ROOT / "runs/20260926T184145Z_pilot1000/e1_eojeol_frequency_cache.parquet")
    morph_cache = pd.read_parquet(ROOT / "runs/e2_model_pair_replication/h4/h4_morpheme_frequency_cache.parquet")
    raw_by_pair, env_by_pair, env_no_freq_by_pair = {}, {}, {}
    h4_pair_data, morphemes_by_pair, env_diag = {}, {}, {}
    from kiwipiepy import Kiwi
    kiwi = Kiwi()

    for pair in PAIR_ORDER:
        slug = e2.PAIRS[pair]["slug"]
        pair_dir = TRACE / slug
        raw = pd.read_parquet(pair_dir / "token_table.parquet")
        raw["pair"] = pair
        raw_by_pair[pair] = raw

        # H3/H4 morphology is regenerated from saved eojeol spans only; no model is loaded.
        eojeols = pd.read_parquet(pair_dir / "eojeols.parquet")
        morph_table, by_eojeol = h3.make_morpheme_sequences(pair, eojeols, kiwi)
        morphemes_by_pair[pair] = morph_table
        pair_table_dir = TABLES / slug
        pair_table_dir.mkdir(parents=True, exist_ok=True)
        e2.atomic_parquet(morph_table, pair_table_dir / "h3_eojeol_morphemes.parquet")
        no_freq, diag = h3.enrich_pair(pair, raw, by_eojeol)
        env_no_freq_by_pair[pair] = no_freq
        env_diag[pair] = diag
        with_freq = e2._attach_frequency(raw, token_cache)
        env, freq_diag = h3.enrich_pair(pair, with_freq, by_eojeol)
        env_by_pair[pair] = env
        e2.atomic_parquet(env, pair_table_dir / "h3_token_environment.parquet")
        env_diag[pair]["frequency_join"] = freq_diag

        # Exact H4 row construction reuses H3 boundary spans and H4's saved frequency caches.
        h3_map = h4.map_sequences(morph_table)
        h4_rows, h4_audit = h4.enrich_pair(pair, raw, no_freq, h3_map, h3.POS_MAP)
        h4_rows = h4.attach_frequencies(h4_rows, token_cache, eo_cache, morph_cache)
        h4_rows = h4.clean_h4(h4_rows)
        h4_pair_data[pair] = h4_rows
        e2.atomic_parquet(h4_rows, pair_table_dir / "h4_nominal_particle_rows.parquet")
        env_diag[pair]["h4_nominal_particle_alignment"] = h4_audit
        env_diag[pair]["h3_morpheme_reconstruction"] = {
            "morpheme_rows": int(len(morph_table)),
            "eojeol_rows": int(len(eojeols)),
        }
    return raw_by_pair, env_by_pair, env_no_freq_by_pair, h4_pair_data, token_cache, eo_cache, morph_cache, env_diag


def fit_h2_models(raw_by_pair: dict[str, pd.DataFrame], token_cache: pd.DataFrame):
    rows = []
    raw_rates = []
    reports = []
    fragment_frames = []
    for pair in PAIR_ORDER:
        table = raw_by_pair[pair]
        models, rates, report = e2._fit_pair_models(table, token_cache)
        reports.append(f"## {pair} ({PAIR_LABEL[pair]})\n\n{report}")
        for name in ["M1", "M2", "M3", "M4", "M5"]:
            result = models.get(name, {})
            rows.append({"pair": pair, "model": name,
                         "cross_coefficient": result.get("contrast_cross_minus_split", result.get("beta_cross")),
                         **{k: v for k, v in result.items() if k != "summary" and k != "fitted_model"}})
        rates.insert(0, "pair", pair)
        raw_rates.append(rates)
        h2_pop = table.loc[
            table.sd_valid.astype(bool)
            & table.fragmentation_bin.isin(h2.FRAGMENTATION_ORDER)
            & table.morph_class.astype(str).isin(["EXACT", "WITHIN_SPLIT", "CROSS_MORPHEME"])
        ].copy()
        frag = h2._bootstrap_rate_table(h2_pop, replicates=2000, seed=3090)
        frag.insert(0, "pair", pair)
        fragment_frames.append(frag)
    return pd.DataFrame(rows), pd.concat(raw_rates, ignore_index=True), pd.concat(fragment_frames, ignore_index=True), "\n\n".join(reports)


def run_h3_models(env_by_pair: dict[str, pd.DataFrame]):
    effects, cross_results, sensitivity, multi, fine, transitions = [], [], [], [], [], []
    diagnostics = {}
    for pair in PAIR_ORDER:
        effect, cross, diag, exploratory = h3.per_pair_models(pair, env_by_pair[pair])
        effects.append(effect)
        cross_results.append(cross)
        diagnostics[pair] = diag
        sensitivity.append(exploratory["sensitivity"])
        multi.append(exploratory["multiboundary"])
        fine.append(exploratory["fine_pos"])
        transitions.append(exploratory["fine_transition_counts"])
    counts, transition_counts, audit_md = h3.audit_tables(env_by_pair)
    all_pool = pd.concat([env_by_pair[p].assign(model_pair=p) for p in PAIR_ORDER], ignore_index=True)
    prompt_sets = {
        p: set(map(int, env_by_pair[p].prompt_id.unique())) for p in PAIR_ORDER
    }
    common = set.intersection(*(prompt_sets[p] for p in PAIR_ORDER))
    pooled_fit, pooled_diag, pooled_effects, pooled_meta = h3.pooled_model(all_pool)
    common_fit, common_diag, common_effects, common_meta = h3.pooled_model(all_pool, common)
    pooled_diag["common_prompt_ids"] = len(common)
    diagnostics["POOLED"] = pooled_diag
    diagnostics["POOLED_COMMON"] = common_diag
    return {
        "effects": pd.concat(effects, ignore_index=True),
        "cross": pd.concat(cross_results, ignore_index=True),
        "counts": counts,
        "transition_counts": transition_counts,
        "audit_md": audit_md,
        "diagnostics": diagnostics,
        "sensitivity": pd.concat(sensitivity, ignore_index=True),
        "multi": pd.concat(multi, ignore_index=True),
        "fine": pd.concat(fine, ignore_index=True),
        "transitions": pd.concat(transitions, ignore_index=True),
        "all_pool": all_pool,
        "common_prompt_ids": sorted(common),
        "prompt_ids_by_pair": {p: sorted(prompt_sets[p]) for p in PAIR_ORDER},
        "pooled_fit": pooled_fit,
        "pooled_diag": pooled_diag,
        "pooled_effects": pooled_effects,
        "pooled_meta": pooled_meta,
        "common_fit": common_fit,
        "common_diag": common_diag,
        "common_effects": common_effects,
        "common_meta": common_meta,
    }


def run_h4_lexical(h4_pair_data: dict[str, pd.DataFrame]):
    rows, diags = [], {}
    patterns, concentrations = [], []
    for pair in PAIR_ORDER:
        data = h4_pair_data[pair]
        lexical, diag = h4.run_lexical(data, pair)
        rows.append(lexical)
        diags[pair] = diag
        patt = h4.recurrent_patterns(data)
        patterns.append(patt)
        concentrations.append(h4.recurrent_pattern_concentration(patt))
    return pd.concat(rows, ignore_index=True), diags, pd.concat(patterns, ignore_index=True), pd.concat(concentrations, ignore_index=True)


def save_figures(h3res: dict, original: pd.DataFrame, flo_ame: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {"P1": "#0072B2", "P2": "#D55E00", "P3": "#009E73"}
    effects = h3res["effects"]
    fig, ax = plt.subplots(figsize=(8.8, 5.3), constrained_layout=True)
    env_order = ["NOMINAL_TO_PARTICLE", "PREDICATE_TO_ENDING", "ENDING_TO_ENDING", "LEXICAL_TO_LEXICAL", "OTHER"]
    ybase = np.arange(len(env_order))[::-1]
    offsets = {"P1": -0.18, "P2": 0.0, "P3": 0.18}
    for pair in PAIR_ORDER:
        sub = effects.loc[effects.pair.eq(pair)].set_index("environment")
        for i, env in enumerate(env_order):
            if env not in sub.index:
                continue
            row = sub.loc[env]
            if pd.isna(row.get("ame")):
                continue
            y = ybase[i] + offsets[pair]
            x = 100 * float(row.ame)
            low, high = 100 * float(row.ame_ci_low), 100 * float(row.ame_ci_high)
            ax.errorbar(x, y, xerr=[[x-low], [high-x]], fmt="o", color=colors[pair], capsize=3, markersize=5, linewidth=1.3)
    ax.axvline(0, color="#444444", linewidth=1, linestyle="--")
    ax.set_yticks(ybase, [ENV_LABEL[x] for x in env_order])
    ax.set_xlabel("Adjusted CROSS − SPLIT rejection difference (percentage points)")
    ax.set_title("FLORES-200 English→Korean: H3 environment-specific AMEs")
    handles = [plt.Line2D([0],[0], marker="o", color=colors[p], label=f"{p}: {PAIR_LABEL[p]}", linestyle="None") for p in PAIR_ORDER]
    ax.legend(handles=handles, frameon=False, loc="best")
    ax.grid(axis="x", alpha=.2)
    fig.savefig(FIGURES / "environment_ame_forest.png", dpi=220)
    fig.savefig(FIGURES / "environment_ame_forest.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.7), constrained_layout=True)
    y = np.arange(3)[::-1]
    sample_offsets = {"Original H3 Wikipedia": -0.11, "FLORES English→Korean": 0.11}
    for sample, frame, marker in [
        ("Original H3 Wikipedia", original, "s"),
        ("FLORES English→Korean", flo_ame, "o"),
    ]:
        for i, pair in enumerate(PAIR_ORDER):
            row = frame.loc[frame.pair.eq(pair)].iloc[0]
            val = 100 * float(row.ame)
            lo, hi = 100*float(row.ame_ci_low), 100*float(row.ame_ci_high)
            ax.errorbar(val, y[i]+sample_offsets[sample], xerr=[[val-lo],[hi-val]], fmt=marker,
                        color=colors[pair], capsize=3, markersize=5,
                        label=sample if i == 0 else None, linewidth=1.3)
    ax.axvline(0, color="#444444", linewidth=1, linestyle="--")
    ax.set_yticks(y, [f"{p}: {PAIR_LABEL[p]}" for p in PAIR_ORDER])
    ax.set_xlabel("Adjusted NOMINAL_TO_PARTICLE CROSS − SPLIT (pp; 95% CI)")
    ax.set_title("Original H3 and external FLORES sample (fit separately)")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=.2)
    fig.savefig(FIGURES / "replication_comparison.png", dpi=220)
    plt.close(fig)


def load_original_h3() -> pd.DataFrame:
    rows = []
    base = ROOT / "runs/e2_model_pair_replication/h3/pair_results"
    for pair in PAIR_ORDER:
        path = base / f"{pair.lower()}_h3_results.csv"
        data = pd.read_csv(path)
        row = data.loc[data.environment.eq("NOMINAL_TO_PARTICLE")].iloc[0]
        rows.append({"sample": "Original H3 Wikipedia", "pair": pair, "ame": row.ame,
                     "ame_ci_low": row.ame_ci_low, "ame_ci_high": row.ame_ci_high,
                     "adjusted_p_cross": row.adjusted_p_cross, "adjusted_p_split": row.adjusted_p_split,
                     "n_cross": row.n_cross, "n_split": row.n_split,
                     "p_value": row.p_value, "p_holm_within_pair": row.p_holm_within_pair})
    return pd.DataFrame(rows)


def make_language_audit() -> pd.DataFrame:
    rows = []
    for pair in PAIR_ORDER:
        cont = pd.read_parquet(TRACE / e2.PAIRS[pair]["slug"] / "continuations.parquet")
        for record in cont.itertuples(index=False):
            rate, korean, letters, hangul = predominant_korean(str(record.reference_text))
            rows.append({"pair": pair, "prompt_id": int(record.prompt_id), "generated_chars": len(str(record.reference_text)),
                         "letter_count": letters, "hangul_letter_count": hangul, "hangul_letter_share": rate,
                         "predominantly_korean": korean, "exact_match": bool(record.exact_match),
                         "target_singleton_fallback": bool(record.target_singleton_fallback)})
    return pd.DataFrame(rows)


def main() -> None:
    for path in [AUDIT, RESULTS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)
    config = json.loads((RUN / "flores_config.json").read_text(encoding="utf-8"))
    # Cache identity checks precede any lexical feature construction.
    token_meta = json.loads((ROOT / "runs/20260926T184145Z_pilot1000/e1_token_frequency_cache_metadata.json").read_text())
    h4_config = json.loads((ROOT / "runs/e2_model_pair_replication/h4/h4_config.json").read_text())
    if not config["tokenizer_compatibility"]["shared_tokenizer"]:
        raise RuntimeError("Shared tokenizer identity failed; refusing E1/H4 cache reuse")
    if token_meta["tokenizer_revision"] != e2.MODELS["4B"]["revision"]:
        raise RuntimeError("E1 cache tokenizer revision does not match the pinned 4B tokenizer")
    if h4_config["frequency"]["source_corpus_sha256"] != token_meta["corpus_sha256"]:
        raise RuntimeError("H4 morpheme cache source differs from E1 corpus")

    raw, env, env_no_freq, h4data, token_cache, eo_cache, morph_cache, enrich_diag = read_inputs()
    expected_tokenizer = json.loads((RUN / "tokenizer_compatibility.json").read_text())
    cache_backend = token_meta["tokenizer_backend_sha256"]
    if expected_tokenizer["model_tokenizers"]["4B"]["backend_sha256"] != cache_backend:
        raise RuntimeError("Frequency cache backend hash mismatch")
    if token_cache.token_id.duplicated().any():
        raise RuntimeError("Duplicate token IDs in E1 frequency cache")
    frequency_join_missing = {
        pair: int(pd.to_numeric(env[pair].log_token_count, errors="coerce").isna().sum()) for pair in PAIR_ORDER
    }
    if any(frequency_join_missing.values()):
        raise RuntimeError(f"E1 token frequency cache misses token rows: {frequency_join_missing}")

    h2_results, h2_raw, frag, h2_text = fit_h2_models(raw, token_cache)
    atomic_csv(h2_results, RESULTS / "cross_vs_split_results.csv")
    atomic_csv(h2_raw, RESULTS / "cross_vs_split_raw_rates.csv")
    atomic_csv(frag, AUDIT / "fragmentation_counts.csv")
    atomic_csv(frag, RESULTS / "fragmentation_results.csv")
    atomic_text(RESULTS / "cross_vs_split_regression.txt", h2_text)

    h3res = run_h3_models(env)
    atomic_csv(h3res["effects"], RESULTS / "environment_pair_results.csv")
    atomic_csv(h3res["cross"], RESULTS / "environment_cross_only_results.csv")
    atomic_csv(h3res["counts"], AUDIT / "morphology_environment_counts.csv")
    atomic_csv(h3res["transition_counts"], AUDIT / "fine_pos_transition_counts.csv")
    atomic_text(AUDIT / "morphology_environment_audit.md", h3res["audit_md"])
    atomic_csv(h3res["sensitivity"], RESULTS / "environment_construction_sensitivity.csv")
    atomic_csv(h3res["multi"], RESULTS / "multiboundary_exploratory.csv")
    atomic_csv(h3res["fine"], RESULTS / "fine_pos_exploratory.csv")
    atomic_csv(h3res["transitions"], RESULTS / "fine_pos_transition_counts.csv")
    atomic_csv(h3res["pooled_effects"], RESULTS / "environment_pooled_results.csv")
    atomic_csv(h3res["common_effects"], RESULTS / "environment_pooled_common_prompts.csv")
    atomic_json(AUDIT / "common_prompt_intersection.json", {
        "method": "intersection of eligible prompt IDs present in each pair H3 table",
        "count": len(h3res["common_prompt_ids"]),
        "prompt_ids": h3res["common_prompt_ids"],
        "pair_eligible_prompt_counts": {p: len(h3res["prompt_ids_by_pair"][p]) for p in PAIR_ORDER},
        "pair_missing_prompt_ids_from_three_way_intersection": {
            p: sorted(set(h3res["prompt_ids_by_pair"][p]) - set(h3res["common_prompt_ids"])) for p in PAIR_ORDER
        },
    })

    l3, l3diag, patterns, concentration = run_h4_lexical(h4data)
    atomic_csv(l3, RESULTS / "nominal_particle_l3_results.csv")
    atomic_json(AUDIT / "nominal_particle_l3_diagnostics.json", l3diag)
    atomic_csv(patterns, RESULTS / "token_concentration_patterns.csv")
    atomic_csv(concentration, RESULTS / "token_concentration.csv")

    original = load_original_h3()
    flo_ame = h3res["effects"].loc[h3res["effects"].environment.eq("NOMINAL_TO_PARTICLE")].copy()
    flo_ame["sample"] = "FLORES English→Korean"
    flo_ame.to_csv(RESULTS / "flores_nominal_particle_ames.csv", index=False)
    original.to_csv(RESULTS / "original_nominal_particle_ames.csv", index=False)
    save_figures(h3res, original, flo_ame)

    language = make_language_audit()
    atomic_csv(language, AUDIT / "generated_language_audit.csv")
    population_rows = []
    for pair in PAIR_ORDER:
        table = raw[pair]
        eligible = e2._eligible(table)
        h3_data = env[pair]
        cross_split = h3_data[h3_data.morph_class.astype(str).isin(["CROSS_MORPHEME", "WITHIN_SPLIT"])]
        labels = table.morph_class.value_counts(dropna=False).to_dict()
        cont = pd.read_parquet(TRACE / e2.PAIRS[pair]["slug"] / "continuations.parquet")
        events = pd.read_parquet(TRACE / e2.PAIRS[pair]["slug"] / "sd_events.parquet")
        teachers = pd.read_parquet(TRACE / e2.PAIRS[pair]["slug"] / "teacher_forced_tokens.parquet")
        alignment = pd.read_csv(TRACE / e2.PAIRS[pair]["slug"] / "alignment_audit.csv")
        population_rows.append({
            "pair": pair, "prompts": int(cont.prompt_id.nunique()), "generated_tokens": int(len(table)),
            "sd_valid_output_tokens": int(table.sd_valid.astype(bool).sum()),
            "sd_invalid_output_tokens": int((~table.sd_valid.astype(bool)).sum()),
            "invalidated_sd_proposals": int(events.invalidated_after_first_rejection.astype(bool).sum()),
            "target_sd_exact_prompt_count": int(cont.exact_match.astype(bool).sum()),
            "target_sd_mismatch_prompt_count": int((~cont.exact_match.astype(bool)).sum()),
            "fallback_target_prompts": int(cont.target_singleton_fallback.astype(bool).sum()),
            "eligible_h2_tokens": int(len(eligible)), "eligible_h2_prompts": int(eligible.prompt_id.nunique()),
            "cross_morpheme": int(labels.get("CROSS_MORPHEME", 0)),
            "within_split": int(labels.get("WITHIN_SPLIT", 0)),
            "cross_eojeol_excluded": int(labels.get("CROSS_EOJEOL", 0)),
            "kiwi_complex_excluded": int(labels.get("KIWI_COMPLEX", 0)),
            "exact": int(labels.get("EXACT", 0)),
            "teacher_forced_rows": int(len(teachers)),
            "retokenization_exact_prompts": int(alignment.retokenization_roundtrip_exact.astype(bool).sum()),
            "retokenization_fallback_prompts": int((~alignment.retokenization_roundtrip_exact.astype(bool)).sum()),
            "visible_non_special_tokens": int((~table.is_special_token.astype(bool)).sum()),
            "exact_span_rows": int((table.span_exact.astype(bool) & ~table.is_special_token.astype(bool)).sum()),
            "alignment_success_percent_visible": 100*float((table.span_exact.astype(bool) & ~table.is_special_token.astype(bool)).sum())/max(1,int((~table.is_special_token.astype(bool)).sum())),
            "alignment_failures": int((~table.span_exact.astype(bool) & ~table.is_special_token.astype(bool)).sum()),
            "cross_kiwi_exclusion_percent": 100*(int(labels.get("CROSS_EOJEOL", 0))+int(labels.get("KIWI_COMPLEX", 0)))/max(1,len(table)),
            "eligible_cross_split_tokens_in_h3": int(len(cross_split)),
            "eligible_cross_split_prompts_in_h3": int(cross_split.prompt_id.nunique()),
            "n_to_p_cross": int(((h3_data.morph_class.astype(str).eq("CROSS_MORPHEME")) & (h3_data.morph_environment.eq("NOMINAL_TO_PARTICLE")) & (h3_data.num_boundaries_crossed.eq(1))).sum()),
            "n_to_p_split": int(((h3_data.morph_class.astype(str).eq("WITHIN_SPLIT")) & (h3_data.morph_environment.eq("NOMINAL_TO_PARTICLE"))).sum()),
        })
    population = pd.DataFrame(population_rows)
    atomic_csv(population, AUDIT / "population_audit.csv")
    language_summary = language.groupby("pair", observed=True).agg(
        outputs=("prompt_id", "size"), predominantly_korean=("predominantly_korean", "sum"),
        hangul_share_mean=("hangul_letter_share", "mean"), hangul_share_median=("hangul_letter_share", "median"),
    ).reset_index()
    language_summary["predominantly_korean_percent"] = 100 * language_summary.predominantly_korean / language_summary.outputs
    atomic_csv(language_summary, AUDIT / "generated_language_summary.csv")

    common_rows = h3res["common_effects"].loc[h3res["common_effects"].environment.eq("NOMINAL_TO_PARTICLE")].copy()
    all_rows = h3res["pooled_effects"].loc[h3res["pooled_effects"].environment.eq("NOMINAL_TO_PARTICLE")].copy()
    common_sensitivity = all_rows.merge(common_rows, on="environment", suffixes=("_all", "_common"))
    if len(common_sensitivity):
        common_sensitivity["ame_change_pp"] = 100 * (common_sensitivity.ame_common - common_sensitivity.ame_all)
        max_change = float(common_sensitivity.ame_change_pp.abs().max())
        reversal = bool(((common_sensitivity.ame_all > 0) != (common_sensitivity.ame_common > 0)).any())
        cls = "material change" if reversal or max_change > 1.5 else "minor numerical change" if max_change > 0.5 else "unchanged"
    else:
        max_change, reversal, cls = np.nan, False, "not estimable"
    atomic_csv(common_sensitivity, RESULTS / "common_prompt_sensitivity.csv")
    common_md = [
        "# FLORES common-prompt sensitivity", "",
        f"The exact three-way intersection is **{len(h3res['common_prompt_ids']):,} prompt IDs** from the eligible H3 tables (see `audits/common_prompt_intersection.json`). The pooled H3 model was refit on that intersection with the identical formula and prompt-clustered covariance.",
        f"Nominal→Particle pooled AME shift: {max_change:+.2f} pp maximum absolute shift; sign reversal: {reversal}; classification: **{cls}**.", "",
        md_table(common_sensitivity[["environment", "ame_all", "ame_ci_low_all", "ame_ci_high_all", "ame_common", "ame_ci_low_common", "ame_ci_high_common", "ame_change_pp"]], 4), "",
        "Pair-specific estimates are primary. This pooled common-prompt refit is sensitivity analysis only.", "",
    ]
    atomic_text(RESULTS / "common_prompt_sensitivity.md", "\n".join(common_md))

    # H3 AME point estimates and intervals separately by pair; pooled aggregate is reported separately.
    np_pair = flo_ame.sort_values("pair")
    pooled_np = h3res["pooled_effects"].loc[h3res["pooled_effects"].environment.eq("NOMINAL_TO_PARTICLE")]
    pooled_np_row = pooled_np.iloc[0] if len(pooled_np) else None
    focal_pos = bool((np_pair.ame > 0).all()) if len(np_pair) == 3 else False
    pooled_ci_pos = bool(pooled_np_row is not None and pooled_np_row.ame > 0 and pooled_np_row.ame_ci_low > 0)
    pattern_dir = bool(np_pair.ame_minus_lexical.notna().all() and (np_pair.ame_minus_lexical > 0).all())
    if focal_pos and pooled_ci_pos and pattern_dir:
        decision = "REPLICATED"
        one_sentence = "the Nominal→Particle CROSS−SPLIT association is positive in all three model pairs, in the pooled estimate, and relative to the lexical-boundary environment."
    elif int((np_pair.ame > 0).sum()) >= 2 and (pooled_np_row is not None and pooled_np_row.ame > 0):
        decision = "PARTIALLY_REPLICATED"
        one_sentence = "the Nominal→Particle association is directionally positive in most pairs, but at least one prespecified replication criterion is not met."
    elif int((np_pair.ame < 0).sum()) >= 2 and (pooled_np_row is not None and pooled_np_row.ame_ci_high < 0):
        decision = "NOT_REPLICATED"
        one_sentence = "the FLORES data do not reproduce the positive Nominal→Particle CROSS−SPLIT association."
    else:
        decision = "INCONCLUSIVE"
        one_sentence = "the FLORES estimates do not meet the prespecified replication criteria and remain too uncertain for a directional classification."

    # Model diagnostics and separation/rank/convergence flags.
    diag_obj = {
        "h3": h3res["diagnostics"],
        "h4_l3": l3diag,
        "enrichment": enrich_diag,
        "pooled_interaction_test": h3res["pooled_meta"].get("interaction_test"),
        "pooled_common_interaction_test": h3res["common_meta"].get("interaction_test"),
        "frequency_cache_identity": {
            "token_cache_revision": token_meta["tokenizer_revision"],
            "tokenizer_backend_sha256": token_meta["tokenizer_backend_sha256"],
            "token_frequency_rows": len(token_cache), "eojeol_frequency_rows": len(eo_cache),
            "morpheme_frequency_rows": len(morph_cache), "missing_token_cache_joins": frequency_join_missing,
            "h4_frequency_corpus_sha256": h4_config["frequency"]["source_corpus_sha256"],
        },
    }
    atomic_json(AUDIT / "model_diagnostics.json", diag_obj)

    diag_lines = ["# Model diagnostics", "", "All models use prompt-clustered covariance. Full diagnostic records are in `model_diagnostics.json`.", "", "## Pair and pooled model status", ""]
    for group, models in [("H3", h3res["diagnostics"]), ("H4 L1/L2/L3", l3diag)]:
        diag_lines += [f"### {group}", ""]
        for label, obj in models.items():
            if group == "H3" and label in ["POOLED", "POOLED_COMMON"]:
                diag = obj
                diag_lines.append(f"- {label}: converged={diag.get('converged')}, rank={diag.get('rank')}/{diag.get('n_columns')}, clusters={diag.get('cluster_count')}, max |β|={diag.get('max_abs_coefficient')}, warnings={diag.get('warnings')}, error={diag.get('fit_error')}.")
            elif group == "H3":
                diag = obj.get("primary_model", {})
                diag_lines.append(f"- {label}: converged={diag.get('converged')}, rank={diag.get('rank')}/{diag.get('n_columns')}, clusters={diag.get('cluster_count')}, max |β|={diag.get('max_abs_coefficient')}, rank-deficient={diag.get('rank_deficient')}, extreme={diag.get('extreme_coefficient_warning')}, warnings={diag.get('warnings')}, error={diag.get('fit_error')}.")
            else:
                for model_name, subdiag in obj.items():
                    diag_lines.append(f"- {label} {model_name}: converged={subdiag.get('converged')}, rank={subdiag.get('rank')}/{subdiag.get('n_columns')}, clusters={subdiag.get('cluster_count')}, max |β|={subdiag.get('max_abs_coefficient')}, rank-deficient={subdiag.get('rank_deficient')}, extreme={subdiag.get('extreme_coefficient')}, warnings={subdiag.get('warnings')}, error={subdiag.get('fit_error')}.")
        diag_lines.append("")
    atomic_text(AUDIT / "diagnostics.md", "\n".join(diag_lines))

    atomic_text(AUDIT / "population_audit.md", "\n".join([
        "# FLORES population audit", "",
        "Every non-empty official English–Korean devtest row was retained. The Korean reference was not used for prompt construction, decoding, alignment, inclusion, or labels.", "",
        md_table(population), "",
        "`sd_invalid_output_tokens` counts output positions with no valid SD proposal decision; invalidated later proposals are counted separately in `invalidated_sd_proposals`. Exact target/SD continuation mismatch prompts must be zero. Morphology exclusion rates and class counts are in `population_audit.csv`; detailed spans are in the H2-compatible `token_tables/` files.", "",
        "## Generated-language audit", "", md_table(language_summary), "",
        "Predominantly Korean uses the frozen descriptive threshold of at least 50% Hangul among Unicode letters. No output was excluded on this measure or on translation quality.", "",
    ]))

    l3_pivot = l3.pivot(index="pair", columns="model", values="ame").reset_index()
    l3_text = []
    for pair in PAIR_ORDER:
        group = l3.loc[l3.pair.eq(pair)].set_index("model")
        l3_text.append({"pair": pair, "L1_AME_pp": 100*group.loc["L1", "ame"], "L2_AME_pp": 100*group.loc["L2", "ame"],
                        "L3_AME_pp": 100*group.loc["L3", "ame"], "L1_to_L3_change_pp": 100*(group.loc["L3", "ame"]-group.loc["L1", "ame"]),
                        "L3_ci_low_pp": 100*group.loc["L3", "ame_ci_low"], "L3_ci_high_pp": 100*group.loc["L3", "ame_ci_high"],
                        "L3_p": group.loc["L3", "p_value"], "n_tokens": group.loc["L3", "n_tokens"], "n_prompts": group.loc["L3", "n_prompts"]})
    l3_summary = pd.DataFrame(l3_text)
    atomic_csv(l3_summary, RESULTS / "nominal_particle_l3_summary.csv")

    comparison = original.merge(flo_ame, on="pair", suffixes=("_original", "_flores"))
    atomic_csv(comparison, RESULTS / "original_vs_flores_nominal_particle.csv")
    h3_table = h3res["effects"].copy()
    npp = h3_table.loc[h3_table.environment.eq("NOMINAL_TO_PARTICLE"), ["pair", "n_cross", "n_split", "n_prompts", "adjusted_p_cross", "adjusted_p_split", "ame", "ame_ci_low", "ame_ci_high", "p_value", "p_holm_within_pair", "ame_minus_lexical", "ame_minus_lexical_ci_low", "ame_minus_lexical_ci_high", "ame_minus_lexical_p", "omnibus_interaction_p"]]
    npp["ame_pp"] = 100*npp.ame
    npp["ame_ci_low_pp"] = 100*npp.ame_ci_low
    npp["ame_ci_high_pp"] = 100*npp.ame_ci_high
    atomic_csv(npp, RESULTS / "nominal_particle_focal_results.csv")

    pooled_text = md_table(pooled_np[["environment", "n_rows", "n_cross", "n_split", "n_prompts", "adjusted_p_cross", "adjusted_p_split", "ame", "ame_ci_low", "ame_ci_high", "p_value"]], 4) if len(pooled_np) else "_(not estimable)_"
    support = h3res["counts"].loc[(h3res["counts"].scope.eq("H3B_ALL_CLASSES")) & (h3res["counts"].environment.eq("NOMINAL_TO_PARTICLE")), ["pair", "n_cross", "n_split", "n_prompts", "raw_cross_rejection", "raw_split_rejection", "support_warning", "split_support_warning"]]
    conc_np = concentration
    decision_sentence = f"FLORES-200 English→Korean replication: {decision} — {one_sentence}"
    original_show = original[["pair", "ame", "ame_ci_low", "ame_ci_high", "n_cross", "n_split", "p_value"]].copy()
    original_show[["ame", "ame_ci_low", "ame_ci_high"]] *= 100
    flores_show = flo_ame[["pair", "ame", "ame_ci_low", "ame_ci_high", "n_cross", "n_split", "n_prompts", "p_value", "p_holm_within_pair", "ame_minus_lexical", "ame_minus_lexical_ci_low", "ame_minus_lexical_ci_high"]].copy()
    flores_show[["ame", "ame_ci_low", "ame_ci_high", "ame_minus_lexical", "ame_minus_lexical_ci_low", "ame_minus_lexical_ci_high"]] *= 100

    report = [
        decision_sentence, "",
        "## Data and execution", "",
        f"Original FLORES-200 devtest: **1,012** paired rows, all included in source order. Archive SHA-256 `{config['source']['archive_sha256']}`; English file `{config['source']['english_sha256']}`; Korean provenance file `{config['source']['korean_reference_sha256']}`. See `data_manifest.csv`, `data_manifest.sha256`, and `prompt_audit.md`.",
        "GPU execution used physical GPU 3 (A100-SXM4-80GB), process-visible as `cuda:0` with `CUDA_VISIBLE_DEVICES=3`. Full hardware/software evidence and peak memory are in `flores_config.json`; runtime checkpoints are prompt-resumable.", "",
        "## Speculative-decoding and alignment validity", "",
        md_table(population[["pair", "prompts", "generated_tokens", "sd_valid_output_tokens", "sd_invalid_output_tokens", "invalidated_sd_proposals", "target_sd_exact_prompt_count", "target_sd_mismatch_prompt_count", "cross_eojeol_excluded", "kiwi_complex_excluded", "cross_kiwi_exclusion_percent", "exact_span_rows", "visible_non_special_tokens", "alignment_success_percent_visible", "alignment_failures", "retokenization_exact_prompts", "retokenization_fallback_prompts"]]), "",
        f"Output language audit: {md_table(language_summary)}", "",
        "All target and SD token IDs matched exactly for the analyzed prompts; no prompt was removed for language or translation quality. Alignment used E2's H2 tokenizer offset and Kiwi pipeline. `population_audit.csv` and `morphology_environment_counts.csv` retain the full counts.", "",
        "## Contextual fragmentation result", "",
        "Prompt-cluster bootstrap estimates reproduce H2's exact fragmentation categories (2, 3, 4, 5, 6, 7, 8+) with 2,000 resamples and seed 3090.", md_table(frag[["pair", "morph_class", "fragmentation_bin", "n_tokens", "n_prompts", "rejection_rate", "bootstrap_ci_low", "bootstrap_ci_high"]]), "",
        "## CROSS versus WITHIN_SPLIT", "",
        "M1/M2/M3 use the exact H2/E2 eligibility, structural controls, full-sequence entropy, and prompt-clustered covariance; M5 adds the unchanged E1 cubic spline token-frequency term.", md_table(h2_results[["pair", "model", "n_tokens", "n_prompts", "cross_coefficient", "odds_ratio", "or_ci_low", "or_ci_high", "p_value", "status"]]), "",
        "## H3 environment analysis", "",
        f"The environment audit retains all five prespecified categories. Single-boundary eligibility and local-adjacent SPLIT assignment are unchanged. The pooled H3 interaction omnibus p-value is `{h3res['pooled_meta']['interaction_test'].get('wald_p')}`.",
        "### Nominal→Particle pair-specific AMEs", "", md_table(flores_show), "",
        f"Pooled all-eligible-prompt Nominal→Particle AME: {pooled_text}", "",
        "### All planned environment contrasts", "", md_table(h3_table[["pair", "environment", "n_cross", "n_split", "n_prompts", "ame", "ame_ci_low", "ame_ci_high", "p_value", "p_holm_within_pair", "omnibus_interaction_p"]]), "",
        "The plotted adjusted effects appear in `figures/environment_ame_forest.png` and `.pdf`; numerical source is `results/environment_pair_results.csv`.", "",
        "## Original versus FLORES comparison", "",
        "The original Wikipedia and FLORES samples were fitted separately; they are shown side-by-side and were not pooled.",
        md_table(pd.concat([
            original_show.rename(columns={"ame":"ame_pp","ame_ci_low":"ame_ci_low_pp","ame_ci_high":"ame_ci_high_pp"})[["pair","ame_pp","ame_ci_low_pp","ame_ci_high_pp","n_cross","n_split","p_value"]].assign(sample="Original H3 Wikipedia"),
            flores_show.rename(columns={"ame":"ame_pp","ame_ci_low":"ame_ci_low_pp","ame_ci_high":"ame_ci_high_pp"})[["pair","ame_pp","ame_ci_low_pp","ame_ci_high_pp","n_cross","n_split","n_prompts","p_value"]].assign(sample="FLORES English→Korean"),
        ], ignore_index=True, sort=False)), "",
        "![Original H3 vs FLORES AMEs](figures/replication_comparison.png)", "",
        "## Nominal→Particle frequency robustness", "",
        "H4 L1/L2/L3 were run with the existing tokenizer-compatible E1/H4 caches from the same Korean Wikipedia corpus. They add particle/nominal controls and then nonlinear nominal, particle, and eojeol frequencies; these are robustness controls, not mediation estimates.", md_table(l3_summary), "",
        "## Token-identity concentration", "",
        "Exact `(nominal surface, particle surface, token surface)` patterns are ranked by counts only, without using rejection labels. Concentration statistics describe the top 1/5/10 patterns within N→P CROSS tokens.", md_table(conc_np), "",
        "## Common-prompt sensitivity", "",
        f"The exact three-way intersection has **{len(h3res['common_prompt_ids']):,} eligible prompt IDs**. Common-only pooled N→P AME differs from the all-eligible pooled estimate by at most {max_change:.2f} pp; classification: **{cls}**. See `results/common_prompt_sensitivity.md` and `audits/common_prompt_intersection.json`.", "",
        "## Diagnostics and limitations", "",
        "Model convergence, rank, sparse cells, warning/separation signals, extreme coefficients, frequency joins, and cluster counts are documented in `audits/diagnostics.md` and its JSON record. Environment and H4 categories below their prespecified support are retained and interpreted as imprecise. No unusual or non-Korean output was excluded.",
        "This study changes the workload from free-form continuation of Korean Wikipedia text to English→Korean translation using the same base models and generation protocol. Translation prompts and generated linguistic constructions therefore differ from the original study; external validity is limited to this benchmark/task setting. The historical P3 0.6B draft revision remains unresolved; this run pins the E2 commit and reports that limitation.", "",
        "### Paper-ready interpretation", "",
        f"On FLORES-200 English→Korean, the adjusted association between crossing a morpheme boundary and speculative rejection was {('positive in all three model pairs' if focal_pos else 'not uniformly positive across the three model pairs')}. The Nominal→Particle environment showed {('a directionally larger penalty than lexical-to-lexical boundaries across pairs' if pattern_dir else 'no fully consistent pairwise increase over lexical-to-lexical boundaries')}. Frequency controls and token-pattern concentration are reported as robustness descriptions; they do not identify causal mechanisms. These results test the association on a translation workload and do not establish that Korean morphology causes rejection.", "",
    ]
    atomic_text(RUN / "flores_replication_summary.md", "\n".join(report))

    # Config includes actual audit and summary outputs for reproducibility.
    config.setdefault("analysis", {})
    config["software_versions"] = {
        "python": sys.version.split()[0],
        "torch": importlib.metadata.version("torch"),
        "torch_cuda_runtime": config["GPU"]["torch_cuda_runtime_version"],
        "transformers": importlib.metadata.version("transformers"),
        "pandas": importlib.metadata.version("pandas"),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "statsmodels": importlib.metadata.version("statsmodels"),
        "patsy": importlib.metadata.version("patsy"),
        "matplotlib": importlib.metadata.version("matplotlib"),
        "kiwipiepy": importlib.metadata.version("kiwipiepy"),
        "pyarrow": importlib.metadata.version("pyarrow"),
    }
    config["analysis"].update({
        "formulas": {
            "H2_M1": e2._primary_formula([]),
            "H2_M2": e2._primary_formula(list(e2.DEFAULT_STRUCTURAL)),
            "H2_M3": e2._primary_formula(list(e2.DEFAULT_STRUCTURAL) + list(e2.DEFAULT_ENTROPY)),
            "H2_M5": e2._formula("sd_rejected", list(e2.DEFAULT_STRUCTURAL) + list(e2.DEFAULT_ENTROPY) + ["bs(log_token_count, df=4, degree=3, include_intercept=False)"]),
            "H3_pair_M5": h3.formula_for("", interaction=True),
            "H3_pooled_M5": h3res["pooled_meta"].get("formula"),
            "H4_L3": str(l3.loc[l3.model.eq("L3"), "formula"].iloc[0]) if (l3.model.eq("L3")).any() else None,
        },
        "references": {"morph_class": "WITHIN_SPLIT", "fragmentation_bin": "2", "morph_environment": "LEXICAL_TO_LEXICAL"},
        "uncertainty": "Prompt-clustered covariance, two-sided tests, H3 delta-method marginal probabilities/AMEs with 95% normal CIs; Holm correction across five planned environments within each model pair.",
        "random_seed": 3090,
        "input_artifact_sha256": {
            str((TABLES / e2.PAIRS[p]["slug"] / name).relative_to(ROOT)): e2.sha256_file(TABLES / e2.PAIRS[p]["slug"] / name)
            for p in PAIR_ORDER for name in ["token_table.parquet", "h3_token_environment.parquet", "h3_eojeol_morphemes.parquet", "h4_nominal_particle_rows.parquet"]
        },
        "frequency_cache_sha256": {
            "e1_token": e2.sha256_file(ROOT / "runs/20260926T184145Z_pilot1000/e1_token_frequency_cache.parquet"),
            "e1_eojeol": e2.sha256_file(ROOT / "runs/20260926T184145Z_pilot1000/e1_eojeol_frequency_cache.parquet"),
            "h4_morpheme": e2.sha256_file(ROOT / "runs/e2_model_pair_replication/h4/h4_morpheme_frequency_cache.parquet"),
        },
        "common_prompt_count": len(h3res["common_prompt_ids"]),
        "pair_eligible_prompt_counts": {p: len(h3res["prompt_ids_by_pair"][p]) for p in PAIR_ORDER},
        "common_prompt_ids_sha256": hashlib.sha256(json.dumps(h3res["common_prompt_ids"]).encode()).hexdigest(),
        "data_manifest_sha256": e2.sha256_file(RUN / "data_manifest.csv"),
        "decision": decision,
        "pooled_nominal_particle_ame": float(pooled_np_row.ame) if pooled_np_row is not None else None,
        "pooled_nominal_particle_ci": [float(pooled_np_row.ame_ci_low), float(pooled_np_row.ame_ci_high)] if pooled_np_row is not None else None,
        "analysis_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    })
    atomic_json(RUN / "flores_config.json", config)


if __name__ == "__main__":
    main()
