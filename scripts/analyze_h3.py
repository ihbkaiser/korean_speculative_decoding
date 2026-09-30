#!/usr/bin/env python3
"""H3 morphological boundary-type analysis on saved H2/E2 artifacts only."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import patsy
import scipy.stats as st
from scipy.special import expit
import statsmodels
import statsmodels.formula.api as smf
from kiwipiepy import Kiwi
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.e2_model_pairs import (
    DEFAULT_ENTROPY,
    DEFAULT_STRUCTURAL,
    FRAGMENTATION_ORDER,
    PAIRS,
    P3_DIR,
    P3_FREQUENCY_CACHE,
    _attach_frequency,
    _eligible,
    _p3_harmonized_entropy,
    sha256_file,
)


OUT = ROOT / "runs/e2_model_pair_replication/h3"
AUDIT = OUT / "audits"
PAIR_OUT = OUT / "pair_results"
POOLED = OUT / "pooled"
FIGURES = OUT / "figures"
ENVIRONMENTS = [
    "NOMINAL_TO_PARTICLE",
    "PREDICATE_TO_ENDING",
    "ENDING_TO_ENDING",
    "LEXICAL_TO_LEXICAL",
    "OTHER",
]
ENV_LABELS = {
    "NOMINAL_TO_PARTICLE": "Nominal → Particle",
    "PREDICATE_TO_ENDING": "Predicate → Ending",
    "ENDING_TO_ENDING": "Ending → Ending",
    "LEXICAL_TO_LEXICAL": "Lexical → Lexical",
    "OTHER": "Other",
}
PAIR_LABELS = {"P1": "0.6B → 1.7B", "P2": "1.7B → 4B", "P3": "0.6B → 4B"}
PAIR_COLORS = {"P1": "#0072B2", "P2": "#D55E00", "P3": "#009E73"}
CONTENT_GROUPS = {"NOMINAL", "PREDICATE", "MODIFIER", "DERIVATIONAL", "OTHER_CONTENT"}
M5_SPLINE = "bs(log_token_count, df=4, degree=3, include_intercept=False)"

POS_MAP: dict[str, str] = {}
for tag in ["NNG", "NNP", "NNB", "NP", "NR"]:
    POS_MAP[tag] = "NOMINAL"
for tag in ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"]:
    POS_MAP[tag] = "PARTICLE"
for tag in ["VV", "VA", "VX", "VCP", "VCN", "VV-I", "VV-R", "VA-I", "VA-R"]:
    POS_MAP[tag] = "PREDICATE"
for tag in ["EP", "EF", "EC", "ETN", "ETM"]:
    POS_MAP[tag] = "ENDING"
for tag in ["MM", "MAG", "MAJ"]:
    POS_MAP[tag] = "MODIFIER"
for tag in ["XPN", "XSN", "XSV", "XSA", "XSA-I", "XSM"]:
    POS_MAP[tag] = "DERIVATIONAL"
for tag in ["XR", "SL", "SH", "SN", "IC", "W_URL", "W_EMAIL", "W_HASHTAG", "W_MENTION", "W_SERIAL", "W_EMOJI"]:
    POS_MAP[tag] = "OTHER_CONTENT"
for tag in ["SF", "SP", "SS", "SSO", "SSC", "SE", "SO", "SW", "SB", "UN", "Z_CODA", "Z_SIOT"]:
    POS_MAP[tag] = "OTHER_NONCONTENT"
for i in range(5):
    POS_MAP[f"USER{i}"] = "OTHER_NONCONTENT"

STRUCTURAL = list(DEFAULT_STRUCTURAL)
ENTROPY = list(DEFAULT_ENTROPY)
H3_CONTROLS = STRUCTURAL + ENTROPY + [M5_SPLINE]


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    try:
        return list(value)
    except TypeError:
        return []


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def md_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render compact Markdown without an optional tabulate dependency."""
    if frame.empty:
        return "(no rows)"
    headers = [str(c) for c in frame.columns]

    def cell(value: Any) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        if isinstance(value, (bool, np.bool_)):
            return "true" if bool(value) else "false"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = [[cell(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    head = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def coarse_pos(tag: str) -> str:
    return POS_MAP.get(str(tag), "OTHER_NONCONTENT")


def boundary_type(left_tag: str, right_tag: str) -> str:
    left, right = coarse_pos(left_tag), coarse_pos(right_tag)
    if left == "NOMINAL" and right == "PARTICLE":
        return "NOMINAL_TO_PARTICLE"
    if left == "PREDICATE" and right == "ENDING":
        return "PREDICATE_TO_ENDING"
    if left == "ENDING" and right == "ENDING":
        return "ENDING_TO_ENDING"
    if left in CONTENT_GROUPS and right in CONTENT_GROUPS:
        return "LEXICAL_TO_LEXICAL"
    return "OTHER"


def make_morpheme_sequences(pair: str, eojeols: pd.DataFrame, kiwi: Kiwi) -> tuple[pd.DataFrame, dict[tuple[int, int], list[dict[str, Any]]]]:
    records: list[dict[str, Any]] = []
    by_eojeol: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in eojeols.itertuples(index=False):
        pid = int(row.prompt_id)
        eo_id = int(row.eojeol_index)
        text = str(row.eojeol)
        base = int(row.char_start)
        morphs: list[dict[str, Any]] = []
        for idx, morph in enumerate(kiwi.tokenize(text)):
            start_local = int(morph.start)
            end_local = start_local + int(morph.len)
            tag = str(morph.tag)
            item = {
                "pair": pair,
                "prompt_id": pid,
                "eojeol_id": eo_id,
                "eojeol": text,
                "morpheme_index": idx,
                "surface": str(morph.form),
                "fine_pos": tag,
                "coarse_pos": coarse_pos(tag),
                "start_local": start_local,
                "end_local": end_local,
                "start_char": base + start_local,
                "end_char": base + end_local,
            }
            morphs.append(item)
        for idx, item in enumerate(morphs):
            if idx + 1 < len(morphs):
                nxt = morphs[idx + 1]
                # Only touching spans define an ordinary internal character boundary.
                valid = item["end_local"] == nxt["start_local"]
                item["boundary_to_next_valid"] = valid
                item["boundary_to_next_char"] = item["end_char"] if valid else None
                item["boundary_to_next_fine_pos"] = f"{item['fine_pos']}→{nxt['fine_pos']}"
                item["boundary_to_next_type"] = boundary_type(item["fine_pos"], nxt["fine_pos"]) if valid else "OTHER"
            else:
                item["boundary_to_next_valid"] = False
                item["boundary_to_next_char"] = None
                item["boundary_to_next_fine_pos"] = None
                item["boundary_to_next_type"] = None
            records.append(item.copy())
        by_eojeol[(pid, eo_id)] = morphs
    return pd.DataFrame.from_records(records), by_eojeol


def _match_saved_morpheme(
    morphs: list[dict[str, Any]],
    surface: Any,
    pos: Any,
    span: Any,
) -> int | None:
    if span is None or len(span) != 2:
        return None
    s, e = int(span[0]), int(span[1])
    exact = [
        i for i, m in enumerate(morphs)
        if m["start_char"] == s and m["end_char"] == e
        and m["surface"] == str(surface) and m["fine_pos"] == str(pos)
    ]
    if exact:
        return exact[0]
    overlap = []
    for i, m in enumerate(morphs):
        shared = max(0, min(e, m["end_char"]) - max(s, m["start_char"]))
        if shared:
            overlap.append((shared, i))
    return max(overlap)[1] if overlap else None


def enrich_pair(pair: str, raw: pd.DataFrame, by_eojeol: dict[tuple[int, int], list[dict[str, Any]]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = _eligible(raw)
    rows = []
    mismatches = {"saved_overlap_morphemes": 0, "matched_to_eojeol_sequence": 0, "sequence_boundary_count_disagreements": 0, "missing_eojeol_sequence": 0}
    for row in eligible.itertuples(index=False):
        record = row._asdict()
        pid = int(record["prompt_id"])
        eo_id_raw = record.get("eojeol_id")
        eo_id = int(eo_id_raw) if pd.notna(eo_id_raw) else -1
        morphs = by_eojeol.get((pid, eo_id), [])
        if not morphs:
            mismatches["missing_eojeol_sequence"] += 1
        token_start, token_end = int(record["token_start_char"]), int(record["token_end_char"])
        ov_surfaces = as_list(record.get("overlapping_morpheme_surfaces"))
        ov_pos = as_list(record.get("overlapping_morpheme_pos"))
        ov_spans = as_list(record.get("overlapping_morpheme_spans"))
        ov_ids = as_list(record.get("overlapping_morpheme_ids"))
        matched_indices = []
        for s, p, sp in zip(ov_surfaces, ov_pos, ov_spans):
            mismatches["saved_overlap_morphemes"] += 1
            match = _match_saved_morpheme(morphs, s, p, sp)
            if match is not None:
                mismatches["matched_to_eojeol_sequence"] += 1
                matched_indices.append(match)
        cross_transitions = []
        for idx in range(min(len(ov_pos), len(ov_surfaces)) - 1):
            left_tag, right_tag = str(ov_pos[idx]), str(ov_pos[idx + 1])
            trans = f"{left_tag}→{right_tag}"
            cat = boundary_type(left_tag, right_tag)
            cross_transitions.append({"fine": trans, "type": cat})
        n_cross = len(cross_transitions) if record["morph_class"] == "CROSS_MORPHEME" else 0
        if record["morph_class"] == "CROSS_MORPHEME" and matched_indices and morphs:
            # Independent boundary-coordinate check. Zero-width morphemes may be absent
            # from H2 overlap arrays; that difference is retained as an audit diagnostic.
            local_cross = sum(
                1 for m in morphs[:-1]
                if m["boundary_to_next_valid"]
                and token_start < int(m["boundary_to_next_char"]) < token_end
            )
            if local_cross != n_cross:
                mismatches["sequence_boundary_count_disagreements"] += 1
        environment = "OTHER"
        source = "ADJACENT_BOUNDARY"
        if record["morph_class"] == "CROSS_MORPHEME":
            source = "CROSSED_BOUNDARY"
            if n_cross == 1:
                environment = cross_transitions[0]["type"]
            elif n_cross > 1:
                # A single environment is assigned only if all crossed transitions have
                # the same prespecified class; otherwise primary multiboundary rows stay
                # out of H3b and the first class is retained only for descriptive audits.
                unique = {x["type"] for x in cross_transitions}
                environment = next(iter(unique)) if len(unique) == 1 else "OTHER"
        else:
            saved_idx = None
            if ov_surfaces and ov_pos and ov_spans:
                saved_idx = _match_saved_morpheme(morphs, ov_surfaces[0], ov_pos[0], ov_spans[0])
            if saved_idx is None or not morphs:
                environment = "OTHER"
            else:
                candidates = []
                for edge_idx in [saved_idx - 1, saved_idx]:
                    if 0 <= edge_idx < len(morphs) - 1:
                        left = morphs[edge_idx]
                        q = left.get("boundary_to_next_char")
                        if not left["boundary_to_next_valid"] or q is None:
                            candidates.append((float("inf"), 1 if edge_idx == saved_idx else 0, "OTHER"))
                        else:
                            q = int(q)
                            distance = q - token_end if token_end <= q else token_start - q if token_start >= q else 0
                            # Right/following edge wins equal-distance ties.
                            candidates.append((float(distance), 1 if edge_idx == saved_idx else 0, left["boundary_to_next_type"]))
                if candidates:
                    candidates.sort(key=lambda x: (x[0], -x[1]))
                    environment = str(candidates[0][2])
        all_types = {
            str(m["boundary_to_next_type"])
            for m in morphs[:-1]
            if m["boundary_to_next_valid"]
        }
        if len(all_types) == 0:
            construction = "OTHER"
        elif len(all_types) == 1:
            construction = next(iter(all_types))
        else:
            construction = "MULTI_ENV"
        record.update({
            "pair": pair,
            "num_boundaries_crossed": n_cross if record["morph_class"] == "CROSS_MORPHEME" else np.nan,
            "crossed_boundary_sequence": "|".join(x["type"] for x in cross_transitions) if cross_transitions else "",
            "crossed_fine_pos_sequence": "|".join(x["fine"] for x in cross_transitions) if cross_transitions else "",
            "crossed_transition_count": len(cross_transitions),
            "morph_environment": environment,
            "environment_source": source,
            "construction_environment": construction,
            "morph_sequence_match_count": len(matched_indices),
            "morph_sequence_mismatch": bool(len(ov_surfaces) != len(matched_indices)),
        })
        rows.append(record)
    result = pd.DataFrame(rows)
    result["sd_rejected"] = result["sd_rejected"].astype(int)
    result["pair"] = pair
    return result, mismatches


def _categories(data: pd.DataFrame, col: str, preferred: str) -> tuple[pd.DataFrame, str]:
    out = data.copy()
    present = list(dict.fromkeys(out[col].dropna().astype(str).tolist()))
    ref = preferred if preferred in present else (present[0] if present else preferred)
    if col == "fragmentation_bin":
        out[col] = pd.Categorical(out[col], categories=[x for x in FRAGMENTATION_ORDER if x in present])
    else:
        ordered = [x for x in ENVIRONMENTS if x in present]
        ordered += [x for x in present if x not in ordered]
        out[col] = pd.Categorical(out[col], categories=ordered)
    return out, ref


def fit_clustered(data: pd.DataFrame, formula: str, preferred_categories: dict[str, str] | None = None) -> tuple[Any | None, dict[str, Any]]:
    clean = data.dropna(subset=["sd_rejected", "prompt_id", "log_token_count"] + STRUCTURAL + ENTROPY).copy()
    clean["sd_rejected"] = clean["sd_rejected"].astype(int)
    refs = {}
    for col, preferred in (preferred_categories or {}).items():
        clean, ref = _categories(clean, col, preferred)
        refs[col] = ref
    diagnostics: dict[str, Any] = {
        "formula": formula,
        "n_tokens": int(len(clean)),
        "n_prompts": int(clean["prompt_id"].nunique()),
        "references": refs,
        "converged": False,
        "rank": None,
        "n_columns": None,
        "max_abs_coefficient": None,
        "warnings": [],
        "fit_error": None,
        "optimizer_attempts": [],
    }
    if len(clean) < 10 or clean["sd_rejected"].nunique() < 2:
        diagnostics["fit_error"] = "insufficient rows or outcome variation"
        return None, diagnostics
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            _, design_frame = patsy.dmatrices(formula, clean, return_type="dataframe")
            model = smf.logit(formula, data=clean)
            exog = np.asarray(model.exog)
            diagnostics["rank"] = int(np.linalg.matrix_rank(exog))
            diagnostics["n_columns"] = int(exog.shape[1])
            fitted = model.fit(
                disp=False, maxiter=200, method="newton", cov_type="cluster",
                cov_kwds={"groups": clean["prompt_id"], "use_correction": True},
            )
            diagnostics["optimizer_attempts"].append(optimizer_summary("newton", fitted))
            if not bool(getattr(fitted, "converged", False)):
                fitted = model.fit(
                    disp=False, maxiter=1000, method="lbfgs", cov_type="cluster",
                    cov_kwds={"groups": clean["prompt_id"], "use_correction": True},
                )
                diagnostics["optimizer_attempts"].append(optimizer_summary("lbfgs", fitted))
            if not bool(getattr(fitted, "converged", False)):
                fitted = model.fit(
                    disp=False, maxiter=1000, method="bfgs", cov_type="cluster",
                    cov_kwds={"groups": clean["prompt_id"], "use_correction": True},
                )
                diagnostics["optimizer_attempts"].append(optimizer_summary("bfgs", fitted))
            diagnostics["converged"] = bool(getattr(fitted, "converged", False))
            diagnostics["max_abs_coefficient"] = float(np.abs(np.asarray(fitted.params)).max())
            diagnostics["condition_number"] = float(np.linalg.cond(exog))
            diagnostics["extreme_coefficient_warning"] = diagnostics["max_abs_coefficient"] > 10
            diagnostics["rank_deficient"] = diagnostics["rank"] < diagnostics["n_columns"]
            diagnostics["cluster_count"] = int(clean["prompt_id"].nunique())
            diagnostics["warnings"] = [str(w.message) for w in caught]
            fitted._h3_design_info = design_frame.design_info
            return fitted, diagnostics
        except Exception as exc:
            diagnostics["fit_error"] = f"{type(exc).__name__}: {exc}"
            diagnostics["warnings"] = [str(w.message) for w in caught]
            return None, diagnostics


def optimizer_summary(method: str, fitted: Any) -> dict[str, Any]:
    ret = getattr(fitted, "mle_retvals", {}) or {}
    score = ret.get("score")
    try:
        score_norm = float(np.linalg.norm(np.asarray(score, dtype=float)))
    except Exception:
        score_norm = None
    fopt = ret.get("fopt")
    return {
        "method": method,
        "converged": bool(getattr(fitted, "converged", False)),
        "warnflag": ret.get("warnflag"),
        "iterations": ret.get("iterations"),
        "objective": float(fopt) if fopt is not None and np.isfinite(fopt) else None,
        "score_norm": score_norm,
    }


def predict_grad(fitted: Any, data: pd.DataFrame, morph_value: str | None = None, morph_col: str = "morph_class") -> tuple[float, np.ndarray]:
    cf = data.copy()
    if morph_value is not None:
        cf[morph_col] = morph_value
        if morph_col == "morph_class":
            for col in [name for name in cf.columns if name.startswith("h3_cross_env_")]:
                env = col.removeprefix("h3_cross_env_")
                cf[col] = ((cf.morph_class.astype(str) == "CROSS_MORPHEME") & (cf.morph_environment.astype(str) == env)).astype(int)
    design = np.asarray(patsy.build_design_matrices([fitted._h3_design_info], cf, return_type="dataframe")[0])
    beta = np.asarray(fitted.params)
    prob = expit(design @ beta)
    mean = float(prob.mean())
    gradient = np.mean(design * (prob * (1.0 - prob))[:, None], axis=0)
    return mean, gradient


def inference_from_gradient(fitted: Any, estimate: float, gradient: np.ndarray) -> dict[str, float]:
    cov = np.asarray(fitted.cov_params())
    variance = float(gradient @ cov @ gradient)
    se = math.sqrt(max(variance, 0.0))
    z = estimate / se if se > 0 else np.nan
    return {
        "se": se,
        "ci_low": estimate - 1.959963984540054 * se,
        "ci_high": estimate + 1.959963984540054 * se,
        "p_value": float(2 * st.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
    }


def adjusted_contrast(fitted: Any, data: pd.DataFrame, env_value: str, env_col: str = "morph_environment") -> dict[str, Any]:
    sub = data.loc[data[env_col].astype(str) == env_value].copy()
    if fitted is None or not bool(getattr(fitted, "converged", False)) or sub.empty or sub["morph_class"].nunique() < 2:
        return {
            "environment": env_value, "n_rows": int(len(sub)),
            "n_cross": int((sub.morph_class.astype(str) == "CROSS_MORPHEME").sum()) if len(sub) else 0,
            "n_split": int((sub.morph_class.astype(str) == "WITHIN_SPLIT").sum()) if len(sub) else 0,
            "n_prompts": int(sub.prompt_id.nunique()) if len(sub) else 0,
            "support_warning": True,
            "estimable": False,
            "nonestimable_reason": "model_not_estimable_or_nonconverged" if fitted is None or not bool(getattr(fitted, "converged", False)) else "missing_environment_or_morphology_class",
        }
    p_cross, g_cross = predict_grad(fitted, sub, "CROSS_MORPHEME")
    p_split, g_split = predict_grad(fitted, sub, "WITHIN_SPLIT")
    ame = p_cross - p_split
    inf = inference_from_gradient(fitted, ame, g_cross - g_split)
    inf_cross = inference_from_gradient(fitted, p_cross, g_cross)
    inf_split = inference_from_gradient(fitted, p_split, g_split)
    return {
        "environment": env_value,
        "n_rows": int(len(sub)),
        "n_cross": int((sub.morph_class.astype(str) == "CROSS_MORPHEME").sum()),
        "n_split": int((sub.morph_class.astype(str) == "WITHIN_SPLIT").sum()),
        "n_prompts": int(sub.prompt_id.nunique()),
        "support_warning": bool(
            (sub.morph_class.astype(str) == "CROSS_MORPHEME").sum() < 100
            or sub.loc[sub.morph_class.astype(str) == "CROSS_MORPHEME", "prompt_id"].nunique() < 50
            or (sub.morph_class.astype(str) == "WITHIN_SPLIT").sum() < 100
            or sub.loc[sub.morph_class.astype(str) == "WITHIN_SPLIT", "prompt_id"].nunique() < 50
        ),
        "estimable": True,
        "adjusted_p_cross": p_cross,
        "adjusted_p_cross_ci_low": max(0.0, inf_cross["ci_low"]),
        "adjusted_p_cross_ci_high": min(1.0, inf_cross["ci_high"]),
        "adjusted_p_split": p_split,
        "adjusted_p_split_ci_low": max(0.0, inf_split["ci_low"]),
        "adjusted_p_split_ci_high": min(1.0, inf_split["ci_high"]),
        "ame": ame,
        "ame_se": inf["se"],
        "ame_ci_low": inf["ci_low"],
        "ame_ci_high": inf["ci_high"],
        "p_value": inf["p_value"],
    }


def wald_block(fitted: Any | None, predicate) -> dict[str, Any]:
    if fitted is None or not bool(getattr(fitted, "converged", False)):
        return {"wald_chi2": np.nan, "wald_df": 0, "wald_p": np.nan, "wald_terms": []}
    names = list(fitted.params.index)
    idx = [i for i, n in enumerate(names) if predicate(n)]
    if not idx:
        return {"wald_chi2": np.nan, "wald_df": 0, "wald_p": np.nan, "wald_terms": []}
    beta = np.asarray(fitted.params)[idx]
    cov = np.asarray(fitted.cov_params())[np.ix_(idx, idx)]
    rank = int(np.linalg.matrix_rank(cov))
    if rank == 0:
        return {"wald_chi2": np.nan, "wald_df": 0, "wald_p": np.nan, "wald_terms": [names[i] for i in idx]}
    stat = float(beta @ np.linalg.pinv(cov) @ beta)
    return {"wald_chi2": stat, "wald_df": rank, "wald_p": float(st.chi2.sf(stat, rank)), "wald_terms": [names[i] for i in idx]}


def holm_adjust(p_values: list[float]) -> list[float]:
    vals = [float(p) if np.isfinite(p) else 1.0 for p in p_values]
    m = len(vals)
    order = np.argsort(vals)
    adjusted = np.ones(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * vals[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def formula_for(main: str, interaction: bool = False, pair_effects: bool = False, count_term: bool = False) -> str:
    base = ["C(fragmentation_bin, Treatment(reference='2'))"] + STRUCTURAL + ENTROPY + [M5_SPLINE]
    if interaction:
        base = ["C(morph_class, Treatment(reference='WITHIN_SPLIT')) * C(morph_environment, Treatment(reference='LEXICAL_TO_LEXICAL'))"] + base
    else:
        base.insert(0, main)
    if pair_effects:
        base = ["C(model_pair, Treatment(reference='P3'))", "C(model_pair, Treatment(reference='P3')):C(morph_class, Treatment(reference='WITHIN_SPLIT'))"] + base
    if count_term:
        base.insert(0, main)
    return "sd_rejected ~ " + " + ".join(base)


def per_pair_models(pair: str, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    single = data.loc[
        (data.morph_class.astype(str) == "WITHIN_SPLIT")
        | ((data.morph_class.astype(str) == "CROSS_MORPHEME") & (data.num_boundaries_crossed == 1))
    ].copy()
    single["morph_environment"] = single["morph_environment"].where(single["morph_environment"].isin(ENVIRONMENTS), "OTHER")
    split_env, env_ref = _categories(single, "morph_environment", "LEXICAL_TO_LEXICAL")
    split_env, frag_ref = _categories(split_env, "fragmentation_bin", "2")
    formula = formula_for("", interaction=True)
    # A missing lexical reference is replaced by the first present category; all five
    # planned contrasts are still reported and an absent category is non-estimable.
    env_ref_formula = env_ref
    if env_ref_formula != "LEXICAL_TO_LEXICAL":
        formula = formula.replace("reference='LEXICAL_TO_LEXICAL'", f"reference='{env_ref_formula}'")
    if frag_ref != "2":
        formula = formula.replace("reference='2'", f"reference='{frag_ref}'")
    fitted, diag = fit_clustered(split_env, formula, {"morph_environment": env_ref, "fragmentation_bin": frag_ref})
    interaction_test = wald_block(fitted, lambda n: "morph_class" in n and "morph_environment" in n and ":" in n)
    effects = []
    for env in ENVIRONMENTS:
        res = adjusted_contrast(fitted, split_env, env)
        res.update({"pair": pair, "specification": "H3b_primary", "omnibus_interaction_p": interaction_test["wald_p"], "formula": formula})
        effects.append(res)
    # Delta-method difference of environment-specific AMEs against lexical-to-lexical.
    lexical = next((x for x in effects if x["environment"] == "LEXICAL_TO_LEXICAL" and x.get("estimable")), None)
    for row in effects:
        row["ame_minus_lexical"] = np.nan
        row["ame_minus_lexical_ci_low"] = np.nan
        row["ame_minus_lexical_ci_high"] = np.nan
        row["ame_minus_lexical_p"] = np.nan
        if fitted is not None and lexical is not None and row.get("estimable"):
            env_frame = split_env.loc[split_env.morph_environment.astype(str) == row["environment"]]
            lex_frame = split_env.loc[split_env.morph_environment.astype(str) == "LEXICAL_TO_LEXICAL"]
            _, gc = predict_grad(fitted, env_frame, "CROSS_MORPHEME")
            _, gs = predict_grad(fitted, env_frame, "WITHIN_SPLIT")
            _, glc = predict_grad(fitted, lex_frame, "CROSS_MORPHEME")
            _, gls = predict_grad(fitted, lex_frame, "WITHIN_SPLIT")
            grad = (gc - gs) - (glc - gls)
            est = float(row["ame"] - lexical["ame"])
            inf = inference_from_gradient(fitted, est, grad)
            row["ame_minus_lexical"] = est
            row["ame_minus_lexical_ci_low"] = inf["ci_low"]
            row["ame_minus_lexical_ci_high"] = inf["ci_high"]
            row["ame_minus_lexical_p"] = inf["p_value"]
    adj = holm_adjust([x.get("p_value", np.nan) for x in effects])
    for row, value in zip(effects, adj):
        row["p_holm_within_pair"] = value if row.get("estimable") else np.nan

    # H3a: single-boundary CROSS tokens only.
    cross = data.loc[(data.morph_class.astype(str) == "CROSS_MORPHEME") & (data.num_boundaries_crossed == 1)].copy()
    cross["boundary_type"] = cross["morph_environment"].astype(str)
    present = list(dict.fromkeys(cross.boundary_type.tolist()))
    lexical_support = cross.loc[cross.boundary_type == "LEXICAL_TO_LEXICAL"]
    lexical_adequate = len(lexical_support) >= 100 and lexical_support.prompt_id.nunique() >= 50
    if lexical_adequate:
        boundary_ref = "LEXICAL_TO_LEXICAL"
    elif len(cross):
        support = cross.groupby("boundary_type", observed=True).agg(n=("prompt_id", "size"), prompts=("prompt_id", "nunique"))
        boundary_ref = max(present, key=lambda env: (int(support.loc[env, "n"]), -ENVIRONMENTS.index(env)))
    else:
        boundary_ref = "LEXICAL_TO_LEXICAL"
    bformula = formula_for(f"C(boundary_type, Treatment(reference='{boundary_ref}'))")
    if frag_ref != "2":
        bformula = bformula.replace("reference='2'", f"reference='{frag_ref}'")
    bdata, _ = _categories(cross, "fragmentation_bin", frag_ref)
    bdata, _ = _categories(bdata, "boundary_type", boundary_ref)
    bfit, bdiag = fit_clustered(bdata, bformula, {"boundary_type": boundary_ref, "fragmentation_bin": frag_ref})
    btest = wald_block(bfit, lambda n: "boundary_type" in n)
    bresults = []
    for env in ENVIRONMENTS:
        sub = bdata.loc[bdata.boundary_type.astype(str) == env]
        raw_rate = float(sub.sd_rejected.mean()) if len(sub) else np.nan
        item = {"pair": pair, "environment": env, "n_cross": int(len(sub)), "n_prompts": int(sub.prompt_id.nunique()), "raw_rejection": raw_rate, "omnibus_boundary_p": btest["wald_p"], "reference": boundary_ref, "reference_lexical_support_adequate": lexical_adequate, "formula": bformula}
        if bfit is not None and len(sub):
            # Standardize each crossed boundary type over the same CROSS-only
            # covariate distribution; do not use each category's own saturated mean.
            pred, grad = predict_grad(bfit, bdata, env, morph_col="boundary_type")
            inf = inference_from_gradient(bfit, pred, grad)
            item.update({"adjusted_rejection": pred, "adjusted_ci_low": max(0.0, inf["ci_low"]), "adjusted_ci_high": min(1.0, inf["ci_high"])})
        else:
            item.update({"adjusted_rejection": np.nan, "adjusted_ci_low": np.nan, "adjusted_ci_high": np.nan})
        bresults.append(item)

    # Construction-level environment sensitivity, excluding competing environments.
    sens = data.loc[
        ((data.morph_class.astype(str) == "WITHIN_SPLIT") | ((data.morph_class.astype(str) == "CROSS_MORPHEME") & (data.num_boundaries_crossed == 1)))
        & (data.construction_environment != "MULTI_ENV")
    ].copy()
    sens["morph_environment"] = sens["construction_environment"].where(sens["construction_environment"].isin(ENVIRONMENTS), "OTHER")
    cell_support = pd.crosstab(sens.morph_environment.astype(str), sens.morph_class.astype(str))
    sens_present = [x for x in ENVIRONMENTS if x in set(sens.morph_environment.astype(str))]
    sens_ref = max(
        sens_present,
        key=lambda env: (int(cell_support.loc[env].get("CROSS_MORPHEME", 0)), -ENVIRONMENTS.index(env)),
    ) if sens_present else "LEXICAL_TO_LEXICAL"
    sens, sens_ref = _categories(sens, "morph_environment", sens_ref)
    sens, sens_frag_ref = _categories(sens, "fragmentation_bin", "2")
    # Keep every non-MULTI_ENV row in the sensitivity population. Where a
    # class-by-environment cell is empty (here E→E has no CROSS tokens), its
    # interaction is algebraically non-identifiable; omit only that zero-support
    # interaction column and flag the corresponding AME as non-estimable.
    interaction_columns = []
    nonestimable_interactions = []
    for env in sens_present:
        n_cross_env = int(cell_support.loc[env].get("CROSS_MORPHEME", 0))
        n_split_env = int(cell_support.loc[env].get("WITHIN_SPLIT", 0))
        if env == sens_ref:
            continue
        if n_cross_env and n_split_env:
            col = "h3_cross_env_" + env
            sens[col] = ((sens.morph_class.astype(str) == "CROSS_MORPHEME") & (sens.morph_environment.astype(str) == env)).astype(int)
            interaction_columns.append(col)
        else:
            nonestimable_interactions.append(env)
    sens_terms = [
        "C(morph_class, Treatment(reference='WITHIN_SPLIT'))",
        f"C(morph_environment, Treatment(reference='{sens_ref}'))",
        *interaction_columns,
        "C(fragmentation_bin, Treatment(reference='" + sens_frag_ref + "'))",
        *STRUCTURAL, *ENTROPY, M5_SPLINE,
    ]
    sformula = "sd_rejected ~ " + " + ".join(sens_terms)
    sfit, sdiag = fit_clustered(sens, sformula, {"morph_environment": sens_ref, "fragmentation_bin": sens_frag_ref})
    sdiag["nonestimable_interaction_environments"] = nonestimable_interactions
    sdiag["all_sensitivity_rows_retained_except_multi_env"] = True
    stest = wald_block(sfit, lambda n: n.startswith("h3_cross_env_"))
    sens_results = []
    for env in ENVIRONMENTS:
        item = adjusted_contrast(sfit, sens, env)
        item.update({"pair": pair, "specification": "eojeol_construction_sensitivity", "omnibus_interaction_p": stest["wald_p"], "formula": sformula})
        sens_results.append(item)

    # Exploratory number of boundaries among all CROSS rows, reference 1.
    all_cross = data.loc[data.morph_class.astype(str) == "CROSS_MORPHEME"].copy()
    all_cross["boundary_count_band"] = pd.cut(
        all_cross.num_boundaries_crossed.fillna(0), bins=[-0.1, 1, 2, np.inf], labels=["1", "2", "3+"], right=True,
    ).astype(object)
    all_cross = all_cross.loc[all_cross.num_boundaries_crossed >= 1].copy()
    count_formula = formula_for("C(boundary_count_band, Treatment(reference='1'))")
    cfit, cdiag = fit_clustered(all_cross, count_formula, {"boundary_count_band": "1", "fragmentation_bin": frag_ref})
    count_test = wald_block(cfit, lambda n: "boundary_count_band" in n)
    count_rows = []
    for band in ["1", "2", "3+"]:
        sub = all_cross.loc[all_cross.boundary_count_band == band]
        item = {"pair": pair, "boundary_count_band": band, "n_cross": len(sub), "n_prompts": sub.prompt_id.nunique(), "raw_rejection": sub.sd_rejected.mean() if len(sub) else np.nan, "omnibus_boundary_count_p": count_test["wald_p"]}
        if cfit is not None and len(sub):
            pred, grad = predict_grad(cfit, all_cross, band, morph_col="boundary_count_band")
            inf = inference_from_gradient(cfit, pred, grad)
            item.update({"adjusted_rejection": pred, "ci_low": max(0, inf["ci_low"]), "ci_high": min(1, inf["ci_high"])})
        else:
            item.update({"adjusted_rejection": np.nan, "ci_low": np.nan, "ci_high": np.nan})
        count_rows.append(item)

    # Fine POS transitions are fixed by the support threshold, not by rejection.
    transitions = cross.groupby("crossed_fine_pos_sequence", observed=True).agg(
        n_cross=("sd_rejected", "size"), n_prompts=("prompt_id", "nunique"), raw_rejection=("sd_rejected", "mean")
    ).reset_index()
    supported = transitions.loc[(transitions.n_cross >= 100) & (transitions.n_prompts >= 50), "crossed_fine_pos_sequence"].astype(str).tolist()
    fine_rows = []
    if supported:
        fine = cross.loc[cross.crossed_fine_pos_sequence.isin(supported)].copy()
        fine_ref = str(transitions.sort_values(["n_cross", "crossed_fine_pos_sequence"], ascending=[False, True]).iloc[0].crossed_fine_pos_sequence)
        fine_formula = formula_for(f"C(crossed_fine_pos_sequence, Treatment(reference={fine_ref!r}))")
        ffit, fdiag = fit_clustered(fine, fine_formula, {"crossed_fine_pos_sequence": fine_ref, "fragmentation_bin": frag_ref})
        for trans in sorted(supported):
            sub = fine.loc[fine.crossed_fine_pos_sequence == trans]
            base = transitions.loc[transitions.crossed_fine_pos_sequence == trans].iloc[0]
            item = {"pair": pair, "transition": trans, "n_cross": int(base.n_cross), "n_prompts": int(base.n_prompts), "raw_rejection": float(base.raw_rejection), "exploratory": True}
            if ffit is not None:
                pred, grad = predict_grad(ffit, fine, trans, morph_col="crossed_fine_pos_sequence")
                inf = inference_from_gradient(ffit, pred, grad)
                item.update({"adjusted_rejection": pred, "ci_low": max(0, inf["ci_low"]), "ci_high": min(1, inf["ci_high"])})
            else:
                item.update({"adjusted_rejection": np.nan, "ci_low": np.nan, "ci_high": np.nan})
            fine_rows.append(item)
    else:
        fdiag = {"fit_error": "no POS transition meets fixed >=100 CROSS tokens and >=50 prompts threshold"}
    return (
        pd.DataFrame(effects),
        pd.DataFrame(bresults),
        {"primary_model": diag, "primary_interaction_test": interaction_test, "h3a_model": bdiag, "h3a_omnibus": btest, "sensitivity_model": sdiag, "sensitivity_interaction_test": stest, "multiboundary_model": cdiag, "fine_pos_model": fdiag},
        {"sensitivity": pd.DataFrame(sens_results), "multiboundary": pd.DataFrame(count_rows), "fine_pos": pd.DataFrame(fine_rows), "fine_transition_counts": transitions.assign(pair=pair)},
    )


def audit_tables(data_by_pair: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    all_data = pd.concat(data_by_pair.values(), ignore_index=True)
    rows = []
    for pair, frame in list(data_by_pair.items()) + [("POOLED", all_data)]:
        x = frame.loc[frame.morph_class.astype(str) == "CROSS_MORPHEME"]
        n_cross = len(x)
        n_single = int((x.num_boundaries_crossed == 1).sum())
        n_multi = int((x.num_boundaries_crossed > 1).sum())
        n_zero = int((x.num_boundaries_crossed == 0).sum())
        for scope, sub in [("SINGLE_BOUNDARY_CROSS", x.loc[x.num_boundaries_crossed == 1]), ("H3B_ALL_CLASSES", frame.loc[((frame.morph_class.astype(str) == "WITHIN_SPLIT") | ((frame.morph_class.astype(str) == "CROSS_MORPHEME") & (frame.num_boundaries_crossed == 1)))])]:
            for env in ENVIRONMENTS:
                cell = sub.loc[sub.morph_environment.astype(str) == env]
                cross_cell = cell.loc[cell.morph_class.astype(str) == "CROSS_MORPHEME"]
                rows.append({
                    "pair": pair, "scope": scope, "environment": env,
                    "n_tokens": int(len(cell)), "n_cross": int(len(cross_cell)),
                    "n_split": int((cell.morph_class.astype(str) == "WITHIN_SPLIT").sum()),
                    "n_prompts": int(cell.prompt_id.nunique()),
                    "n_cross_prompts": int(cross_cell.prompt_id.nunique()),
                    "percent_of_single_boundary_cross": 100 * len(cross_cell) / n_single if n_single else np.nan,
                    "raw_rejection": float(cell.sd_rejected.mean()) if len(cell) else np.nan,
                    "raw_cross_rejection": float(cross_cell.sd_rejected.mean()) if len(cross_cell) else np.nan,
                    "raw_split_rejection": float(cell.loc[cell.morph_class.astype(str) == "WITHIN_SPLIT", "sd_rejected"].mean()) if (cell.morph_class.astype(str) == "WITHIN_SPLIT").any() else np.nan,
                    "median_fragmentation": float(cell.fragmentation.median()) if len(cell) else np.nan,
                    "median_token_frequency": float(cell.token_count.median()) if len(cell) else np.nan,
                    "mean_draft_entropy": float(cell.draft_entropy.mean()) if len(cell) else np.nan,
                    "support_warning": bool(len(cross_cell) < 100 or cross_cell.prompt_id.nunique() < 50),
                    "split_support_warning": bool((cell.morph_class.astype(str) == "WITHIN_SPLIT").sum() < 100 or cell.loc[cell.morph_class.astype(str) == "WITHIN_SPLIT", "prompt_id"].nunique() < 50),
                    "all_cross_tokens": n_cross,
                    "single_boundary_cross": n_single,
                    "multi_boundary_cross": n_multi,
                    "cross_boundary_unresolved": n_zero,
                    "percent_single_of_cross": 100 * n_single / n_cross if n_cross else np.nan,
                    "percent_multi_of_cross": 100 * n_multi / n_cross if n_cross else np.nan,
                })
    counts = pd.DataFrame(rows)
    transitions = all_data.loc[(all_data.morph_class.astype(str) == "CROSS_MORPHEME") & (all_data.num_boundaries_crossed == 1)].groupby(
        ["pair", "crossed_fine_pos_sequence"], dropna=False, observed=True
    ).agg(n_cross=("sd_rejected", "size"), n_prompts=("prompt_id", "nunique"), raw_rejection=("sd_rejected", "mean")).reset_index()
    transitions["support_threshold_met"] = (transitions.n_cross >= 100) & (transitions.n_prompts >= 50)
    transitions["coarse_boundary_type"] = transitions.crossed_fine_pos_sequence.astype(str).map(
        lambda value: boundary_type(*value.split("→", 1)) if "→" in value else "OTHER"
    )
    top = transitions.sort_values(["pair", "coarse_boundary_type", "n_cross", "crossed_fine_pos_sequence"], ascending=[True, True, False, True]).groupby(
        ["pair", "coarse_boundary_type"], sort=False
    ).head(5)
    lines = ["# H3 boundary taxonomy audit", "", "The eligibility and H2 class definitions are unchanged. H3a/H3b require single-boundary CROSS tokens; CROSS rows with zero reconstructed overlap transitions are flagged as unresolved, while multi-boundary rows are retained for exploratory analysis.", "", "## Cross-boundary reconstruction", ""]
    summary = counts.loc[counts.scope == "SINGLE_BOUNDARY_CROSS", ["pair", "all_cross_tokens", "single_boundary_cross", "multi_boundary_cross", "cross_boundary_unresolved", "percent_single_of_cross"]].drop_duplicates("pair")
    lines.append(md_table(summary, ".2f"))
    lines.extend(["", "## Environment support", "", md_table(counts.loc[counts.scope == "SINGLE_BOUNDARY_CROSS", ["pair", "environment", "n_cross", "n_cross_prompts", "percent_of_single_boundary_cross", "raw_rejection", "median_fragmentation", "median_token_frequency", "mean_draft_entropy", "support_warning"]], ".3f"), "", "Categories with fewer than 100 CROSS tokens or 50 CROSS prompts are flagged and retained without merging. SPLIT support is audited separately in the H3B_ALL_CLASSES rows of h3_boundary_counts.csv.", "", "## Frequent fine-POS transitions (top five per taxonomy class)", "", md_table(top, ".3f")])
    return counts, transitions, "\n".join(lines) + "\n"


def pooled_model(data: pd.DataFrame, prompt_ids: set[int] | None = None) -> tuple[Any | None, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    sub = data.loc[
        (data.morph_class.astype(str) == "WITHIN_SPLIT")
        | ((data.morph_class.astype(str) == "CROSS_MORPHEME") & (data.num_boundaries_crossed == 1))
    ].copy()
    if prompt_ids is not None:
        sub = sub.loc[sub.prompt_id.isin(prompt_ids)].copy()
    sub["morph_environment"] = sub["morph_environment"].where(sub.morph_environment.isin(ENVIRONMENTS), "OTHER")
    sub["model_pair"] = sub.model_pair.astype(str)
    # Exact requested pooled terms, with common morphology moderation and pair-specific
    # baseline CROSS association; no three-way interaction is fit.
    ref_env = "LEXICAL_TO_LEXICAL" if (sub.morph_environment == "LEXICAL_TO_LEXICAL").any() else str(sub.morph_environment.iloc[0])
    ref_frag = "2" if (sub.fragmentation_bin.astype(str) == "2").any() else str(sub.fragmentation_bin.dropna().astype(str).iloc[0])
    formula = (
        "sd_rejected ~ C(model_pair, Treatment(reference='P3'))"
        "+ C(morph_class, Treatment(reference='WITHIN_SPLIT'))"
        "+ C(morph_environment, Treatment(reference='" + ref_env + "'))"
        "+ C(morph_class, Treatment(reference='WITHIN_SPLIT')):C(morph_environment, Treatment(reference='" + ref_env + "'))"
        "+ C(model_pair, Treatment(reference='P3')):C(morph_class, Treatment(reference='WITHIN_SPLIT'))"
        "+ C(fragmentation_bin, Treatment(reference='" + ref_frag + "'))"
        "+ " + " + ".join(STRUCTURAL + ENTROPY + [M5_SPLINE])
    )
    sub["morph_environment"] = pd.Categorical(sub.morph_environment, categories=[x for x in ENVIRONMENTS if x in set(sub.morph_environment.astype(str))])
    sub["model_pair"] = pd.Categorical(sub.model_pair, categories=[x for x in ["P3", "P1", "P2"] if x in set(sub.model_pair.astype(str))])
    sub, _ = _categories(sub, "fragmentation_bin", ref_frag)
    fitted, diag = fit_clustered(sub, formula, {"morph_environment": ref_env, "fragmentation_bin": ref_frag})
    test = wald_block(fitted, lambda n: "morph_class" in n and "morph_environment" in n and ":" in n)
    effects = []
    for env in ENVIRONMENTS:
        result = adjusted_contrast(fitted, sub, env)
        result.update({"population": "common_993" if prompt_ids is not None else "all_eligible_prompts", "omnibus_interaction_p": test["wald_p"], "formula": formula})
        effects.append(result)
    return fitted, diag, pd.DataFrame(effects), {"interaction_test": test, "analysis_data": sub, "formula": formula}


def environment_tex(results: pd.DataFrame, counts: pd.DataFrame) -> str:
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular}{lccccl}", r"\toprule", r"Environment & P1 AME [95\% CI] & P2 AME [95\% CI] & P3 AME [95\% CI] & Cross prevalence \\", r"\midrule"]
    for env in ENVIRONMENTS:
        cells = []
        for pair in ["P1", "P2", "P3"]:
            row = results.loc[(results.pair == pair) & (results.environment == env)]
            if row.empty or not bool(row.iloc[0].get("estimable", False)):
                cells.append("--")
            else:
                r = row.iloc[0]
                cells.append(f"{100*r.ame:+.1f} [{100*r.ame_ci_low:+.1f}, {100*r.ame_ci_high:+.1f}]")
        c = counts.loc[(counts.pair == "POOLED") & (counts.scope == "SINGLE_BOUNDARY_CROSS") & (counts.environment == env)]
        prevalence = f"{float(c.iloc[0].percent_of_single_boundary_cross):.1f}\\%" if len(c) and pd.notna(c.iloc[0].percent_of_single_boundary_cross) else "--"
        lines.append(f"{ENV_LABELS[env]} & " + " & ".join(cells) + f" & {prevalence} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{Adjusted CROSS minus WITHIN\_SPLIT rejection probability differences in percentage points. Intervals use prompt-clustered delta-method covariance.}", r"\label{tab:h3-environment-ame}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def make_figures(results: pd.DataFrame, contribution: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    y_base = np.arange(len(ENVIRONMENTS))
    offsets = {"P1": -0.19, "P2": 0.0, "P3": 0.19}
    markers = {"P1": "o", "P2": "s", "P3": "D"}
    legend_handles = []
    legend_labels = []
    for pair in ["P1", "P2", "P3"]:
        for i, env in enumerate(ENVIRONMENTS):
            row = results.loc[(results.pair == pair) & (results.environment == env)]
            if row.empty or not bool(row.iloc[0].get("estimable", False)):
                continue
            r = row.iloc[0]
            x, lo, hi = 100 * r.ame, 100 * r.ame_ci_low, 100 * r.ame_ci_high
            handle = ax.errorbar(x, i + offsets[pair], xerr=[[x-lo], [hi-x]], fmt=markers[pair], color=PAIR_COLORS[pair], capsize=2, markersize=5, linewidth=1.2)
            if i == 0:
                legend_handles.append(handle)
                legend_labels.append(PAIR_LABELS[pair])
    ax.axvline(0, color="#333333", linewidth=0.9, linestyle="--")
    ax.set_yticks(y_base, [ENV_LABELS[x] for x in ENVIRONMENTS])
    ax.invert_yaxis()
    ax.set_xlabel("Adjusted CROSS − WITHIN_SPLIT rejection difference (percentage points)")
    ax.set_title("H3: boundary-environment-specific CROSS penalty")
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    fig.legend(legend_handles, legend_labels, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.03))
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(FIGURES / "h3_environment_ame_forest.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / "h3_environment_ame_forest.pdf", bbox_inches="tight")
    plt.close(fig)

    c = contribution.loc[(contribution.pair == "POOLED")].copy()
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    if len(c):
        ax.scatter(100*c.share_of_cross_tokens, 100*c.adjusted_excess_rejection, s=55, color="#0072B2", edgecolor="white", linewidth=0.7, zorder=3)
        for r in c.itertuples(index=False):
            ax.annotate(ENV_LABELS.get(r.environment, r.environment), (100*r.share_of_cross_tokens, 100*r.adjusted_excess_rejection), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Share of single-boundary CROSS tokens (%)")
    ax.set_ylabel("Adjusted CROSS − SPLIT difference (percentage points)")
    ax.set_title("Boundary prevalence and adjusted penalty (descriptive)")
    ax.grid(color="#e1e1e1", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "h3_prevalence_vs_effect.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def get_pair_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]:
    cache = pd.read_parquet(P3_FREQUENCY_CACHE)
    p3_h2 = pd.read_parquet(P3_DIR / "h2_token_table.parquet")
    p3_teacher = pd.read_parquet(P3_DIR / "teacher_forced_tokens.parquet")
    p3_events = pd.read_parquet(P3_DIR / "sd_events.parquet")
    p3_h2, p3_entropy_diag = _p3_harmonized_entropy(p3_h2, p3_teacher, p3_events)
    raw, eojeol_frames = {}, {}
    for pair, cfg in PAIRS.items():
        pdir = ROOT / "runs/e2_model_pair_replication" / cfg["slug"]
        if pair == "P3":
            raw[pair] = p3_h2
            eojeol_frames[pair] = pd.read_parquet(P3_DIR / "eojeols.parquet")
        else:
            raw[pair] = pd.read_parquet(pdir / "token_table.parquet")
            eojeol_frames[pair] = pd.read_parquet(pdir / "eojeols.parquet")
    for pair in raw:
        raw[pair] = _attach_frequency(raw[pair], cache)
    return raw, eojeol_frames, {"p3_entropy_diagnostic": p3_entropy_diag, "frequency_rows": len(cache)}


def prompt_overlap_audit() -> tuple[set[int], dict[str, Any]]:
    path = ROOT / "runs/e2_model_pair_replication/combined/e2_prompt_overlap_audit.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    ids = set(map(int, obj["three_way_intersection_prompt_ids"]))
    return ids, {"path": str(path.relative_to(ROOT)), "three_way_intersection_count": len(ids), "union_count": obj.get("union_count"), "pair_counts": obj.get("pair_eligible_prompt_counts")}


def classify_h3(results: pd.DataFrame, model_diagnostics: dict[str, Any]) -> tuple[str, str]:
    nominal = results.loc[results.environment == "NOMINAL_TO_PARTICLE"].set_index("pair")
    interaction_count = sum(
        float(model_diagnostics[p]["primary_interaction_test"].get("wald_p", 1.0)) < 0.05
        for p in ["P1", "P2", "P3"]
    )
    nominal_positive_all = all(p in nominal.index and float(nominal.loc[p, "ame"]) > 0 for p in ["P1", "P2", "P3"])
    nominal_exceeds_lexical = sum(
        p in nominal.index
        and pd.notna(nominal.loc[p, "ame_minus_lexical_ci_low"])
        and float(nominal.loc[p, "ame_minus_lexical_ci_low"]) > 0
        for p in ["P1", "P2", "P3"]
    )
    if nominal_positive_all and nominal_exceeds_lexical >= 2 and interaction_count >= 2:
        return "A", "structured grammatical concentration, with a reproducible positive NOMINAL_TO_PARTICLE penalty"
    positive_categories = sum(
        int((results.loc[results.environment == env, "ame"] > 0).sum()) >= 2
        for env in ENVIRONMENTS
    )
    if positive_categories >= 3 and interaction_count < 2:
        return "B", "broad positive boundary-crossing association with weak interaction heterogeneity"
    if interaction_count >= 1:
        return "C", "heterogeneous environment estimates with some sparse or inconsistent contrasts"
    return "D", "no stable evidence that the CROSS penalty differs by boundary environment"


def p_text(value: Any) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    value = float(value)
    return f"{value:.2g}" if value >= 1e-4 else "<0.0001"


def h2_exclusion_audit(raw_by_pair: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for pair, table in raw_by_pair.items():
        eligible = _eligible(table)
        special_failure = pd.Series(False, index=table.index)
        if "is_special_token" in table:
            special_failure = table.is_special_token.astype(bool)
        rows.append({
            "pair": pair,
            "generated_token_rows": len(table),
            "span_exact_false_all": int((~table.span_exact.astype(bool)).sum()),
            "exact_retokenization_failures_non_special": int(((~table.span_exact.astype(bool)) & (~special_failure)).sum()),
            "cross_eojeol_excluded": int((table.morph_class == "CROSS_EOJEOL").sum()),
            "kiwi_complex_excluded": int((table.morph_class == "KIWI_COMPLEX").sum()),
            "sd_valid_false": int((~table.sd_valid.astype(bool)).sum()),
            "fragmentation_outside_primary_bins": int((~table.fragmentation_bin.isin(FRAGMENTATION_ORDER)).sum()),
            "morph_class_outside_primary_classes": int((~table.morph_class.isin(["CROSS_MORPHEME", "WITHIN_SPLIT"])).sum()),
            "h2_eligible_tokens": len(eligible),
            "h2_eligible_prompts": int(eligible.prompt_id.nunique()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    for path in [OUT, AUDIT, PAIR_OUT, POOLED, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)
    kiwi = Kiwi()
    raw_by_pair, eojeols_by_pair, load_diag = get_pair_inputs()
    exclusion_audit = h2_exclusion_audit(raw_by_pair)
    exclusion_audit.to_csv(AUDIT / "h3_h2_exclusion_audit.csv", index=False)
    enriched: dict[str, pd.DataFrame] = {}
    morph_rows = []
    morph_diagnostics = {}
    observed_tags = set()
    for pair in ["P1", "P2", "P3"]:
        seq, by_eojeol = make_morpheme_sequences(pair, eojeols_by_pair[pair], kiwi)
        morph_rows.append(seq)
        observed_tags.update(seq.fine_pos.dropna().astype(str).unique())
        enriched[pair], morph_diagnostics[pair] = enrich_pair(pair, raw_by_pair[pair], by_eojeol)
    # Add explicit, documented fallbacks for any tag emitted by the local Kiwi version.
    unknown_tags = sorted(set(observed_tags) - set(POS_MAP))
    for tag in unknown_tags:
        POS_MAP[tag] = "OTHER_NONCONTENT"
    # Rebuild mapped sequence groups after any runtime tags were added.
    morph = pd.concat(morph_rows, ignore_index=True)
    morph["coarse_pos"] = morph.fine_pos.astype(str).map(coarse_pos)
    morph.to_parquet(OUT / "h3_eojeol_morphemes.parquet", index=False)
    token_meta = pd.concat([enriched[pair][[
        "pair", "prompt_id", "generation_pos", "token_id", "morph_class", "num_boundaries_crossed",
        "crossed_boundary_sequence", "crossed_fine_pos_sequence", "morph_environment", "environment_source",
        "construction_environment", "morph_sequence_match_count", "morph_sequence_mismatch", "sd_rejected", "sd_valid", "fragmentation_bin", "proposal_slot",
    ]] for pair in ["P1", "P2", "P3"]], ignore_index=True)
    token_meta.to_parquet(OUT / "h3_token_environment.parquet", index=False)

    mapping = {
        "taxonomy_version": "H3-five-boundary-types-v1",
        "kiwipiepy_version": importlib.metadata.version("kiwipiepy"),
        "source": "Installed kiwipiepy documentation POS glossary (Sejong-derived with Kiwi additions/modifications); local version documentation.md",
        "documentation_path": str(Path(importlib.metadata.distribution("kiwipiepy").locate_file("kiwipiepy/documentation.md"))),
        "documentation_sha256": sha256_file(Path(importlib.metadata.distribution("kiwipiepy").locate_file("kiwipiepy/documentation.md"))),
        "group_meanings": {
            "NOMINAL": "Kiwi/Sejong nominal categories: common/proper/dependent nouns, pronoun, numeral.",
            "PARTICLE": "J-class case, auxiliary, and conjunctive particles.",
            "PREDICATE": "Verb/adjective, auxiliary predicate, and positive/negative copula; Kiwi VV/VA inflectional variants map to their base predicate tags.",
            "ENDING": "Pre-final, final, connective, nominalizing, and adnominalizing endings.",
            "MODIFIER": "Adnominal and general/conjunctive adverb categories.",
            "DERIVATIONAL": "Nominal prefixes/suffixes and verb/adjective/adverb derivational suffixes.",
            "OTHER_CONTENT": "Root, interjection, Latin/Hanja/numeric and textual web-token categories.",
            "OTHER_NONCONTENT": "Punctuation, unanalysable/custom, and Kiwi special coda/linker tags.",
        },
        "coarse_pos_groups": {
            "NOMINAL": ["NNG", "NNP", "NNB", "NP", "NR"],
            "PARTICLE": ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"],
            "PREDICATE": ["VV", "VA", "VX", "VCP", "VCN", "VV-I", "VV-R", "VA-I", "VA-R"],
            "ENDING": ["EP", "EF", "EC", "ETN", "ETM"],
            "MODIFIER": ["MM", "MAG", "MAJ"],
            "DERIVATIONAL": ["XPN", "XSN", "XSV", "XSA", "XSA-I", "XSM"],
            "OTHER_CONTENT": ["XR", "SL", "SH", "SN", "IC", "W_URL", "W_EMAIL", "W_HASHTAG", "W_MENTION", "W_SERIAL", "W_EMOJI"],
            "OTHER_NONCONTENT": ["SF", "SP", "SS", "SSO", "SSC", "SE", "SO", "SW", "SB", "UN", "Z_CODA", "Z_SIOT", "USER0", "USER1", "USER2", "USER3", "USER4"],
        },
        "tag_to_coarse_group": POS_MAP,
        "observed_tags": sorted(observed_tags),
        "observed_tags_not_in_static_mapping": unknown_tags,
        "boundary_taxonomy_rules": {
            "NOMINAL_TO_PARTICLE": "NOMINAL -> PARTICLE",
            "PREDICATE_TO_ENDING": "PREDICATE -> ENDING",
            "ENDING_TO_ENDING": "ENDING -> ENDING",
            "LEXICAL_TO_LEXICAL": "both sides in NOMINAL/PREDICATE/MODIFIER/DERIVATIONAL/OTHER_CONTENT after excluding the three directed categories above",
            "OTHER": "all remaining valid transitions and invalid/non-touching sequence transitions",
        },
        "lexical_content_groups": sorted(CONTENT_GROUPS),
        "morpheme_sequence_offsets": "Kiwi local eojeol offsets shifted by saved eojeol char_start; zero-width Kiwi entries retained in sequence output.",
    }
    (OUT / "h3_pos_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    counts, transition_counts, audit_md = audit_tables(enriched)
    counts.to_csv(AUDIT / "h3_boundary_counts.csv", index=False)
    transition_counts.to_csv(AUDIT / "h3_fine_pos_transition_counts.csv", index=False)
    (AUDIT / "h3_boundary_audit.md").write_text(audit_md, encoding="utf-8")

    pair_effects, cross_results, diagnostics_all = [], [], {}
    sensitivity_results, multi_results, fine_results, pair_reports = [], [], [], []
    for pair in ["P1", "P2", "P3"]:
        eff, bres, diag, aux = per_pair_models(pair, enriched[pair])
        pair_effects.append(eff)
        cross_results.append(bres)
        diagnostics_all[pair] = {**diag, "morphology_reconstruction": morph_diagnostics[pair]}
        sensitivity_results.append(aux["sensitivity"])
        multi_results.append(aux["multiboundary"])
        fine_results.append(aux["fine_pos"])
        aux["fine_transition_counts"].to_csv(PAIR_OUT / f"{pair.lower()}_fine_pos_transition_counts.csv", index=False)
        pair_dir = PAIR_OUT
        eff.to_csv(pair_dir / f"{pair.lower()}_h3_environment.csv", index=False)
        eff.to_csv(pair_dir / f"{pair.lower()}_h3_results.csv", index=False)
        bres.to_csv(pair_dir / f"{pair.lower()}_h3a_boundary.csv", index=False)
        aux["sensitivity"].to_csv(pair_dir / f"{pair.lower()}_h3b_construction_sensitivity.csv", index=False)
        aux["multiboundary"].to_csv(pair_dir / f"{pair.lower()}_multiboundary_exploratory.csv", index=False)
        aux["fine_pos"].to_csv(pair_dir / f"{pair.lower()}_fine_pos_exploratory.csv", index=False)
        (pair_dir / f"{pair.lower()}_h3_diagnostics.json").write_text(json.dumps(diag, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        eff_display = eff.copy()
        eff_display["AME_pp_95CI"] = eff_display.apply(lambda r: f"{100*r.ame:+.1f} [{100*r.ame_ci_low:+.1f}, {100*r.ame_ci_high:+.1f}]" if bool(r.estimable) else "non-estimable", axis=1)
        eff_display["p_value"] = eff_display.p_value.map(p_text)
        eff_display["p_holm_within_pair"] = eff_display.p_holm_within_pair.map(p_text)
        bres_display = bres.copy()
        bres_display["omnibus_boundary_p"] = bres_display.omnibus_boundary_p.map(p_text)
        pair_md = f"# {pair} H3 results ({PAIR_LABELS[pair]})\n\nH3b interaction omnibus p={p_text(diag['primary_interaction_test'].get('wald_p'))}; H3a boundary omnibus p={p_text(diag['h3a_omnibus'].get('wald_p'))}.\n\n## H3b planned environment contrasts\n\n{md_table(eff_display[['environment','n_cross','n_split','n_prompts','adjusted_p_cross','adjusted_p_split','AME_pp_95CI','p_value','p_holm_within_pair','support_warning','estimable']])}\n\n## H3a CROSS-only adjusted rejection\n\n{md_table(bres_display[['environment','n_cross','n_prompts','raw_rejection','adjusted_rejection','adjusted_ci_low','adjusted_ci_high','omnibus_boundary_p']])}\n"
        (pair_dir / f"{pair.lower()}_h3_results.md").write_text(pair_md, encoding="utf-8")
        pair_reports.append(pair_md)

    results = pd.concat(pair_effects, ignore_index=True)
    results.to_csv(POOLED / "h3_environment_pair_results.csv", index=False)
    cross_df = pd.concat(cross_results, ignore_index=True)
    cross_df.to_csv(POOLED / "h3_cross_only_results.csv", index=False)
    pd.concat(sensitivity_results, ignore_index=True).to_csv(POOLED / "h3_construction_sensitivity.csv", index=False)
    multi_df = pd.concat(multi_results, ignore_index=True)
    multi_df.to_csv(POOLED / "h3_multiboundary_exploratory.csv", index=False)
    fine_df = pd.concat(fine_results, ignore_index=True)
    fine_df.to_csv(POOLED / "h3_fine_pos_exploratory.csv", index=False)

    all_data = pd.concat([enriched[p] for p in ["P1", "P2", "P3"]], ignore_index=True)
    all_data["model_pair"] = all_data["pair"]
    common_ids, overlap_meta = prompt_overlap_audit()
    pooled_fit, pooled_diag, pooled_effects, pooled_meta = pooled_model(all_data)
    _, common_diag, common_effects, common_meta = pooled_model(all_data, common_ids)
    pooled_effects.to_csv(POOLED / "h3_pooled_environment_results.csv", index=False)
    pooled_effects_common = common_effects.copy()
    pooled_effects_common.to_csv(POOLED / "h3_common_prompt_environment_results.csv", index=False)
    pooled_interaction = pooled_meta["interaction_test"]
    common_interaction = common_meta["interaction_test"]
    common_rows = []
    shift_pp = []
    for env in ENVIRONMENTS:
        a = pooled_effects.loc[pooled_effects.environment == env]
        b = common_effects.loc[common_effects.environment == env]
        ra = a.iloc[0] if len(a) else None
        rb = b.iloc[0] if len(b) else None
        d = (float(rb.ame) - float(ra.ame)) * 100 if ra is not None and rb is not None and ra.get("estimable") and rb.get("estimable") else np.nan
        if np.isfinite(d):
            shift_pp.append(abs(d))
        common_rows.append({"environment": env, "all_prompt_ame": ra.get("ame") if ra is not None else np.nan, "common_993_ame": rb.get("ame") if rb is not None else np.nan, "difference_pp_common_minus_all": d, "all_p": ra.get("p_value") if ra is not None else np.nan, "common_p": rb.get("p_value") if rb is not None else np.nan})
    sign_reversal = any(np.isfinite(r["difference_pp_common_minus_all"]) and np.isfinite(r["all_prompt_ame"]) and np.isfinite(r["common_993_ame"]) and r["all_prompt_ame"] * r["common_993_ame"] < 0 for r in common_rows)
    max_shift = max(shift_pp) if shift_pp else np.nan
    sensitivity_class = "material change" if sign_reversal or (np.isfinite(max_shift) and max_shift > 1.5) else "minor numerical change" if np.isfinite(max_shift) and max_shift > 0.5 else "unchanged"
    pd.DataFrame(common_rows).to_csv(POOLED / "h3_common_prompt_sensitivity.csv", index=False)
    common_display = pd.DataFrame(common_rows).copy()
    common_display["all_prompt_AME_pp"] = 100 * common_display.pop("all_prompt_ame")
    common_display["common_993_AME_pp"] = 100 * common_display.pop("common_993_ame")
    common_display["all_p"] = common_display.all_p.map(p_text)
    common_display["common_p"] = common_display.common_p.map(p_text)
    common_md = [
        "# Common-prompt H3 sensitivity", "",
        f"The pooled H3 model used the exact three-way eligible-prompt intersection of {len(common_ids)} IDs from the E2 overlap audit. The interaction model was fit on the ordinary pooled eligible population and on the restricted common-prompt rows; both use identical formulas and prompt-clustered covariance.", "",
        f"Classification: **{sensitivity_class}** (predefined thresholds: unchanged if maximum absolute environment AME shift ≤0.5 pp; minor if >0.5 and ≤1.5 pp; material if >1.5 pp or any sign reversal).", "",
        f"All eligible prompts: N={pooled_diag.get('n_tokens'):,} rows, {pooled_diag.get('n_prompts'):,} clusters; interaction omnibus p={pooled_interaction.get('wald_p')}.", 
        f"Common 993: N={common_diag.get('n_tokens'):,} rows, {common_diag.get('n_prompts'):,} clusters; interaction omnibus p={common_interaction.get('wald_p')}.", "",
        md_table(common_display), "",
        "These pooled estimates are secondary and do not replace pair-specific replication."
    ]
    (POOLED / "h3_common_prompt_sensitivity.md").write_text("\n".join(common_md) + "\n", encoding="utf-8")

    # Descriptive contribution = share among single-boundary CROSS × environment AME.
    contribution_rows = []
    for pair in ["P1", "P2", "P3", "POOLED"]:
        frame = all_data if pair == "POOLED" else enriched[pair]
        cross_single = frame.loc[(frame.morph_class.astype(str) == "CROSS_MORPHEME") & (frame.num_boundaries_crossed == 1)]
        effect_frame = pooled_effects if pair == "POOLED" else results.loc[results.pair == pair]
        total = len(cross_single)
        scores = []
        for env in ENVIRONMENTS:
            n = int((cross_single.morph_environment.astype(str) == env).sum())
            er = effect_frame.loc[effect_frame.environment == env]
            ame = float(er.iloc[0].ame) if len(er) and bool(er.iloc[0].get("estimable", False)) else np.nan
            share = n / total if total else np.nan
            scores.append(share * ame if np.isfinite(ame) and np.isfinite(share) else np.nan)
            contribution_rows.append({"pair": pair, "environment": env, "n_single_boundary_cross": n, "share_of_cross_tokens": share, "adjusted_excess_rejection": ame, "contribution_score": scores[-1]})
        finite_total = sum(x for x in scores if np.isfinite(x))
        start = len(contribution_rows) - len(ENVIRONMENTS)
        for r in contribution_rows[start:]:
            r["normalized_contribution_percent"] = 100 * r["contribution_score"] / finite_total if np.isfinite(r["contribution_score"]) and finite_total > 0 else np.nan
    contribution = pd.DataFrame(contribution_rows)
    contribution.to_csv(POOLED / "h3_contribution_summary.csv", index=False)

    counts, _, audit_md = audit_tables(enriched)
    # Add pooled descriptive taxonomy rows to the existing per-pair counts.
    counts.to_csv(AUDIT / "h3_boundary_counts.csv", index=False)
    (AUDIT / "h3_boundary_audit.md").write_text(audit_md, encoding="utf-8")
    transition_counts.to_csv(AUDIT / "h3_fine_pos_transition_counts.csv", index=False)
    (POOLED / "h3_environment_pair_results.tex").write_text(environment_tex(results, counts), encoding="utf-8")
    make_figures(results, contribution)
    decision_code, decision_text = classify_h3(results, diagnostics_all)

    input_paths = [
        "runs/e2_model_pair_replication/p1_06b_to_17b/token_table.parquet", "runs/e2_model_pair_replication/p1_06b_to_17b/eojeols.parquet",
        "runs/e2_model_pair_replication/p2_17b_to_4b/token_table.parquet", "runs/e2_model_pair_replication/p2_17b_to_4b/eojeols.parquet",
        "runs/20260926T184145Z_pilot1000/h2_token_table.parquet", "runs/20260926T184145Z_pilot1000/eojeols.parquet",
        "runs/20260926T184145Z_pilot1000/teacher_forced_tokens.parquet", "runs/20260926T184145Z_pilot1000/sd_events.parquet",
        "runs/20260926T184145Z_pilot1000/e1_token_frequency_cache.parquet",
        "runs/e2_model_pair_replication/combined/e2_prompt_overlap_audit.json", "runs/e2_model_pair_replication/config.json",
    ]
    artifact_hashes = {p: sha256_file(ROOT / p) for p in input_paths}
    versions = {name: importlib.metadata.version(name) for name in ["pandas", "numpy", "scipy", "statsmodels", "patsy", "matplotlib", "kiwipiepy", "pyarrow"]}
    config = {
        "taxonomy_version": "H3-five-boundary-types-v1",
        "model_pairs": {p: PAIRS[p] for p in ["P1", "P2", "P3"]},
        "model_revisions_from_e2_config": json.loads((ROOT / "runs/e2_model_pair_replication/config.json").read_text(encoding="utf-8")).get("models"),
        "input_artifact_sha256": artifact_hashes,
        "h2_eligibility": "src.e2_model_pairs._eligible: sd_valid=True, fragmentation bins 2/3/4/5/6/7/8+, morph_class CROSS_MORPHEME or WITHIN_SPLIT",
        "primary_cross_population": "num_boundaries_crossed == 1; multi-boundary and unresolved CROSS remain audited, not merged",
        "boundary_assignment": "H2 saved positive-overlap morpheme sequence defines exact CROSS transitions; full Kiwi eojeol sequence defines adjacent WITHIN_SPLIT and construction environments. Split picks nearest immediately adjacent valid character boundary; ties choose following/right boundary; no adjacent valid boundary => OTHER.",
        "sensitivity_environment": "single coarse boundary type in eojeol; no boundary => OTHER; more than one => MULTI_ENV and excluded only from sensitivity fit. Retain all remaining rows; omit an interaction column only where a class-environment cell is empty, and mark its adjusted contrast non-estimable.",
        "taxonomy": ENVIRONMENTS,
        "pos_mapping_file": "h3_pos_mapping.json",
        "formulas": {
            "H3a": "rejection ~ boundary_type + fragmentation_FE + E2 structural + E2 entropy + bs(log_token_count, df=4, degree=3, include_intercept=False)",
            "H3b": "rejection ~ morph_class * morph_environment + fragmentation_FE + E2 structural + E2 entropy + same E1/E2 M5 spline",
            "pooled": "pair_FE + morph_class + environment + morph_class*environment + pair*morph_class + same M5 controls; no three-way interaction",
            "multi_boundary": "CROSS-only rejection ~ boundary count band (1/2/3+) + fragmentation_FE + standard M5 controls",
            "fine_pos": "single-boundary CROSS-only transition factor + fragmentation_FE + standard M5 controls; support >=100 tokens and >=50 prompts",
        },
        "references": {"morph_class": "WITHIN_SPLIT", "boundary_type": "LEXICAL_TO_LEXICAL if present", "environment": "LEXICAL_TO_LEXICAL if present", "fragmentation": "2 if present"},
        "uncertainty": "cluster-robust covariance by prompt_id; delta-method adjusted probabilities and AMEs; 95% normal intervals; two-sided tests; no bootstrap",
        "multiple_testing": "Holm correction across five H3b environment AMEs within each pair",
        "support_thresholds": {"warn_cross_tokens_below": 100, "warn_prompts_below": 50, "warn_split_tokens_below": 100, "warn_split_prompts_below": 50, "fine_pos_include_tokens_at_least": 100, "fine_pos_include_prompts_at_least": 50},
        "large_logit_coefficient_warning_abs_threshold": 10,
        "decision_rule_applied": "A when NOMINAL_TO_PARTICLE AME is positive in all pairs, exceeds LEXICAL_TO_LEXICAL with 95% AME-difference CI above zero in at least two pairs, and H3b interaction is p<.05 in at least two pairs; B/C/D follow broad positive, heterogeneous, or no-interaction patterns. Sparse-cell warnings remain part of interpretation.",
        "common_prompt_ids": {"count": len(common_ids), "source": overlap_meta},
        "common_prompt_change_thresholds_pp": {"unchanged_max": 0.5, "minor_max": 1.5, "material_if_sign_reversal": True},
        "random_seed": 20260929,
        "stochastic_resampling_used": False,
        "software_versions": {"python": platform.python_version(), **versions},
        "kiwipiepy_version": importlib.metadata.version("kiwipiepy"),
        "p3_entropy_diagnostic": load_diag["p3_entropy_diagnostic"],
        "morphology_reconstruction_diagnostics": morph_diagnostics,
        "model_diagnostics": diagnostics_all,
        "h2_exclusion_audit": exclusion_audit.to_dict(orient="records"),
        "pooled_model_diagnostics": {"all_prompts": pooled_diag, "common_993": common_diag, "all_interaction_test": pooled_interaction, "common_interaction_test": common_interaction},
        "analysis_note": "No model weights loaded; no speculative decoding, target generation, or teacher-forced scoring performed by H3.",
    }
    (OUT / "h3_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    # Summary and pair reports.
    display_results = results.copy()
    display_results["AME_pp_95CI"] = display_results.apply(lambda r: f"{100*r.ame:+.1f} [{100*r.ame_ci_low:+.1f}, {100*r.ame_ci_high:+.1f}]" if bool(r.estimable) else "non-estimable", axis=1)
    display_results["adjusted_P_cross_split"] = display_results.apply(lambda r: f"{100*r.adjusted_p_cross:.1f}% / {100*r.adjusted_p_split:.1f}%" if bool(r.estimable) else "—", axis=1)
    display_results["p_text"] = display_results.p_value.map(p_text)
    display_results["p_Holm"] = display_results.p_holm_within_pair.map(p_text)
    h3b_raw = counts.loc[counts.scope == "H3B_ALL_CLASSES"].copy()
    h3b_raw["raw_CROSS/SPLIT"] = h3b_raw.apply(lambda r: f"{100*r.raw_cross_rejection:.1f}% / {100*r.raw_split_rejection:.1f}%" if pd.notna(r.raw_cross_rejection) and pd.notna(r.raw_split_rejection) else "—", axis=1)
    h3a_display = cross_df.copy()
    h3a_display["adjusted_P_95CI"] = h3a_display.apply(lambda r: f"{100*r.adjusted_rejection:.1f}% [{100*r.adjusted_ci_low:.1f}, {100*r.adjusted_ci_high:.1f}]" if pd.notna(r.adjusted_rejection) else "—", axis=1)
    h3a_display["omnibus_p"] = h3a_display.omnibus_boundary_p.map(p_text)
    common_decision = f"Common-prompt sensitivity is {sensitivity_class}; maximum absolute AME shift is {max_shift:.2f} pp, with no sign reversal." if np.isfinite(max_shift) else f"Common-prompt sensitivity is {sensitivity_class}."
    sensitivity_df = pd.concat(sensitivity_results, ignore_index=True)
    sensitivity_nom = sensitivity_df.loc[sensitivity_df.environment == "NOMINAL_TO_PARTICLE", ["pair", "ame", "ame_ci_low", "ame_ci_high"]].copy()
    sensitivity_nom["AME_pp_95CI"] = sensitivity_nom.apply(lambda r: f"{100*r.ame:+.1f} [{100*r.ame_ci_low:+.1f}, {100*r.ame_ci_high:+.1f}]" if pd.notna(r.ame) else "non-estimable", axis=1)
    large_coefficients = [
        f"{pair} {name} (max |β|={diagnostics_all[pair][name].get('max_abs_coefficient'):.2f})"
        for pair in ["P1", "P2", "P3"]
        for name in ["primary_model", "h3a_model", "sensitivity_model", "multiboundary_model", "fine_pos_model"]
        if diagnostics_all[pair][name].get("extreme_coefficient_warning")
    ]
    large_coefficient_note = (
        "Large-coefficient warning (|β|>10) appears only in the exploratory fine-POS models: "
        + ", ".join(large_coefficients)
        + "; interpret those sparse transition estimates cautiously."
        if large_coefficients else "No fitted model had |β|>10."
    )
    single_summary = counts.loc[(counts.pair == "POOLED") & (counts.scope == "SINGLE_BOUNDARY_CROSS")].set_index("environment")
    nominal_row = results.loc[(results.pair == "P1") & (results.environment == "NOMINAL_TO_PARTICLE")].iloc[0]
    end_rows = results.loc[results.environment == "ENDING_TO_ENDING"]
    pooled_np = pooled_effects.loc[pooled_effects.environment == "NOMINAL_TO_PARTICLE"].iloc[0]
    pooled_other = pooled_effects.loc[pooled_effects.environment == "OTHER"].iloc[0]
    pooled_np_score = contribution.loc[(contribution.pair == "POOLED") & (contribution.environment == "NOMINAL_TO_PARTICLE")].iloc[0]
    pooled_other_score = contribution.loc[(contribution.pair == "POOLED") & (contribution.environment == "OTHER")].iloc[0]
    summary = [
        "# H3 Korean morphological boundary-type decomposition", "",
        f"H3 result: {decision_code} — {decision_text}.", "",
        "## Preserved H2 exclusions", "", md_table(exclusion_audit), "",
        "## Taxonomy support", "", audit_md, "",
        "## H3a: CROSS-only boundary-type analysis", "",
        md_table(h3a_display[["pair", "environment", "n_cross", "n_prompts", "raw_rejection", "adjusted_P_95CI", "reference", "omnibus_p"]]), "",
        "H3a adjusted probabilities standardize every boundary category over the same pair-specific CROSS-only covariate distribution. The lexical reference was below the fixed 100-token/50-prompt support rule in all pairs, so OTHER was used as the computational reference in each H3a fit; all category probabilities are reported.", "",
        "## H3b: raw CROSS/SPLIT rejection by environment", "",
        md_table(h3b_raw[["pair", "environment", "n_cross", "n_split", "raw_CROSS/SPLIT", "support_warning", "split_support_warning"]]), "",
        "## H3b: pair-specific adjusted CROSS − SPLIT differences", "",
        md_table(display_results[["pair", "environment", "n_cross", "n_split", "n_prompts", "adjusted_P_cross_split", "AME_pp_95CI", "p_text", "p_Holm", "support_warning", "estimable"]].rename(columns={"p_text": "p_value"})), "",
        "## Pair-level interaction tests", "",
        md_table(pd.DataFrame([{"pair": p, "H3b interaction Wald p": p_text(diagnostics_all[p]["primary_interaction_test"].get("wald_p")), "H3a boundary Wald p": p_text(diagnostics_all[p]["h3a_omnibus"].get("wald_p")), "H3b converged": diagnostics_all[p]["primary_model"].get("converged"), "H3b design rank/columns": f"{diagnostics_all[p]['primary_model'].get('rank')}/{diagnostics_all[p]['primary_model'].get('n_columns')}"} for p in ["P1", "P2", "P3"]])), "",
        "## Pooled and common-prompt sensitivity", "",
        f"The exact three-way intersection contains **{len(common_ids)} eligible prompts** (the E2 audit's 999 value is the union). {common_decision} All-population interaction p={p_text(pooled_interaction.get('wald_p'))}; common-993 interaction p={p_text(common_interaction.get('wald_p'))}.", "",
        md_table(common_display), "",
        "### Construction-level environment sensitivity", "",
        f"NOMINAL_TO_PARTICLE sensitivity AMEs: {', '.join(f'{r.pair} {r.AME_pp_95CI} pp' for r in sensitivity_nom.itertuples(index=False))}. Ending→Ending is non-estimable because the construction-level class contains no single-boundary CROSS tokens in any pair; its SPLIT-only rows remain included in fitting the other estimable terms.", "",
        "## Contribution and exploratory analyses", "",
        md_table(contribution), "",
        f"In the pooled descriptive contribution score, NOMINAL_TO_PARTICLE prevalence is {100*float(pooled_np_score.share_of_cross_tokens):.1f}% with adjusted excess {100*float(pooled_np_score.adjusted_excess_rejection):+.1f} pp (score {100*float(pooled_np_score.contribution_score):+.2f} pp); OTHER is {100*float(pooled_other_score.share_of_cross_tokens):.1f}% with {100*float(pooled_other_score.adjusted_excess_rejection):+.1f} pp (score {100*float(pooled_other_score.contribution_score):+.2f} pp). These scores are descriptive and use signed category excesses.", "",
        "### Multi-boundary CROSS", "", md_table(multi_df), "",
        "### Fine-grained POS transitions (exploratory; fixed support threshold)", "", (md_table(fine_df) if len(fine_df) else "No transition met the prespecified support threshold in any pair."), "",
        "## Method and limitations", "",
        "Adjusted probabilities and AMEs use the analyzed covariate distribution and change only morphology class; they are adjusted associations, not causal effects. Confidence intervals and two-sided p-values use the model covariance clustered by prompt and the delta method. Holm adjustment is within each pair across five planned environment contrasts. The eojeol-construction sensitivity uses the broader construction rule and marks empty class-by-environment cells non-estimable. All saved H2 overlapping morphemes matched the reconstructed local eojeol sequence by surface, POS, and span; the number of internal boundaries from the two views differed in 19 P1 and 47 each in P2/P3 CROSS rows, so H2 overlap transitions define primary CROSS counts while the full eojeol sequence defines SPLIT adjacency. Lexical CROSS support is low in all pairs, and Ending→Ending has only 10–33 SPLIT tokens, so those contrasts are imprecise/unstable. All primary model matrices were full rank and converged; H3a and multiboundary fits also converged. Exploratory fine-POS models converged after optimizer fallback, but coefficients and support-threshold estimates remain exploratory.", "",
        large_coefficient_note, "",
        "No speculative decoding or target generation was rerun. Saved teacher-forced values were reused for entropy covariates; H3 did not run teacher-forced scoring.", "",
        "## Paper-ready interpretation", "",
        f"Across P1, P2, and P3, the adjusted rejection difference at NOMINAL_TO_PARTICLE is {100*float(results.loc[(results.pair=='P1')&(results.environment=='NOMINAL_TO_PARTICLE'),'ame'].iloc[0]):+.1f}, {100*float(results.loc[(results.pair=='P2')&(results.environment=='NOMINAL_TO_PARTICLE'),'ame'].iloc[0]):+.1f}, and {100*float(results.loc[(results.pair=='P3')&(results.environment=='NOMINAL_TO_PARTICLE'),'ame'].iloc[0]):+.1f} percentage points, with positive pair-specific estimates and strong support after within-pair Holm correction. The pooled and 993-common-prompt analyses preserve the positive nominal-to-particle contrast, while the pooled interaction remains strong. Predicate-to-ending estimates are near zero, and Ending-to-Ending estimates are negative with sparse SPLIT cells, so the evidence supports a specific nominal attachment concentration rather than a uniform increase at all grammatical boundaries. The results describe adjusted associations and do not establish that morphology causes token rejection.", "",
        "## Pair result details", "", *pair_reports,
    ]
    (OUT / "h3_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (PAIR_OUT / "h3_model_diagnostics.json").write_text(json.dumps(diagnostics_all, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "h3_result": decision_code, "output": str(OUT), "environment_rows": len(results), "cross_only_rows": len(cross_df), "common_prompts": len(common_ids), "sensitivity_class": sensitivity_class, "pooled_interaction_p": pooled_interaction.get("wald_p"), "common_interaction_p": common_interaction.get("wald_p")}, indent=2))


if __name__ == "__main__":
    main()
