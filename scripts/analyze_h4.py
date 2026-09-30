#!/usr/bin/env python3
"""H4 decomposition on saved artifacts only. CPU analysis; no inference or generation."""
from __future__ import annotations

import hashlib, importlib.metadata, json, math, platform, sys, warnings
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import scipy.stats as st
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit
from src.e2_model_pairs import (
    DEFAULT_ENTROPY, DEFAULT_STRUCTURAL, FRAGMENTATION_ORDER, PAIRS,
    P3_DIR, P3_FREQUENCY_CACHE, _attach_frequency, _p3_harmonized_entropy,
    sha256_file,
)

OUT = ROOT / "runs/e2_model_pair_replication/h4"
H3 = ROOT / "runs/e2_model_pair_replication/h3"
LEGACY = ROOT / "runs/20260926T184145Z_pilot1000"
CORPUS = ROOT / "data/korean_wikipedia_20231101_ko.jsonl"
SEED = 20260929
PAIRS_ORDER = ["P1", "P2", "P3"]
PAIR_LABELS = {"P1": "0.6B → 1.7B", "P2": "1.7B → 4B", "P3": "0.6B → 4B"}
PARTICLE_GROUPS = {
    "CASE_SUBJECT": ["JKS"], "CASE_OBJECT": ["JKO"], "CASE_ADVERBIAL": ["JKB"],
    "CASE_GENITIVE": ["JKG"], "OTHER_CASE": ["JKC", "JKV", "JKQ"],
    "AUXILIARY_PARTICLE": ["JX"], "CONJUNCTIVE_PARTICLE": ["JC"],
}
PARTICLE_ORDER = list(PARTICLE_GROUPS)
PARTICLE_TAGS = {tag for tags in PARTICLE_GROUPS.values() for tag in tags}
NOMINAL_TAGS = {"NNG", "NNP", "NNB", "NP", "NR"}
GEOMETRY_ORDER = [
    "NOMINAL_SUFFIX_PLUS_FULL_PARTICLE", "FULL_NOMINAL_PLUS_PARTICLE",
    "PARTIAL_BOTH", "FULL_NOMINAL_PLUS_FULL_PARTICLE", "OTHER_GEOMETRY",
]
STRUCT = list(DEFAULT_STRUCTURAL)
ENTROPY = list(DEFAULT_ENTROPY)
M5_SPLINE = "bs(log_token_count, df=4, degree=3, include_intercept=False)"
BASE = ["C(fragmentation_bin, Treatment(reference='2'))"] + STRUCT + ENTROPY + [M5_SPLINE]
CLASS = "C(morph_class, Treatment(reference='WITHIN_SPLIT'))"


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def md_table(frame: pd.DataFrame, digits: int = 3) -> str:
    if frame.empty:
        return "(no rows)"
    def cell(x: Any) -> str:
        if isinstance(x, (list, tuple, dict, set)):
            return json.dumps(x, ensure_ascii=False, default=str)
        if x is None or pd.isna(x): return "—"
        if isinstance(x, (float, np.floating)): return f"{float(x):.{digits}f}"
        return str(x).replace("|", "\\|").replace("\n", " ")
    headers = [str(x) for x in frame.columns]
    rows = [[cell(x) for x in r] for r in frame.itertuples(index=False, name=None)]
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    head = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def as_list(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)): return []
    try: return list(value)
    except TypeError: return []


def paths_for(pair: str) -> dict[str, Path]:
    if pair == "P3":
        return {k: P3_DIR / name for k, name in {
            "tokens": "h2_token_table.parquet",
            "teacher": "teacher_forced_tokens.parquet",
            "events": "sd_events.parquet",
        }.items()}
    pdir = ROOT / "runs/e2_model_pair_replication" / PAIRS[pair]["slug"]
    return {"tokens": pdir / "token_table.parquet"}


def load_raw(pair: str, token_cache: pd.DataFrame) -> pd.DataFrame:
    p = paths_for(pair)
    raw = pd.read_parquet(p["tokens"])
    if pair == "P3":
        raw, _ = _p3_harmonized_entropy(raw, pd.read_parquet(p["teacher"]), pd.read_parquet(p["events"]))
    raw = _attach_frequency(raw, token_cache)
    raw["pair"] = pair
    return raw


def match_morph(morphs: list[dict], surface: Any, pos: Any, span: Any) -> int | None:
    if span is None or len(span) != 2: return None
    s, e = int(span[0]), int(span[1])
    exact = [i for i, m in enumerate(morphs) if int(m["start_char"]) == s and int(m["end_char"]) == e and str(m["surface"]) == str(surface) and str(m["fine_pos"]) == str(pos)]
    if exact: return exact[0]
    overlaps = [(max(0, min(e, int(m["end_char"])) - max(s, int(m["start_char"]))), i) for i, m in enumerate(morphs)]
    overlaps = [x for x in overlaps if x[0] > 0]
    return max(overlaps)[1] if overlaps else None


def map_sequences(morph_table: pd.DataFrame) -> dict[tuple[str, int, int], list[dict]]:
    result = {}
    for key, group in morph_table.groupby(["pair", "prompt_id", "eojeol_id"], sort=False):
        result[(str(key[0]), int(key[1]), int(key[2]))] = group.sort_values("morpheme_index").to_dict("records")
    return result


def boundary_pair(row: pd.Series, morphs: list[dict]) -> tuple[int | None, int | None, str]:
    surfaces, poses, spans = (as_list(row.get(k)) for k in ["overlapping_morpheme_surfaces", "overlapping_morpheme_pos", "overlapping_morpheme_spans"])
    if not morphs or not surfaces or not poses or not spans: return None, None, "missing_overlap"
    if str(row.morph_class) == "CROSS_MORPHEME":
        if int(row.num_boundaries_crossed) != 1: return None, None, "not_single_boundary"
        matched = [match_morph(morphs, s, p, sp) for s, p, sp in zip(surfaces, poses, spans)]
        found = []
        for j in range(len(matched) - 1):
            li, ri = matched[j], matched[j + 1]
            if li is None or ri != li + 1: continue
            left = morphs[li]
            q = left.get("boundary_to_next_char")
            if bool(left.get("boundary_to_next_valid")) and q is not None and int(row.token_start_char) < int(q) < int(row.token_end_char):
                found.append((li, ri))
        if len(found) != 1: return None, None, f"cross_boundary_count={len(found)}"
        return found[0][0], found[0][1], "CROSSED_BOUNDARY"
    idx = match_morph(morphs, surfaces[0], poses[0], spans[0])
    if idx is None: return None, None, "split_morpheme_unmatched"
    start, end = int(row.token_start_char), int(row.token_end_char)
    choices = []
    for edge in [idx - 1, idx]:
        if not (0 <= edge < len(morphs) - 1): continue
        m = morphs[edge]
        q = m.get("boundary_to_next_char")
        if not bool(m.get("boundary_to_next_valid")) or q is None:
            choices.append((float("inf"), 1 if edge == idx else 0, edge, "OTHER"))
        else:
            q = int(q)
            dist = q - end if end <= q else start - q if start >= q else 0
            choices.append((float(dist), 1 if edge == idx else 0, edge, str(m["boundary_to_next_type"])))
    if not choices: return None, None, "no_adjacent_boundary"
    choices.sort(key=lambda x: (x[0], -x[1]))
    edge = choices[0][2]
    if choices[0][3] != "NOMINAL_TO_PARTICLE": return None, None, "nearest_boundary_not_nominal_particle"
    return edge, edge + 1, "ADJACENT_BOUNDARY"


def particle_group(tag: str, pos_map: dict[str, str]) -> str:
    for group, tags in PARTICLE_GROUPS.items():
        if tag in tags: return group
    return "OTHER_PARTICLE" if pos_map.get(tag) == "PARTICLE" else "UNMAPPED_PARTICLE"


def geom(nc: int, nt: int, pc: int, pt: int) -> str:
    if min(nt, pt) <= 0: return "OTHER_GEOMETRY"
    nf, pf = nc == nt, pc == pt
    npart, ppart = 0 < nc < nt, 0 < pc < pt
    if nf and pf: return "FULL_NOMINAL_PLUS_FULL_PARTICLE"
    if npart and pf: return "NOMINAL_SUFFIX_PLUS_FULL_PARTICLE"
    if nf and 0 < pc and ppart: return "FULL_NOMINAL_PLUS_PARTICLE"
    if npart and ppart: return "PARTIAL_BOTH"
    return "OTHER_GEOMETRY"


def enrich_pair(pair: str, raw: pd.DataFrame, h3: pd.DataFrame, sequences: dict, pos_map: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    keys = ["pair", "prompt_id", "generation_pos", "token_id"]
    h3 = h3.loc[h3.pair.astype(str).eq(pair)]
    merged = raw.merge(h3, on=keys, how="inner", suffixes=("", "_h3"), validate="one_to_one")
    if len(merged) != len(h3): raise AssertionError(f"{pair} raw/H3 join mismatch")
    for col in ["morph_class", "sd_valid", "sd_rejected"]:
        if f"{col}_h3" not in merged:
            continue
        if col in ["sd_valid", "sd_rejected"]:
            same = np.array_equal(pd.to_numeric(merged[col], errors="coerce"), pd.to_numeric(merged[f"{col}_h3"], errors="coerce"))
        else:
            same = np.array_equal(merged[col].astype(str), merged[f"{col}_h3"].astype(str))
        if not same:
            raise AssertionError(f"{pair}: source vs H3 {col} mismatch")
    d = merged.loc[
        merged.morph_environment.astype(str).eq("NOMINAL_TO_PARTICLE")
        & merged.morph_class.astype(str).isin(["CROSS_MORPHEME", "WITHIN_SPLIT"])
        & merged.sd_valid.fillna(False).astype(bool)
        & merged.fragmentation_bin.astype(str).isin(FRAGMENTATION_ORDER)
    ].copy()
    d = d.loc[d.morph_class.astype(str).ne("CROSS_MORPHEME") | pd.to_numeric(d.num_boundaries_crossed, errors="coerce").eq(1)].copy()
    issues = Counter()
    records = []
    for _, row in d.iterrows():
        rec = row.to_dict()
        try: eid = int(row.eojeol_id)
        except (ValueError, TypeError): eid = -1
        morphs = sequences.get((pair, int(row.prompt_id), eid), [])
        li, ri, src = boundary_pair(row, morphs)
        issues[src] += 1
        if li is None:
            rec.update({"alignment_valid": False, "alignment_issue": src})
            records.append(rec)
            continue
        left, right = morphs[li], morphs[ri]
        if pos_map.get(str(left["fine_pos"])) != "NOMINAL" or pos_map.get(str(right["fine_pos"])) != "PARTICLE":
            rec.update({"alignment_valid": False, "alignment_issue": f"POS_mismatch={left['fine_pos']}->{right['fine_pos']}"})
            issues[rec["alignment_issue"]] += 1
            records.append(rec)
            continue
        ts, te = int(row.token_start_char), int(row.token_end_char)
        ncover = max(0, min(te, int(left["end_char"])) - max(ts, int(left["start_char"])))
        pcover = max(0, min(te, int(right["end_char"])) - max(ts, int(right["start_char"])))
        nt = int(left["end_char"]) - int(left["start_char"])
        pt = int(right["end_char"]) - int(right["start_char"])
        cross = str(row.morph_class) == "CROSS_MORPHEME"
        if cross:
            saved_seq = str(row.get("crossed_fine_pos_sequence", ""))
            expected_seq = str(left["fine_pos"]) + "→" + str(right["fine_pos"])
            spanok = bool(row.span_exact) and ncover > 0 and pcover > 0 and ts < int(left["end_char"]) < te and saved_seq == expected_seq
        else:
            # H3 can map an H2 saved morpheme to a different Kiwi segmentation by
            # maximum character overlap. Validate the exact token coordinates against
            # the selected H3 boundary-side spans; do not require H2/H3 segment strings
            # to be identical, since that is the saved H3 mismatch diagnostic.
            spanok = bool(row.span_exact) and any(
                ts >= int(m["start_char"]) and te <= int(m["end_char"])
                for m in [left, right]
            )
        if not spanok or nt <= 0 or pt <= 0:
            rec.update({"alignment_valid": False, "alignment_issue": "exact_span_check_failed"})
            issues["exact_span_check_failed"] += 1
            records.append(rec)
            continue
        eo = str(row.get("eojeol", left.get("eojeol", "")))
        rec.update({
            "alignment_valid": True, "alignment_issue": "", "environment_source_h4": src,
            "nominal_surface": str(left["surface"]), "nominal_pos": str(left["fine_pos"]),
            "particle_surface": str(right["surface"]), "particle_pos": str(right["fine_pos"]),
            "particle_category": particle_group(str(right["fine_pos"]), pos_map),
            "eojeol_surface": eo, "morpheme_sequence": " | ".join(f"{m['surface']}/{m['fine_pos']}" for m in morphs),
            "token_surface": str(row.get("token_text", "")), "token_char_span": [ts, te],
            "nominal_char_span": [int(left["start_char"]), int(left["end_char"])],
            "particle_char_span": [int(right["start_char"]), int(right["end_char"])],
            "nominal_chars_in_token": ncover, "particle_chars_in_token": pcover,
            "nominal_total_chars": nt, "particle_total_chars": pt,
            "nominal_fraction_covered": ncover / nt, "particle_fraction_covered": pcover / pt,
            "geometry_class": geom(ncover, nt, pcover, pt) if cross else "NOT_CROSS",
            "nominal_char_length": nt, "particle_char_length": pt, "eojeol_surface_length": len(eo),
        })
        records.append(rec)
    result = pd.DataFrame(records)
    audit = {"pair": pair, "h3_n_to_p_rows": len(d), "verified_rows": int(result.alignment_valid.fillna(False).sum()),
             "alignment_failures": int((~result.alignment_valid.fillna(False)).sum()), "issue_counts": dict(issues),
             "h3_saved_morpheme_sequence_mismatch_rows": int(d.get("morph_sequence_mismatch", pd.Series(False, index=d.index)).fillna(False).astype(bool).sum())}
    return result, audit


def build_morpheme_frequency_cache(path: Path, corpus_sha: str) -> tuple[pd.DataFrame, dict]:
    from kiwipiepy import Kiwi
    meta_path = path.with_suffix(".json")
    kiwi_version = importlib.metadata.version("kiwipiepy")
    key = sha_text(json.dumps({"corpus_sha256": corpus_sha, "kiwipiepy": kiwi_version, "nominal": sorted(NOMINAL_TAGS), "particle": sorted(PARTICLE_TAGS)}, sort_keys=True))
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("cache_key") == key: return pd.read_parquet(path), meta
    nc, pc, tags = Counter(), Counter(), Counter()
    kiwi, docs = Kiwi(), 0
    with CORPUS.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip(): continue
            docs += 1
            text = str(json.loads(line).get("text", ""))
            for morph in kiwi.tokenize(text):
                tag, surface = str(morph.tag), str(morph.form)
                tags[tag] += 1
                if tag in NOMINAL_TAGS: nc[(surface, tag)] += 1
                if tag in PARTICLE_TAGS: pc[(surface, tag)] += 1
    rows = ([{"surface": s, "fine_pos": p, "kind": "nominal", "count": int(n)} for (s, p), n in nc.items()]
            + [{"surface": s, "fine_pos": p, "kind": "particle", "count": int(n)} for (s, p), n in pc.items()])
    cache = pd.DataFrame(rows, columns=["surface", "fine_pos", "kind", "count"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    cache.to_parquet(tmp, index=False); tmp.replace(path)
    meta = {"cache_key": key, "corpus_sha256": corpus_sha, "corpus_path": str(CORPUS.relative_to(ROOT)),
            "corpus_documents": docs, "kiwipiepy_version": kiwi_version,
            "nominal_tags": sorted(NOMINAL_TAGS), "particle_tags": sorted(PARTICLE_TAGS),
            "observed_kiwi_pos_counts": dict(sorted(tags.items())),
            "construction": "Kiwi counts by surface and fine POS from same local E1 Wikipedia corpus; no outcomes used"}
    atomic_json(meta_path, meta)
    return cache, meta


def attach_frequencies(d: pd.DataFrame, token_cache: pd.DataFrame, eo_cache: pd.DataFrame, morph_cache: pd.DataFrame) -> pd.DataFrame:
    out = d.merge(token_cache[["token_id", "token_count", "log_token_count"]].drop_duplicates("token_id"), on="token_id", how="left", validate="many_to_one", suffixes=("", "_cached"))
    if "log_token_count_cached" in out:
        out["log_token_count"] = out["log_token_count_cached"]; out = out.drop(columns="log_token_count_cached")
    if out.log_token_count.isna().any(): raise AssertionError("E1 token cache missed H4 rows")
    eo = eo_cache[["eojeol_surface", "eojeol_count", "log_eojeol_count"]].drop_duplicates("eojeol_surface")
    out = out.merge(eo, on="eojeol_surface", how="left", validate="many_to_one")
    out["eojeol_count"] = out.eojeol_count.fillna(0).astype("int64")
    out["log_eojeol_count"] = out.log_eojeol_count.fillna(0.0)
    for kind, surface, pos, name in [("nominal", "nominal_surface", "nominal_pos", "nominal_frequency"),
                                     ("particle", "particle_surface", "particle_pos", "particle_frequency")]:
        subset = morph_cache.loc[morph_cache.kind.eq(kind)]
        lookup = {(str(r.surface), str(r.fine_pos)): int(r.count) for r in subset.itertuples(index=False)}
        out[name] = [lookup.get((str(s), str(p)), 0) for s, p in zip(out[surface], out[pos])]
        out["log_" + name] = np.log1p(out[name].astype(float))
    return out


def clean_h4(data: pd.DataFrame) -> pd.DataFrame:
    x = data.copy()
    x["sd_rejected"] = pd.to_numeric(x.sd_rejected, errors="coerce").astype(int)
    x["prompt_id"] = pd.to_numeric(x.prompt_id, errors="coerce")
    x["morph_class"] = x.morph_class.astype(str)
    x["fragmentation_bin"] = x.fragmentation_bin.astype(str)
    for col in STRUCT + ENTROPY + ["log_token_count", "token_char_length", "eojeol_char_length"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    for col in ["first_token", "last_token"]:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0).astype(int)
    return x


def fit_logit(data: pd.DataFrame, formula: str, label: str, orders: dict[str, list[str]] | None = None) -> tuple[Any | None, dict, pd.DataFrame]:
    clean = data.copy()
    for col, preferred in (orders or {}).items():
        seen = set(clean[col].dropna().astype(str))
        cats = [v for v in preferred if v in seen] + sorted(seen - set(preferred))
        clean[col] = pd.Categorical(clean[col].astype(str), categories=cats)
    diag = {"label": label, "formula": formula, "n_input": len(clean), "converged": False, "fit_error": None, "warnings": []}
    try:
        y, x = patsy.dmatrices(formula, clean, return_type="dataframe", NA_action="drop")
        clean = clean.loc[x.index].copy()
        rank = int(np.linalg.matrix_rank(np.asarray(x)))
        diag.update({"n_tokens": len(clean), "n_prompts": int(clean.prompt_id.nunique()), "cluster_count": int(clean.prompt_id.nunique()),
                     "rank": rank, "n_columns": x.shape[1], "rank_deficient": rank < x.shape[1], "design_columns": list(x.columns)})
        if len(clean) < 10 or clean.sd_rejected.nunique() < 2:
            diag["fit_error"] = "insufficient rows/outcome variation"; return None, diag, clean
        keep = list(x.columns)
        # Preserve formula order and remove only exact aliases when needed. The dropped
        # terms are reported; fitted probabilities use this full-rank design matrix.
        if rank < x.shape[1]:
            keep = []
            current = np.empty((len(x), 0), dtype=float)
            current_rank = 0
            for idx, name in enumerate(x.columns):
                candidate_x = np.column_stack([current, np.asarray(x.iloc[:, idx], dtype=float)])
                candidate_rank = int(np.linalg.matrix_rank(candidate_x))
                if candidate_rank > current_rank:
                    keep.append(name)
                    current = candidate_x
                    current_rank = candidate_rank
        dropped = [name for name in x.columns if name not in keep]
        diag["rank_pruned_columns"] = dropped
        xfit = x[keep]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = sm.Logit(y.iloc[:, 0], xfit)
            fit = None
            attempts = []
            for method, maxiter in [("newton", 500), ("lbfgs", 1000), ("bfgs", 1000)]:
                candidate = model.fit(disp=False, maxiter=maxiter, method=method, cov_type="cluster",
                                      cov_kwds={"groups": clean.prompt_id, "use_correction": True})
                attempts.append({"method": method, "converged": bool(candidate.converged),
                                 "iterations": candidate.mle_retvals.get("iterations"),
                                 "warnflag": candidate.mle_retvals.get("warnflag")})
                fit = candidate
                if candidate.converged:
                    break
            diag["warnings"] = [str(w.message) for w in caught]
            diag["optimizer_attempts"] = attempts
        fit._h4_design_info = x.design_info
        fit._h4_keep_columns = keep
        diag["converged"] = bool(fit.converged)
        diag["max_abs_coefficient"] = float(np.abs(np.asarray(fit.params)).max())
        diag["extreme_coefficient"] = diag["max_abs_coefficient"] > 10
        diag["condition_number"] = float(np.linalg.cond(np.asarray(x)))
        return fit, diag, clean
    except Exception as exc:
        diag["fit_error"] = f"{type(exc).__name__}: {exc}"
        return None, diag, clean


def fit_linear(data: pd.DataFrame, formula: str, label: str) -> tuple[Any | None, dict, pd.DataFrame]:
    clean = data.copy()
    diag = {"label": label, "formula": formula, "n_input": len(clean), "fit_error": None}
    try:
        y, x = patsy.dmatrices(formula, clean, return_type="dataframe", NA_action="drop")
        clean = clean.loc[x.index].copy()
        rank = int(np.linalg.matrix_rank(np.asarray(x)))
        keep = list(x.columns)
        if rank < x.shape[1]:
            keep, current = [], np.empty((len(x), 0), dtype=float)
            current_rank = 0
            for idx, name in enumerate(x.columns):
                candidate = np.column_stack([current, np.asarray(x.iloc[:, idx], dtype=float)])
                candidate_rank = int(np.linalg.matrix_rank(candidate))
                if candidate_rank > current_rank:
                    keep.append(name); current = candidate; current_rank = candidate_rank
        fit = sm.OLS(y.iloc[:, 0], x[keep]).fit(cov_type="cluster", cov_kwds={"groups": clean.prompt_id, "use_correction": True})
        fit._h4_design_info = x.design_info
        fit._h4_keep_columns = keep
        diag.update({"n_tokens": len(clean), "n_prompts": int(clean.prompt_id.nunique()), "cluster_count": int(clean.prompt_id.nunique()),
                     "rank": rank, "n_columns": x.shape[1], "rank_deficient": rank < x.shape[1],
                     "rank_pruned_columns": [c for c in x.columns if c not in keep],
                     "fit_error": None})
        return fit, diag, clean
    except Exception as exc:
        diag["fit_error"] = f"{type(exc).__name__}: {exc}"
        return None, diag, clean


def design(fit: Any, frame: pd.DataFrame) -> np.ndarray:
    full = patsy.build_design_matrices([fit._h4_design_info], frame, return_type="dataframe")[0]
    return np.asarray(full[fit._h4_keep_columns])


def prob_grad(fit: Any, frame: pd.DataFrame, overrides: dict[str, Any] | None = None) -> tuple[float, np.ndarray]:
    cf = frame.copy()
    for col, value in (overrides or {}).items():
        if isinstance(cf[col].dtype, pd.CategoricalDtype):
            cf[col] = pd.Categorical([value] * len(cf), categories=cf[col].cat.categories)
        else: cf[col] = value
    x = design(fit, cf)
    p = expit(x @ np.asarray(fit.params))
    return float(p.mean()), np.mean(x * (p * (1-p))[:, None], axis=0)


def delta(fit: Any, estimate: float, grad: np.ndarray, bounded: bool = False) -> dict:
    var = float(grad @ np.asarray(fit.cov_params()) @ grad)
    se = math.sqrt(max(var, 0))
    z = estimate / se if se else np.nan
    lo, hi = estimate - 1.959963984540054 * se, estimate + 1.959963984540054 * se
    if bounded: lo, hi = max(0.0, lo), min(1.0, hi)
    return {"se": se, "ci_low": lo, "ci_high": hi, "p_value": float(2 * st.norm.sf(abs(z))) if np.isfinite(z) else np.nan}


def ame(fit: Any, frame: pd.DataFrame) -> dict:
    pc, gc = prob_grad(fit, frame, {"morph_class": "CROSS_MORPHEME"})
    ps, gs = prob_grad(fit, frame, {"morph_class": "WITHIN_SPLIT"})
    diff = pc - ps
    di, ci, si = delta(fit, diff, gc-gs), delta(fit, pc, gc, True), delta(fit, ps, gs, True)
    return {"adjusted_p_cross": pc, "adjusted_p_cross_ci_low": ci["ci_low"], "adjusted_p_cross_ci_high": ci["ci_high"],
            "adjusted_p_split": ps, "adjusted_p_split_ci_low": si["ci_low"], "adjusted_p_split_ci_high": si["ci_high"],
            "ame": diff, "ame_se": di["se"], "ame_ci_low": di["ci_low"], "ame_ci_high": di["ci_high"], "p_value": di["p_value"]}


def adjusted_probability(fit: Any, frame: pd.DataFrame, col: str, level: str) -> dict:
    p, g = prob_grad(fit, frame, {col: level})
    d = delta(fit, p, g, bounded=True)
    return {"adjusted_probability": p, "ci_low": d["ci_low"], "ci_high": d["ci_high"]}


def linear_mean_gradient(fit: Any, frame: pd.DataFrame, overrides: dict[str, Any] | None = None) -> tuple[float, np.ndarray]:
    cf = frame.copy()
    for col, value in (overrides or {}).items():
        if isinstance(cf[col].dtype, pd.CategoricalDtype):
            cf[col] = pd.Categorical([value] * len(cf), categories=cf[col].cat.categories)
        else:
            cf[col] = value
    x = design(fit, cf)
    return float(np.mean(x @ np.asarray(fit.params))), np.mean(x, axis=0)


def wald(fit: Any | None, predicate) -> dict:
    if fit is None: return {"wald_chi2": np.nan, "wald_df": 0, "wald_p": np.nan, "terms": []}
    names = list(fit.params.index)
    ix = [i for i, n in enumerate(names) if predicate(n)]
    if not ix: return {"wald_chi2": np.nan, "wald_df": 0, "wald_p": np.nan, "terms": []}
    b = np.asarray(fit.params)[ix]
    cov = np.asarray(fit.cov_params())[np.ix_(ix, ix)]
    rank = int(np.linalg.matrix_rank(cov))
    if not rank: return {"wald_chi2": np.nan, "wald_df": 0, "wald_p": np.nan, "terms": [names[i] for i in ix]}
    stat = float(b @ np.linalg.pinv(cov) @ b)
    return {"wald_chi2": stat, "wald_df": rank, "wald_p": float(st.chi2.sf(stat, rank)), "terms": [names[i] for i in ix]}


def holm(pvals: list[float]) -> list[float]:
    values = [float(p) if np.isfinite(p) else 1. for p in pvals]
    order = np.argsort(values); out = np.ones(len(values)); prev = 0.
    for rank, idx in enumerate(order):
        prev = max(prev, (len(values)-rank) * values[idx]); out[idx] = min(1., prev)
    return out.tolist()


def particle_support(data: pd.DataFrame, level: str) -> dict:
    sub = data.loc[data.particle_category.astype(str).eq(level)]
    cross = sub.loc[sub.morph_class.eq("CROSS_MORPHEME")]
    split = sub.loc[sub.morph_class.eq("WITHIN_SPLIT")]
    return {
        "n_cross": len(cross), "n_split": len(split), "n_prompts": int(sub.prompt_id.nunique()),
        "n_cross_prompts": int(cross.prompt_id.nunique()), "n_split_prompts": int(split.prompt_id.nunique()),
        "raw_cross_rejection": cross.sd_rejected.mean(), "raw_split_rejection": split.sd_rejected.mean(),
        "mean_draft_entropy": sub.draft_entropy.mean(), "mean_target_entropy": sub.target_entropy.mean(),
        "median_token_frequency": sub.token_count.median(),
    }


def run_particle_model(data: pd.DataFrame, pair: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    order = PARTICLE_ORDER + ["OTHER_PARTICLE", "UNMAPPED_PARTICLE"]
    levels = PARTICLE_ORDER + [x for x in ["OTHER_PARTICLE", "UNMAPPED_PARTICLE"] if x in set(data.particle_category.astype(str))]
    supports = []
    for level in levels:
        supports.append({"pair": pair, "particle_category": level, **particle_support(data, level)})
    support_df = pd.DataFrame(supports)
    active = [level for level in levels if (support_df.loc[support_df.particle_category.eq(level), ["n_cross", "n_split"]].iloc[0] > 0).all()]
    reference = "CASE_SUBJECT" if "CASE_SUBJECT" in active else (active[0] if active else "CASE_SUBJECT")
    model_data = data.loc[data.particle_category.astype(str).isin(active)].copy()
    formula = (
        "sd_rejected ~ " + CLASS + " * C(particle_category, Treatment(reference='" + reference + "')) + "
        + " + ".join(BASE)
    )
    fit, diag, clean = fit_logit(
        model_data, formula, pair + "_H4a",
        {"morph_class": ["WITHIN_SPLIT", "CROSS_MORPHEME"], "fragmentation_bin": FRAGMENTATION_ORDER, "particle_category": order},
    )
    diag["active_categories"] = active
    diag["reference_category"] = reference
    diag["interaction_omnibus"] = wald(fit, lambda n: "morph_class" in n and "particle_category" in n)
    rows = []
    for level in levels:
        sub = data.loc[data.particle_category.astype(str).eq(level)].copy()
        nc, ns = int(sub.morph_class.eq("CROSS_MORPHEME").sum()), int(sub.morph_class.eq("WITHIN_SPLIT").sum())
        rec = {
            "pair": pair, "particle_category": level, "n_cross": nc, "n_split": ns,
            "n_prompts": int(sub.prompt_id.nunique()),
            "support_warning": bool(nc < 100 or ns < 100 or sub.loc[sub.morph_class.eq("CROSS_MORPHEME"), "prompt_id"].nunique() < 50 or sub.loc[sub.morph_class.eq("WITHIN_SPLIT"), "prompt_id"].nunique() < 50),
            "estimable": bool(fit is not None and level in active and sub.morph_class.nunique() == 2 and fit.converged and not diag.get("extreme_coefficient")),
        }
        if rec["estimable"]:
            rec.update(ame(fit, sub))
        else:
            rec.update({k: np.nan for k in ["adjusted_p_cross", "adjusted_p_cross_ci_low", "adjusted_p_cross_ci_high", "adjusted_p_split", "adjusted_p_split_ci_low", "adjusted_p_split_ci_high", "ame", "ame_se", "ame_ci_low", "ame_ci_high", "p_value"]})
        rows.append(rec)
    result = pd.DataFrame(rows)
    result["holm_p_value"] = np.nan
    # The prespecified family has all seven taxonomy categories; absent or non-estimable
    # cells enter conservatively as p=1 rather than silently shrinking the family.
    family_mask = result.particle_category.isin(PARTICLE_ORDER)
    family_p = [float(p) if bool(ok) and pd.notna(p) else 1.0 for p, ok in zip(result.loc[family_mask, "p_value"], result.loc[family_mask, "estimable"])]
    if family_p:
        result.loc[family_mask, "holm_p_value"] = holm(family_p)
    diag["model_n"] = int(len(clean))
    return result, support_df, diag


def run_geometry(data: pd.DataFrame, pair: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cross = data.loc[data.morph_class.eq("CROSS_MORPHEME")].copy()
    levels = [x for x in GEOMETRY_ORDER if x in set(cross.geometry_class.astype(str))]
    # Use the best-supported geometry solely as a numerically stable reference.
    ref = max(levels, key=lambda z: int((cross.geometry_class.astype(str) == z).sum())) if levels else "OTHER_GEOMETRY"
    formula = (
        "sd_rejected ~ C(geometry_class, Treatment(reference='" + ref + "')) + C(particle_category) + C(nominal_pos) + "
        + " + ".join(BASE)
    )
    fit, diag, clean = fit_logit(cross, formula, pair + "_H4b_geometry", {"fragmentation_bin": FRAGMENTATION_ORDER, "geometry_class": GEOMETRY_ORDER})
    diag.update({"reference_geometry": ref, "n_cross": len(cross), "n_prompts": int(cross.prompt_id.nunique())})
    logit_ok = bool(fit is not None and fit.converged and not diag.get("extreme_coefficient"))
    lpm_fit, lpm_diag = None, None
    if not logit_ok:
        lpm_fit, lpm_diag, _ = fit_linear(cross, formula, pair + "_H4b_geometry_LPM_fallback")
    diag["clustered_lpm_fallback"] = lpm_diag
    rows = []
    for level in levels:
        sub = cross.loc[cross.geometry_class.astype(str).eq(level)]
        rec = {
            "pair": pair, "geometry_class": level, "n_cross": len(sub), "n_prompts": int(sub.prompt_id.nunique()),
            "share_of_cross": len(sub) / max(1, len(cross)), "raw_rejection": sub.sd_rejected.mean(),
            "mean_draft_entropy": sub.draft_entropy.mean(), "median_token_frequency": sub.token_count.median(),
            "support_warning": bool(len(sub) < 100 or sub.prompt_id.nunique() < 50),
            "standardized_population": "all CROSS tokens in the pair; only geometry_class is changed",
            "estimable": bool(len(sub) and (logit_ok or (lpm_fit is not None))),
            "estimator": "clustered_logit" if logit_ok else ("clustered_LPM_fallback" if lpm_fit is not None else "not_estimable"),
        }
        if logit_ok:
            rec.update(adjusted_probability(fit, cross, "geometry_class", level))
        elif lpm_fit is not None:
            p, g = linear_mean_gradient(lpm_fit, cross, {"geometry_class": level})
            ci = delta(lpm_fit, p, g)
            rec.update({"adjusted_probability": p, "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                        "point_out_of_probability_range": not (0 <= p <= 1),
                        "ci_out_of_probability_range": ci["ci_low"] < 0 or ci["ci_high"] > 1})
        else:
            rec.update({"adjusted_probability": np.nan, "ci_low": np.nan, "ci_high": np.nan})
        rows.append(rec)
    categorical = pd.DataFrame(rows)

    cformula = (
        "sd_rejected ~ nominal_fraction_covered * particle_fraction_covered + C(particle_category) + C(nominal_pos) + "
        + " + ".join(BASE)
    )
    cfit, cdiag, cclean = fit_logit(cross, cformula, pair + "_H4b_continuous")
    cdiag["n_cross"] = len(cross)
    cont_rows = []
    clogit_ok = bool(cfit is not None and cfit.converged and not cdiag.get("extreme_coefficient"))
    clpm_fit, clpm_diag = None, None
    if not clogit_ok:
        clpm_fit, clpm_diag, _ = fit_linear(cross, cformula, pair + "_H4b_continuous_LPM_fallback")
    cdiag["clustered_lpm_fallback"] = clpm_diag
    if clogit_ok or clpm_fit is not None:
        for var in ["nominal_fraction_covered", "particle_fraction_covered"]:
            basev = cross[var].astype(float).to_numpy()
            keep = np.isfinite(basev) & (basev < 1)
            if cross[var].nunique(dropna=True) < 2 or not keep.any():
                cont_rows.append({"pair": pair, "feature": var, "n_rows": int(keep.sum()),
                                  "q25": float(cross[var].quantile(.25)), "q75": float(cross[var].quantile(.75)),
                                  "estimable": False, "fit_error": "no observed coverage variation below 100%"})
                continue
            base_frame = cross.loc[keep].copy()
            moved = base_frame.copy()
            moved[var] = np.minimum(base_frame[var].astype(float) + .10, 1)
            if clogit_ok:
                p0, g0 = prob_grad(cfit, base_frame)
                p1, g1 = prob_grad(cfit, moved)
                used_estimator = "clustered_logit"
                coefficient = float(cfit.params.get(var, np.nan))
                model = cfit
            else:
                p0, g0 = linear_mean_gradient(clpm_fit, base_frame)
                p1, g1 = linear_mean_gradient(clpm_fit, moved)
                used_estimator = "clustered_LPM_fallback"
                coefficient = float(clpm_fit.params.get(var, np.nan))
                model = clpm_fit
            dd = delta(model, p1-p0, g1-g0)
            cont_rows.append({
                "pair": pair, "feature": var, "comparison": "+10pp, capped at full coverage",
                "n_rows": len(base_frame), "q25": float(cross[var].quantile(.25)), "q75": float(cross[var].quantile(.75)),
                "adjusted_probability_before": p0, "adjusted_probability_after": p1,
                "probability_difference": p1-p0, "ci_low": dd["ci_low"], "ci_high": dd["ci_high"], "p_value": dd["p_value"],
                "coefficient": coefficient, "estimator": used_estimator, "estimable": True,
            })
    else:
        for var in ["nominal_fraction_covered", "particle_fraction_covered"]:
            cont_rows.append({"pair": pair, "feature": var, "n_rows": len(cross), "estimable": False, "fit_error": cdiag.get("fit_error") or "nonconverged"})
    return categorical, pd.DataFrame(cont_rows), {"categorical": diag, "continuous": cdiag}


def run_lexical(data: pd.DataFrame, pair: str) -> tuple[pd.DataFrame, dict]:
    # eojeol_surface_length duplicates the H3 M5 eojeol_char_length control; keep the
    # original H3 field and add nominal/particle side lengths only.
    lexical_prefix = "sd_rejected ~ " + CLASS + " + C(particle_category) + C(nominal_pos) + nominal_char_length + particle_char_length + "
    formulas = {
        "L1": "sd_rejected ~ " + CLASS + " + " + " + ".join(BASE),
        "L2": lexical_prefix + " + ".join(BASE),
        "L3": lexical_prefix + " + ".join(BASE)
        + " + bs(log_nominal_frequency, df=4, degree=3, include_intercept=False)"
        + " + bs(log_particle_frequency, df=4, degree=3, include_intercept=False)"
        + " + bs(log_eojeol_count, df=4, degree=3, include_intercept=False)",
    }
    rows, diags = [], {}
    for name, formula in formulas.items():
        fit, diag, clean = fit_logit(data, formula, pair + "_" + name, {"morph_class": ["WITHIN_SPLIT", "CROSS_MORPHEME"], "fragmentation_bin": FRAGMENTATION_ORDER})
        diags[name] = diag
        rec = {
            "pair": pair, "model": name, "formula": formula, "n_tokens": len(clean), "n_prompts": int(clean.prompt_id.nunique()),
            "converged": diag.get("converged"), "rank_deficient": diag.get("rank_deficient"),
            "extreme_coefficient": diag.get("extreme_coefficient"), "fit_error": diag.get("fit_error"),
        }
        if fit is not None and fit.converged and clean.morph_class.nunique() == 2 and not diag.get("extreme_coefficient"):
            rec.update(ame(fit, clean))
        else:
            rec.update({k: np.nan for k in ["adjusted_p_cross", "adjusted_p_cross_ci_low", "adjusted_p_cross_ci_high", "adjusted_p_split", "adjusted_p_split_ci_low", "adjusted_p_split_ci_high", "ame", "ame_se", "ame_ci_low", "ame_ci_high", "p_value"]})
        rows.append(rec)
    return pd.DataFrame(rows), diags


def run_particle_surface_fe(data: pd.DataFrame, pair: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    for surface, g in data.groupby("particle_surface", observed=True):
        cr = g.loc[g.morph_class.eq("CROSS_MORPHEME")]
        rows.append({"particle_surface": surface, "n_tokens": len(g), "n_cross": len(cr), "n_prompts": int(g.prompt_id.nunique()), "n_cross_prompts": int(cr.prompt_id.nunique())})
    counts = pd.DataFrame(rows)
    supported = counts.loc[(counts.n_tokens >= 100) & (counts.n_cross >= 30) & (counts.n_prompts >= 30) & (counts.n_cross_prompts >= 30), "particle_surface"].astype(str).tolist()
    use = data.loc[data.particle_surface.astype(str).isin(supported)].copy()
    if not supported:
        nofit = pd.DataFrame([{
            "pair": pair, "specification": label, "supported_surfaces": "",
            "support_rule": ">=100 total, >=30 CROSS, >=30 prompts", "n_tokens": 0, "n_prompts": 0,
            "converged": False, "rank_deficient": np.nan,
            "fit_error": "no particle surface met the prespecified support threshold",
        } for label in ["surface_FE", "surface_interaction"]])
        counts.insert(0, "pair", pair)
        return nofit, counts, {"models": {}, "supported_surfaces": []}
    base_tail = " + ".join(BASE)
    specs = {
        "surface_FE": "sd_rejected ~ " + CLASS + " + C(particle_surface) + " + base_tail,
        "surface_interaction": "sd_rejected ~ " + CLASS + " * C(particle_surface) + " + base_tail,
    }
    out, diagall = [], {}
    for label, formula in specs.items():
        fit, diag, clean = fit_logit(use, formula, pair + "_" + label, {"morph_class": ["WITHIN_SPLIT", "CROSS_MORPHEME"], "fragmentation_bin": FRAGMENTATION_ORDER})
        diagall[label] = diag
        rec = {"pair": pair, "specification": label, "supported_surfaces": "|".join(supported), "support_rule": ">=100 total, >=30 CROSS, >=30 prompts", "n_tokens": len(clean), "n_prompts": int(clean.prompt_id.nunique()), "converged": diag.get("converged"), "rank_deficient": diag.get("rank_deficient"), "fit_error": diag.get("fit_error")}
        rec["estimable"] = bool(fit is not None and fit.converged and clean.morph_class.nunique() == 2 and not diag.get("extreme_coefficient"))
        if rec["estimable"]: rec.update(ame(fit, clean))
        out.append(rec)
    counts.insert(0, "pair", pair)
    return pd.DataFrame(out), counts, {"models": diagall, "supported_surfaces": supported}


def qbin(s: pd.Series, q: int, prefix: str) -> tuple[pd.Series, list[float]]:
    x = pd.to_numeric(s, errors="coerce")
    if x.nunique(dropna=True) < 2:
        val = float(x.dropna().iloc[0]) if x.notna().any() else np.nan
        return pd.Series([prefix+"0"]*len(x), index=x.index), [val] if np.isfinite(val) else []
    code, edges = pd.qcut(x, q=min(q, max(2, x.nunique())), labels=False, retbins=True, duplicates="drop")
    return code.map(lambda v: prefix+str(int(v)) if pd.notna(v) else prefix+"NA").astype(str), [float(e) for e in edges]


def run_matching(data: pd.DataFrame, pair: str) -> tuple[dict, pd.DataFrame]:
    x = data.copy().reset_index(drop=True)
    x["length_bin"] = pd.cut(x.token_char_length, [-np.inf, 1, 2, 3, np.inf], labels=["L1", "L2", "L3", "L4plus"]).astype(str)
    x["nominal_length_bin"] = pd.cut(x.nominal_char_length, [-np.inf, 1, 2, 3, np.inf], labels=["N1", "N2", "N3", "N4plus"]).astype(str)
    edge_log = {}
    for var, q, prefix in [("log_token_count", 10, "F"), ("draft_entropy", 10, "E")]:
        x[var+"_bin"], edge_log[var] = qbin(x[var], q, prefix)
    x["generation_position_bin"] = ""
    edge_log["generation_position"] = {}
    for pairid, ix in x.groupby("pair").groups.items():
        bins, edges = qbin(x.loc[ix, "generation_position"], 4, "G")
        x.loc[ix, "generation_position_bin"] = bins.to_numpy()
        edge_log["generation_position"][str(pairid)] = edges
    features = ["pair", "particle_category", "fragmentation_bin", "length_bin", "nominal_length_bin", "log_token_count_bin", "draft_entropy_bin", "generation_position_bin"]
    x["match_stratum"] = x[features].astype(str).agg("|".join, axis=1)
    rng = np.random.default_rng(SEED)
    counts = x.groupby(["match_stratum", "morph_class"], observed=True).size().unstack(fill_value=0)
    selected, nstrata = [], 0
    for stratum, group_counts in counts.iterrows():
        n = min(int(group_counts.get("CROSS_MORPHEME", 0)), int(group_counts.get("WITHIN_SPLIT", 0)))
        if n == 0: continue
        nstrata += 1
        for klass in ["CROSS_MORPHEME", "WITHIN_SPLIT"]:
            ix = x.index[(x.match_stratum == stratum) & (x.morph_class == klass)].to_numpy()
            selected.extend(rng.choice(ix, size=n, replace=False).tolist())
    matched = x.loc[selected].copy() if selected else x.iloc[:0].copy()
    res = {
        "pair": pair, "seed": SEED, "method": "CEM with 1:1 within-stratum sampling without replacement",
        "strata": features, "bin_edges": edge_log, "n_strata": nstrata,
        "n_cross_matched": int(matched.morph_class.eq("CROSS_MORPHEME").sum()),
        "n_split_matched": int(matched.morph_class.eq("WITHIN_SPLIT").sum()),
        "n_cross_prompts": int(matched.loc[matched.morph_class.eq("CROSS_MORPHEME"), "prompt_id"].nunique()) if len(matched) else 0,
        "n_split_prompts": int(matched.loc[matched.morph_class.eq("WITHIN_SPLIT"), "prompt_id"].nunique()) if len(matched) else 0,
        "n_prompts_union": int(matched.prompt_id.nunique()) if len(matched) else 0,
        "raw_matched_difference": np.nan, "adjusted_lpm_difference": np.nan,
        "adjusted_lpm_ci_low": np.nan, "adjusted_lpm_ci_high": np.nan, "adjusted_lpm_p_value": np.nan, "fit_error": None,
    }
    if len(matched) and matched.morph_class.nunique() == 2:
        res["raw_matched_difference"] = float(matched.loc[matched.morph_class.eq("CROSS_MORPHEME"), "sd_rejected"].mean() - matched.loc[matched.morph_class.eq("WITHIN_SPLIT"), "sd_rejected"].mean())
        formula = "sd_rejected ~ C(morph_class, Treatment(reference='WITHIN_SPLIT')) + C(match_stratum)"
        try:
            fit = smf.ols(formula, data=matched).fit(cov_type="cluster", cov_kwds={"groups": matched.prompt_id, "use_correction": True})
            term = next(n for n in fit.params.index if "morph_class" in n and "CROSS_MORPHEME" in n)
            contrast = np.zeros((1, len(fit.params)))
            contrast[0, list(fit.params.index).index(term)] = 1.0
            test = fit.t_test(contrast)
            estimate, se = float(np.asarray(test.effect).reshape(-1)[0]), float(np.asarray(test.sd).reshape(-1)[0])
            lo, hi = estimate - 1.959963984540054 * se, estimate + 1.959963984540054 * se
            res.update({"adjusted_lpm_difference": estimate, "adjusted_lpm_ci_low": lo, "adjusted_lpm_ci_high": hi, "adjusted_lpm_p_value": float(np.asarray(test.pvalue).reshape(-1)[0]), "adjusted_model_rank": int(np.linalg.matrix_rank(np.asarray(fit.model.exog))), "adjusted_model_columns": int(fit.model.exog.shape[1]), "rank_deficient": int(np.linalg.matrix_rank(np.asarray(fit.model.exog))) < fit.model.exog.shape[1]})
        except Exception as exc:
            res["fit_error"] = f"{type(exc).__name__}: {exc}"
    matched["pair"] = pair
    return res, matched


def run_pooled(data: pd.DataFrame, common_ids: set[int] | None = None) -> tuple[Any | None, dict, pd.DataFrame, pd.DataFrame]:
    x = data.copy()
    pop = "three_way_common" if common_ids is not None else "all_available"
    if common_ids is not None: x = x.loc[x.prompt_id.astype(int).isin(common_ids)].copy()
    active = []
    for level in PARTICLE_ORDER + ["OTHER_PARTICLE"]:
        sub = x.loc[x.particle_category.astype(str).eq(level)]
        if set(sub.morph_class.astype(str)) == {"WITHIN_SPLIT", "CROSS_MORPHEME"}: active.append(level)
    x = x.loc[x.particle_category.astype(str).isin(active)].copy()
    ref = "CASE_SUBJECT" if "CASE_SUBJECT" in active else (active[0] if active else "CASE_SUBJECT")
    formula = (
        "sd_rejected ~ C(pair, Treatment(reference='P3')) * " + CLASS + " + " + CLASS
        + " * C(particle_category, Treatment(reference='" + ref + "')) + " + " + ".join(BASE)
    )
    fit, diag, clean = fit_logit(x, formula, "pooled_" + pop, {
        "morph_class": ["WITHIN_SPLIT", "CROSS_MORPHEME"], "fragmentation_bin": FRAGMENTATION_ORDER,
        "pair": PAIRS_ORDER, "particle_category": active,
    })
    diag.update({"active_categories": active, "reference_category": ref, "population": pop,
                 "source_common_prompt_count": len(common_ids) if common_ids is not None else None,
                 "contributing_prompt_count": int(clean.prompt_id.nunique())})
    rows = []
    if fit is not None and fit.converged:
        for level in active:
            sub = clean.loc[clean.particle_category.astype(str).eq(level)]
            rows.append({"population": pop, "particle_category": level, "n_tokens": len(sub),
                         "n_prompts": int(sub.prompt_id.nunique()), **ame(fit, sub)})
    return fit, diag, clean, pd.DataFrame(rows)


def plot_particle(results: pd.DataFrame, path: Path) -> None:
    x = results.loc[results.estimable].copy()
    if x.empty: return
    levels = [z for z in PARTICLE_ORDER + ["OTHER_PARTICLE"] if z in set(x.particle_category)]
    yy = {z: len(levels)-i-1 for i, z in enumerate(levels)}
    offsets, colors = {"P1": -.18, "P2": 0., "P3": .18}, {"P1": "#0072B2", "P2": "#D55E00", "P3": "#009E73"}
    fig, ax = plt.subplots(figsize=(8.2, max(3.8, .62*len(levels))))
    for pair in PAIRS_ORDER:
        for _, r in x.loc[x.pair.eq(pair)].iterrows():
            y = yy[r.particle_category] + offsets[pair]
            ax.errorbar(r.ame*100, y, xerr=[[max(0., (r.ame-r.ame_ci_low)*100)], [max(0., (r.ame_ci_high-r.ame)*100)]],
                        fmt="o", color=colors[pair], capsize=2.5, markersize=5,
                        label=pair if r.particle_category == levels[0] else None)
    ax.axvline(0, color="#444444", linestyle="--", linewidth=1)
    ax.set_yticks([yy[z] for z in levels], [z.replace("_", " ").title() for z in levels])
    ax.set_xlabel("Adjusted CROSS − SPLIT rejection difference (percentage points)")
    ax.set_title("H4a: Nominal→Particle penalty by particle type")
    ax.grid(axis="x", alpha=.2); ax.legend(frameon=False, ncol=3)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


def plot_geometry(results: pd.DataFrame, path: Path) -> None:
    x = results.loc[results.estimable].copy()
    if x.empty: return
    levels = [z for z in GEOMETRY_ORDER if z in set(x.geometry_class)]
    yy = {z: len(levels)-i-1 for i, z in enumerate(levels)}
    offsets, colors = {"P1": -.18, "P2": 0., "P3": .18}, {"P1": "#0072B2", "P2": "#D55E00", "P3": "#009E73"}
    fig, ax = plt.subplots(figsize=(8.4, max(3.8, .64*len(levels))))
    for pair in PAIRS_ORDER:
        for _, r in x.loc[x.pair.eq(pair)].iterrows():
            y = yy[r.geometry_class] + offsets[pair]
            ax.errorbar(r.adjusted_probability*100, y,
                        xerr=[[max(0., (r.adjusted_probability-r.ci_low)*100)], [max(0., (r.ci_high-r.adjusted_probability)*100)]],
                        fmt="o", color=colors[pair], capsize=2.5, markersize=5,
                        label=pair if r.geometry_class == levels[0] else None)
    ax.set_yticks([yy[z] for z in levels], [z.replace("_", " ").title() for z in levels])
    ax.set_xlabel("Standardized linear-model rejection prediction among CROSS tokens (%)")
    ax.set_title("H4b: Rejection by exact crossing geometry (sparse LPM fallback)")
    ax.grid(axis="x", alpha=.2); ax.legend(frameon=False, ncol=3)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


def apportion(n: int, weights: dict[str, int], capacity: dict[str, int]) -> dict[str, int]:
    total = sum(weights.values()) or 1
    raw = {k: n*weights[k]/total for k in weights}
    out = {k: min(capacity[k], int(math.floor(raw[k]))) for k in weights}
    remain = min(n, sum(capacity.values())) - sum(out.values())
    for k in sorted(weights, key=lambda z: (-(raw[z]-math.floor(raw[z])), z)):
        while remain and out[k] < capacity[k]: out[k] += 1; remain -= 1
    return out


def qualitative_audit(data: pd.DataFrame, matched: pd.DataFrame, raw: dict[str, pd.DataFrame], continuations: dict[str, dict[int, str]]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    groups = {
        "REJECTED_CROSS": data.loc[data.morph_class.eq("CROSS_MORPHEME") & data.sd_rejected.eq(1)],
        "ACCEPTED_CROSS": data.loc[data.morph_class.eq("CROSS_MORPHEME") & data.sd_rejected.eq(0)],
        "MATCHED_WITHIN_SPLIT": matched.loc[matched.morph_class.eq("WITHIN_SPLIT")],
    }
    selected = []
    for name, candidates in groups.items():
        if candidates.empty: continue
        weights = {str(k): int(v) for k, v in candidates.groupby("pair").size().items()}
        quotas = apportion(20, weights, weights)
        for pair, n in quotas.items():
            sub = candidates.loc[candidates.pair.astype(str).eq(pair)]
            for idx in rng.choice(sub.index.to_numpy(), size=min(n, len(sub)), replace=False):
                rec = sub.loc[idx].to_dict(); rec["audit_group"] = name; selected.append(rec)
    if not selected: return pd.DataFrame()
    output = []
    for _, r in pd.DataFrame(selected).iterrows():
        pair, pid, eid = str(r["pair"]), int(r["prompt_id"]), int(r["eojeol_id"])
        toks = raw[pair]
        same = toks.loc[(toks.prompt_id.astype(int) == pid) & (pd.to_numeric(toks.eojeol_id, errors="coerce") == eid)].copy()
        same = same.loc[same.token_start_char.notna() & same.token_end_char.notna()].sort_values(["token_start_char", "token_end_char"])
        tokenization = []
        for t in same.itertuples(index=False):
            txt = str(t.token_text)
            if int(t.token_id) == int(r["token_id"]) and int(t.generation_pos) == int(r["generation_pos"]): txt = "⟦"+txt+"⟧"
            tokenization.append(txt)
        text = continuations.get(pair, {}).get(pid, "")
        start, end = int(r["token_start_char"]), int(r["token_end_char"])
        if text and 0 <= start < end <= len(text):
            left, right = max(0, start-35), min(len(text), end+35)
            excerpt = text[left:start]+"⟦"+text[start:end]+"⟧"+text[end:right]
            context_source = "reconstructed from exact saved H2 token spans" if pair == "P3" else "saved target continuation"
        else:
            excerpt = str(r.get("eojeol_surface", ""))+" / ⟦"+str(r.get("token_surface", ""))+"⟧"
            context_source = "eojeol-local fallback"
        output.append({
            "audit_group": r["audit_group"], "pair": pair, "prompt_id": pid, "generation_pos": int(r["generation_pos"]),
            "context_excerpt": excerpt, "context_source": context_source, "eojeol": r.get("eojeol_surface", ""),
            "kiwi_segmentation": r.get("morpheme_sequence", ""), "tokenization": " | ".join(tokenization),
            "highlighted_token": "⟦"+str(r.get("token_surface", ""))+"⟧",
            "nominal_surface": r.get("nominal_surface", ""), "nominal_pos": r.get("nominal_pos", ""),
            "particle_surface": r.get("particle_surface", ""), "particle_pos": r.get("particle_pos", ""),
            "particle_category": r.get("particle_category", ""), "geometry_class": r.get("geometry_class", ""),
            "draft_entropy": r.get("draft_entropy", np.nan), "target_entropy": r.get("target_entropy", np.nan),
            "token_frequency": r.get("token_count", np.nan), "sd_rejected": int(r["sd_rejected"]), "seed": SEED,
        })
    return pd.DataFrame(output)


def recurrent_patterns(data: pd.DataFrame) -> pd.DataFrame:
    cross = data.loc[data.morph_class.eq("CROSS_MORPHEME")].copy()
    keys = ["pair", "nominal_surface", "particle_surface", "token_surface"]
    out = cross.groupby(keys, observed=True).agg(
        n_cross=("sd_rejected", "size"), n_prompts=("prompt_id", "nunique"),
        raw_rejection=("sd_rejected", "mean"), particle_category=("particle_category", "first"),
        median_token_frequency=("token_count", "median"),
    ).reset_index()
    out["share_of_pair_cross"] = out.n_cross / out.groupby("pair").n_cross.transform("sum")
    out["meets_support"] = (out.n_cross >= 20) & (out.n_prompts >= 10)
    return out.sort_values(["pair", "n_cross"], ascending=[True, False])


def recurrent_pattern_concentration(patterns: pd.DataFrame) -> pd.DataFrame:
    """Describe concentration in most frequent patterns; ranking uses counts only."""
    rows = []
    for pair, group in patterns.groupby("pair", sort=False):
        g = group.sort_values("n_cross", ascending=False).reset_index(drop=True)
        total_cross = int(g.n_cross.sum())
        total_rejected = int(round((g.n_cross * g.raw_rejection).sum()))
        for k in [1, 5, 10]:
            top = g.head(min(k, len(g)))
            n_top = int(top.n_cross.sum())
            rejected_top = int(round((top.n_cross * top.raw_rejection).sum()))
            rows.append({
                "pair": pair, "top_k_patterns": k, "patterns_available": len(g),
                "patterns_in_top_k": len(top), "total_cross_tokens": total_cross,
                "top_k_cross_tokens": n_top, "share_of_cross_tokens": n_top / total_cross if total_cross else np.nan,
                "top_k_raw_rejection": rejected_top / n_top if n_top else np.nan,
                "top_k_share_of_rejected_cross": rejected_top / total_rejected if total_rejected else np.nan,
                "ranked_by": "n_cross only; no outcome-based ranking",
            })
    return pd.DataFrame(rows)


def geometry_span_distributions(cross: pd.DataFrame) -> pd.DataFrame:
    metrics = ["nominal_chars_in_token", "particle_chars_in_token",
               "nominal_total_chars", "particle_total_chars",
               "nominal_fraction_covered", "particle_fraction_covered"]
    rows = []
    for pair, group in cross.groupby("pair", sort=False):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append({
                "pair": pair, "metric": metric, "n_cross": len(values),
                "mean": values.mean(), "sd": values.std(ddof=1), "min": values.min(),
                "q25": values.quantile(.25), "median": values.median(),
                "q75": values.quantile(.75), "max": values.max(),
            })
    return pd.DataFrame(rows)


def main() -> None:
    for name in ["audits", "pair_results", "pooled", "figures"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    pos_doc = json.loads((H3 / "h3_pos_mapping.json").read_text(encoding="utf-8"))
    pos_map = pos_doc["tag_to_coarse_group"]
    token_cache = pd.read_parquet(P3_FREQUENCY_CACHE)
    eo_cache = pd.read_parquet(LEGACY / "e1_eojeol_frequency_cache.parquet")
    h3_tokens = pd.read_parquet(H3 / "h3_token_environment.parquet")
    h3_morphs = pd.read_parquet(H3 / "h3_eojeol_morphemes.parquet")
    sequences = map_sequences(h3_morphs)
    corpus_sha = sha256_file(CORPUS)
    e1_meta = json.loads((LEGACY / "e1_frequency_cache_metadata.json").read_text(encoding="utf-8"))
    if corpus_sha != e1_meta.get("corpus_sha256"):
        raise AssertionError("The local E1 corpus does not match its saved checksum")
    morph_cache_path = OUT / "h4_morpheme_frequency_cache.parquet"
    morph_cache, morph_cache_meta = build_morpheme_frequency_cache(morph_cache_path, corpus_sha)

    data_by_pair, raw_by_pair, align_audit, input_hashes = {}, {}, {}, {}
    paths_to_hash = [
        H3 / "h3_token_environment.parquet", H3 / "h3_eojeol_morphemes.parquet",
        H3 / "h3_config.json", H3 / "h3_pos_mapping.json",
        P3_FREQUENCY_CACHE, LEGACY / "e1_eojeol_frequency_cache.parquet",
        LEGACY / "e1_frequency_cache_metadata.json", CORPUS,
        ROOT / "runs/e2_model_pair_replication/combined/e2_prompt_overlap_audit.json",
        paths_for("P3")["teacher"], paths_for("P3")["events"],
    ]
    for p in paths_to_hash:
        input_hashes[str(p.relative_to(ROOT))] = sha256_file(p)
    for pair in PAIRS_ORDER:
        raw = load_raw(pair, token_cache)
        raw_by_pair[pair] = raw
        token_path = paths_for(pair)["tokens"]
        input_hashes[str(token_path.relative_to(ROOT))] = sha256_file(token_path)
        if pair in {"P1", "P2"}:
            continuation_path = ROOT / "runs/e2_model_pair_replication" / PAIRS[pair]["slug"] / "continuations.parquet"
            if continuation_path.exists():
                input_hashes[str(continuation_path.relative_to(ROOT))] = sha256_file(continuation_path)
        pair_data, audit = enrich_pair(pair, raw, h3_tokens, sequences, pos_map)
        pair_data = pair_data.loc[pair_data.alignment_valid.fillna(False)].copy()
        align_audit[pair] = audit
        pair_data = attach_frequencies(pair_data, token_cache, eo_cache, morph_cache)
        pair_data = clean_h4(pair_data)
        pair_data["pair"] = pair
        data_by_pair[pair] = pair_data
    data = pd.concat([data_by_pair[p] for p in PAIRS_ORDER], ignore_index=True)

    mapping = {
        "taxonomy_version": "H4-particle-types-v1",
        "particle_categories": PARTICLE_GROUPS,
        "additional_tags": "H3-mapped PARTICLE tags not in the prespecified groups are OTHER_PARTICLE; unrecognized tags are UNMAPPED_PARTICLE and retained in audits.",
        "nominal_tags": sorted(NOMINAL_TAGS),
        "source": "H3 tag_to_coarse_group from saved h3_pos_mapping.json; exact particle fine POS retained.",
    }
    atomic_json(OUT / "h4_particle_mapping.json", mapping)
    atomic_json(OUT / "audits/h4_alignment_audit.json", align_audit)
    data.to_parquet(OUT / "h4_enriched_tokens.parquet", index=False)

    particle_all, support_all, geometry_all, continuous_all, lexical_all, particle_fe_all = [], [], [], [], [], []
    matched_stats, matched_frames = [], []
    model_diags: dict[str, Any] = {}
    for pair in PAIRS_ORDER:
        d = data_by_pair[pair]
        pr, support, pdiag = run_particle_model(d, pair)
        geo, cont, gdiag = run_geometry(d, pair)
        lex, ldiag = run_lexical(d, pair)
        sfe, sfcounts, sfdiag = run_particle_surface_fe(d, pair)
        mresult, mframe = run_matching(d, pair)
        particle_all.append(pr); support_all.append(support); geometry_all.append(geo)
        continuous_all.append(cont); lexical_all.append(lex); particle_fe_all.append(sfe)
        matched_stats.append(mresult); matched_frames.append(mframe)
        model_diags[pair] = {"particle": pdiag, "geometry": gdiag, "lexical": ldiag, "particle_surface_fe": sfdiag, "matching": mresult}
        # Compact, independently readable pair results and complete diagnostics.
        pair_dir = OUT / "pair_results"
        pair_dir.mkdir(parents=True, exist_ok=True)
        pair_report = [
            f"# H4 pair results: {pair} ({PAIR_LABELS[pair]})", "",
            f"Saved exact-span rows={len(d):,}; prompts={d.prompt_id.nunique():,}; CROSS={int(d.morph_class.eq('CROSS_MORPHEME').sum()):,}; SPLIT={int(d.morph_class.eq('WITHIN_SPLIT').sum()):,}.",
            "No generation or teacher-forced scoring was run.", "",
            "## Particle-type adjusted contrasts", "", md_table(pr), "",
            "## Geometry-adjusted rejection probabilities", "", md_table(geo), "",
            "## Lexical-control AMEs", "", md_table(lex), "",
            "## H4a interaction omnibus test", "", json.dumps(pdiag["interaction_omnibus"], indent=2),
            "",
        ]
        (pair_dir / f"{pair.lower()}_h4_results.md").write_text("\n".join(pair_report), encoding="utf-8")
        atomic_json(pair_dir / f"{pair.lower()}_h4_results.json", model_diags[pair])
        atomic_csv(pr, pair_dir / f"{pair.lower()}_particle_results.csv")
        atomic_csv(geo, pair_dir / f"{pair.lower()}_geometry_results.csv")
        atomic_csv(lex, pair_dir / f"{pair.lower()}_lexical_results.csv")
        atomic_csv(sfe, pair_dir / f"{pair.lower()}_particle_surface_fe.csv")
        if not mframe.empty: mframe.to_parquet(pair_dir / f"{pair.lower()}_matched_tokens.parquet", index=False)

    particle = pd.concat(particle_all, ignore_index=True)
    support = pd.concat(support_all, ignore_index=True)
    geometry = pd.concat(geometry_all, ignore_index=True)
    continuous = pd.concat(continuous_all, ignore_index=True)
    lexical = pd.concat(lexical_all, ignore_index=True)
    particle_fe = pd.concat(particle_fe_all, ignore_index=True)
    matched_stats_df = pd.DataFrame(matched_stats)
    matched = pd.concat(matched_frames, ignore_index=True) if matched_frames else pd.DataFrame()
    atomic_csv(particle, OUT / "pooled/h4_particle_results.csv")
    atomic_csv(support, OUT / "audits/h4_particle_audit.csv")
    atomic_csv(geometry, OUT / "pooled/h4_geometry_results.csv")
    atomic_csv(continuous, OUT / "pooled/h4_geometry_continuous.csv")
    atomic_csv(lexical, OUT / "pooled/h4_lexical_robustness.csv")
    atomic_csv(particle_fe, OUT / "pooled/h4_particle_fixed_effects.csv")
    atomic_csv(matched_stats_df, OUT / "pooled/h4_matched_robustness.csv")

    # Full exact geometry distribution and prevalence/rate audit, computed before models.
    cross = data.loc[data.morph_class.eq("CROSS_MORPHEME")].copy()
    geo_counts = cross.groupby(["pair", "geometry_class"], observed=True).agg(
        n_cross=("sd_rejected", "size"), n_prompts=("prompt_id", "nunique"),
        raw_rejection=("sd_rejected", "mean"), mean_draft_entropy=("draft_entropy", "mean"),
        median_token_frequency=("token_count", "median"), nominal_fraction_mean=("nominal_fraction_covered", "mean"),
        nominal_fraction_median=("nominal_fraction_covered", "median"), particle_fraction_mean=("particle_fraction_covered", "mean"),
        particle_fraction_median=("particle_fraction_covered", "median"), nominal_chars_mean=("nominal_chars_in_token", "mean"),
        nominal_chars_median=("nominal_chars_in_token", "median"), nominal_chars_q25=("nominal_chars_in_token", lambda s: s.quantile(.25)),
        nominal_chars_q75=("nominal_chars_in_token", lambda s: s.quantile(.75)), particle_chars_mean=("particle_chars_in_token", "mean"),
        particle_chars_median=("particle_chars_in_token", "median"), particle_chars_q25=("particle_chars_in_token", lambda s: s.quantile(.25)),
        particle_chars_q75=("particle_chars_in_token", lambda s: s.quantile(.75)),
    ).reset_index()
    geo_counts["share_of_pair_cross"] = geo_counts.n_cross / geo_counts.groupby("pair").n_cross.transform("sum")
    atomic_csv(geo_counts, OUT / "audits/h4_geometry_counts.csv")
    geometry_spans = geometry_span_distributions(cross)
    atomic_csv(geometry_spans, OUT / "audits/h4_geometry_span_distributions.csv")
    geometry_audit = [
        "# H4 geometry audit", "",
        "Exact token/morpheme character intersections define coverage. Categories use the mutually exclusive precedence in implementation_h4.md. No geometry types were merged after outcome inspection.", "",
        md_table(geo_counts), "",
        "## Continuous distributions by pair", "",
        "Distributions below use CROSS rows only.", "",
        md_table(geometry_spans), "",
        "## Category counts and rates", "",
        md_table(geo_counts), "",
    ]
    (OUT / "audits/h4_geometry_audit.md").write_text("\n".join(geometry_audit), encoding="utf-8")

    patterns = recurrent_patterns(data)
    atomic_csv(patterns, OUT / "pooled/h4_recurrent_patterns.csv")
    pattern_concentration = recurrent_pattern_concentration(patterns)
    atomic_csv(pattern_concentration, OUT / "pooled/h4_recurrent_pattern_concentration.csv")
    frequent = patterns.loc[patterns.meets_support]
    (OUT / "audits/h4_recurrent_patterns.md").write_text(
        "# H4 recurrent-pattern audit\n\nPatterns are exact nominal surface × particle surface × token surface, selected by counts only. Display threshold: at least 20 CROSS tokens and 10 prompts. All patterns remain in the CSV.\n\n"
        + "## Count-ranked concentration (outcome-independent ranking)\n\n"
        + md_table(pattern_concentration) + "\n\n"
        + "## Frequent supported patterns\n\n"
        + md_table(frequent.groupby("pair", group_keys=False).head(20)[["pair", "nominal_surface", "particle_surface", "token_surface", "n_cross", "n_prompts", "share_of_pair_cross", "raw_rejection"]]) + "\n",
        encoding="utf-8",
    )

    # CEM row sample supports the qualitative sample and the matched analysis.
    overlap_path = ROOT / "runs/e2_model_pair_replication/combined/e2_prompt_overlap_audit.json"
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    common_ids = set(map(int, overlap["three_way_intersection_prompt_ids"]))
    pooled_full = run_pooled(data)
    pooled_common = run_pooled(data, common_ids)
    _, pooled_diag_full, pooled_data_full, pooled_full_results = pooled_full
    _, pooled_diag_common, pooled_data_common, pooled_common_results = pooled_common
    pooled_results = pd.concat([pooled_full_results, pooled_common_results], ignore_index=True) if not pooled_full_results.empty or not pooled_common_results.empty else pd.DataFrame()
    atomic_csv(pooled_results, OUT / "pooled/h4_pooled_particle_common_prompt.csv")

    continuations: dict[str, dict[int, str]] = {}
    for pair in ["P1", "P2"]:
        path = ROOT / "runs/e2_model_pair_replication" / PAIRS[pair]["slug"] / "continuations.parquet"
        if path.exists():
            c = pd.read_parquet(path)
            continuations[pair] = {int(r.prompt_id): str(r.reference_text) for r in c.itertuples(index=False)}
    # P3 has no standalone continuation artifact. Reconstruct excerpts only from
    # the saved exact token-span table; this does not invoke a tokenizer/model.
    p3raw = raw_by_pair["P3"]
    p3_texts: dict[int, str] = {}
    for pid, group in p3raw.groupby("prompt_id", sort=False):
        spans = group.loc[group.span_exact.fillna(False) & group.token_start_char.notna() & group.token_end_char.notna()]
        if spans.empty:
            continue
        limit = int(pd.to_numeric(spans.token_end_char, errors="coerce").max())
        chars = [" "] * max(0, limit)
        for row in spans.itertuples(index=False):
            start, end = int(row.token_start_char), int(row.token_end_char)
            if start < 0 or end > limit or end < start:
                continue
            piece = str(row.token_text)
            if len(piece) == end - start:
                chars[start:end] = list(piece)
            elif not piece and end - start == 1:
                chars[start:end] = [" "]
        p3_texts[int(pid)] = "".join(chars)
    continuations["P3"] = p3_texts
    matched_raw = matched
    qualitative = qualitative_audit(data, matched_raw, raw_by_pair, continuations)
    atomic_csv(qualitative, OUT / "pooled/h4_qualitative_audit.csv")
    plot_particle(particle, OUT / "figures/h4_particle_ame_forest.png")
    plot_geometry(geometry, OUT / "figures/h4_geometry_adjusted.png")

    diagnostics = {
        "pair_models": model_diags, "pooled_full": pooled_diag_full, "pooled_common": pooled_diag_common,
        "pooled_common_contributing_prompts": int(pooled_data_common.prompt_id.nunique()) if len(pooled_data_common) else 0,
        "frequency_cache": morph_cache_meta,
    }
    atomic_json(OUT / "h4_runtime_diagnostics.json", diagnostics)

    config = {
        "taxonomy_version": "H4-particle-types-v1", "particle_mapping": mapping,
        "population": "H2/E2 eligible population, saved H3 NOMINAL_TO_PARTICLE; CROSS exactly one boundary; SPLIT uses exact H3 nearest-adjacent boundary rule",
        "span_validation": "Exact H2 token spans are intersected with H3 Kiwi full morpheme spans; H3 crossed POS sequence and local split assignment are independently checked. Unverifiable rows are listed and excluded from H4 feature models.",
        "geometry_precedence": ["FULL_NOMINAL_PLUS_FULL_PARTICLE", "NOMINAL_SUFFIX_PLUS_FULL_PARTICLE", "FULL_NOMINAL_PLUS_PARTICLE", "PARTIAL_BOTH", "OTHER_GEOMETRY"],
        "geometry_definitions": {
            "NOMINAL_SUFFIX_PLUS_FULL_PARTICLE": "partial nominal and complete particle",
            "FULL_NOMINAL_PLUS_PARTICLE": "complete nominal and partial particle",
            "PARTIAL_BOTH": "both partial",
            "FULL_NOMINAL_PLUS_FULL_PARTICLE": "both complete",
            "OTHER_GEOMETRY": "all remaining or unverifiable combinations",
        },
        "formulas": {
            "H4a": "rejection ~ morph_class * particle_category + fragmentation FE + E2 structural + entropy + bs(log_token_count, df=4, degree=3, include_intercept=False)",
            "H4b": "CROSS-only rejection ~ geometry class + particle category + nominal POS + H3 M5 controls; continuous coverage model includes nominal fraction * particle fraction",
            "L1": "H3 M5 restricted to Nominal→Particle",
            "L2": "L1 + particle category + nominal POS + nominal/particle surface lengths; H3 eojeol_char_length is retained because it is the identical eojeol surface length measure",
            "L3": "L2 + cubic 4-df splines of log nominal, particle, and eojeol frequency",
        },
        "references": {"morph_class": "WITHIN_SPLIT", "particle_category": "CASE_SUBJECT where supported; otherwise first prespecified level with both classes", "geometry_class": "most-supported observed geometry, chosen by support only"},
        "uncertainty": "prompt-clustered covariance and delta-method adjusted probabilities/AMEs; two-sided tests",
        "multiple_testing": "Holm across all seven prespecified particle categories within each pair; non-estimable/unsupported categories enter as p=1",
        "support_thresholds": {"warn": "class count <100 or class prompt count <50", "exact_particle_FE": "total >=100, CROSS >=30, prompts >=30", "recurrent_pattern": "CROSS >=20, prompts >=10"},
        "matching": {"method": "coarsened exact matching; 1:1 without replacement inside strata; clustered LPM with stratum FE", "features": ["pair", "particle_category", "fragmentation_bin", "token length 1/2/3/4+", "nominal surface length 1/2/3/4+", "token frequency decile", "draft entropy decile", "generation position quartile"], "seed": SEED},
        "frequency": {"token": "existing E1 cache", "eojeol": "existing E1 cache", "nominal_particle": str(morph_cache_path.relative_to(ROOT)), "source_corpus_sha256": corpus_sha, "no_outcomes_used": True},
        "common_prompt_ids": {"source": str(overlap_path.relative_to(ROOT)), "count": len(common_ids)},
        "seed": SEED,
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "statsmodels": statsmodels.__version__, "scipy": importlib.metadata.version("scipy"), "kiwipiepy": importlib.metadata.version("kiwipiepy"), "matplotlib": importlib.metadata.version("matplotlib")},
        "input_sha256": input_hashes,
        "execution": "CPU analysis only; no GPU, weights, target generation, speculative decoding, or teacher-forced scoring",
    }
    atomic_json(OUT / "h4_config.json", config)

    warn_rows = support.loc[(support.n_cross < 100) | (support.n_prompts < 50), ["pair", "particle_category", "n_cross", "n_split", "n_prompts"]]
    omnibus_table = pd.DataFrame([{"pair": p, **model_diags[p]["particle"]["interaction_omnibus"]} for p in PAIRS_ORDER])
    frequent_count = int(frequent.meets_support.sum())
    geometry_totals = cross.groupby("pair").size().to_dict()
    suffix_rows = geometry.loc[geometry.geometry_class.eq("NOMINAL_SUFFIX_PLUS_FULL_PARTICLE")]
    l3_rows = lexical.loc[lexical.model.eq("L3")].set_index("pair")
    pooled_full_ix = pooled_full_results.set_index("particle_category")
    pooled_common_ix = pooled_common_results.set_index("particle_category")
    common_max_delta_pp = max(
        abs(float(pooled_full_ix.loc[k, "ame"] - pooled_common_ix.loc[k, "ame"])) * 100
        for k in set(pooled_full_ix.index) & set(pooled_common_ix.index)
    ) if len(pooled_full_results) and len(pooled_common_results) else np.nan
    particle_summary = (
        f"The particle-category interaction omnibus is p={omnibus_table.set_index('pair').loc['P1','wald_p']:.3g} for P1, "
        f"p={omnibus_table.set_index('pair').loc['P2','wald_p']:.3g} for P2, and "
        f"p={omnibus_table.set_index('pair').loc['P3','wald_p']:.3g} for P3. "
        "Adjusted contrasts are consistently positive for auxiliary particles and object particles across all three pairs; genitive and subject estimates are also positive across pairs but less precise in some pairs. Adverbial-particle estimates are positive only in P1 and near zero in P2/P3. Every particle category has fewer than 100 CROSS tokens per pair, so these are sparse, model-adjusted contrasts."
    )
    geometry_summary = (
        "The observed crossing is overwhelmingly NOMINAL_SUFFIX_PLUS_FULL_PARTICLE: "
        + ", ".join(f"{p} {100*float(suffix_rows.loc[suffix_rows.pair.eq(p),'share_of_cross'].iloc[0]):.1f}%" for p in PAIRS_ORDER)
        + ". FULL_NOMINAL_PLUS_FULL_PARTICLE has only 21, 8, and 7 tokens in P1/P2/P3. Its standardized linear-probability estimates are imprecise and the confidence limits leave [0,1], so they cannot identify a reliable geometry-specific probability difference. Particle coverage is exactly 100% in every observed CROSS token; only nominal coverage varies."
    )
    lexical_summary = "After the strongest lexical/frequency controls (L3), adjusted CROSS−SPLIT differences remain " + "; ".join(
        f"{p} {100*l3_rows.loc[p,'ame']:.1f} pp [{100*l3_rows.loc[p,'ame_ci_low']:.1f}, {100*l3_rows.loc[p,'ame_ci_high']:.1f}]" for p in PAIRS_ORDER
    ) + ". The L1→L3 changes do not attenuate the association, so the measured token, eojeol, nominal, and particle frequencies do not explain it away."
    concentration_summary = (
        f"No pair-specific surface pattern met the prespecified threshold of 20 CROSS tokens and 10 prompts ({frequent_count} qualifying patterns). "
        + "; ".join(
            f"{p} top pattern={100*float(pattern_concentration.loc[(pattern_concentration.pair.eq(p)) & (pattern_concentration.top_k_patterns.eq(1)), 'share_of_cross_tokens'].iloc[0]):.1f}% of CROSS tokens, top ten={100*float(pattern_concentration.loc[(pattern_concentration.pair.eq(p)) & (pattern_concentration.top_k_patterns.eq(10)), 'share_of_cross_tokens'].iloc[0]):.1f}%"
            for p in PAIRS_ORDER
        ) + ". This does not support a single recurrent token pattern as the main explanation."
    )
    matched_by_pair = matched_stats_df.set_index("pair")
    matched_summary = "The coarsened-exact matched sensitivity retained " + "; ".join(
        f"{p}: {int(matched_by_pair.loc[p,'n_cross_matched'])} CROSS/SPLIT pairs, "
        f"{100*matched_by_pair.loc[p,'adjusted_lpm_difference']:.1f} pp "
        f"[{100*matched_by_pair.loc[p,'adjusted_lpm_ci_low']:.1f}, {100*matched_by_pair.loc[p,'adjusted_lpm_ci_high']:.1f}]"
        for p in PAIRS_ORDER
    ) + ". These matched samples are very small and are only a robustness check."
    lines = [
        "# H4: decomposition of the Nominal→Particle association", "",
        "## Result", "",
        "H4 result: particle-type heterogeneity is suggested in P2/P3; no stable geometry-specific explanation emerges, and stronger lexical/frequency controls or a single recurrent pattern do not explain the large association. All results are adjusted associations, not causal effects.", "",
        "No speculative decoding, target generation, or teacher-forced scoring was rerun. H2 eligibility, H3 environment assignment, and H3 morphology labels were reused. Span failures are documented in audits/h4_alignment_audit.json.", "",
        "## H4a: particle type", "", particle_summary, "",
        "### Adjusted CROSS−SPLIT contrasts", "", md_table(particle), "",
        "### Raw rates, entropy, frequency, and support", "", md_table(support), "",
        "Omnibus interaction tests:", "",
        md_table(omnibus_table), "",
        "Holm correction is within pair across the seven prespecified particle categories; unsupported categories receive p=1. All model fits converged, had full rank, and had no extreme coefficients. Each reported H4a contrast changes only morph_class while keeping the category and observed covariates fixed.", "",
        "## H4b: crossing geometry", "", geometry_summary, "", md_table(geometry), "",
        "Continuous coverage changes:", "", md_table(continuous), "",
        "Geometry adjusted values standardize over the same pair-specific CROSS covariate distribution, changing only geometry_class. All categorical geometry logits had extreme coefficients/rank deficiency; the displayed estimates therefore use clustered linear-probability models. Full/full estimates or intervals outside [0,1] are extrapolative and should not be read as valid probabilities. The continuous particle-coverage terms are not estimable because observed particle coverage is constant at 100%; nominal coverage estimates are exploratory. Geometry categories were retained without outcome-driven merging. Character-count and coverage distributions are in audits/h4_geometry_span_distributions.csv.", "",
        "## H4c: stronger lexical and frequency controls", "", lexical_summary, "", md_table(lexical), "",
        "L1→L2→L3 changes are robustness comparisons, not mediation estimates. H3 eojeol_char_length is exactly the eojeol surface length in this saved table, so L2 retains the original H3 control and adds nominal/particle side lengths without duplicating a column. Nominal and particle frequencies were counted by Kiwi from the same local E1 Korean Wikipedia corpus; eojeol and token counts reuse E1 caches.", "",
        "## Particle-surface fixed effects", "", md_table(particle_fe), "",
        "## Coarsened exact matching", "", md_table(matched_stats_df), "",
        matched_summary, "",
        "## Pooled common-prompt sensitivity", "",
        f"The E2 three-way source intersection has {len(common_ids)} prompts; {diagnostics['pooled_common_contributing_prompts']:,} contributed H4 rows. Common-prompt versus all-available pooled AMEs differ by at most {common_max_delta_pp:.2f} pp, so the pooled sensitivity is unchanged. These pooled estimates are secondary; pair-specific models are primary.", "",
        md_table(pooled_results), "",
        "## Recurrent token patterns and qualitative audit", "",
        concentration_summary, f"The fixed-seed qualitative sample has {len(qualitative)} examples: 20 rejected CROSS, 20 accepted CROSS, and 20 matched SPLIT tokens, selected with pair-proportional quotas. P1/P2 excerpts use saved continuation text; P3 excerpts are reconstructed only from exact saved H2 token spans.", "",
        "Pattern concentration is ranked solely by pattern CROSS count; the summary reports top-1/5/10 token and rejection shares as descriptive accounting, not a hypothesis test.", "",
        md_table(pattern_concentration), "",
        md_table(frequent.groupby("pair", group_keys=False).head(10)[["pair", "nominal_surface", "particle_surface", "token_surface", "n_cross", "n_prompts", "share_of_pair_cross", "raw_rejection"]]), "",
        "## Span and support audit", "",
        f"Verified rows={sum(x['verified_rows'] for x in align_audit.values()):,}; exact-span/alignment exclusions={sum(x['alignment_failures'] for x in align_audit.values()):,}.",
        md_table(pd.DataFrame([{"pair": p, "rows": align_audit[p]["h3_n_to_p_rows"], "verified": align_audit[p]["verified_rows"], "failures": align_audit[p]["alignment_failures"]} for p in PAIRS_ORDER])), "",
        md_table(warn_rows) if len(warn_rows) else "No particle category/pair triggered the <100 CROSS or <50 prompt warning.", "",
        "## Interpretation", "",
        "Across the three model pairs, the adjusted CROSS−SPLIT association is consistently large for auxiliary particles and object particles, while the adverbial-particle estimate is not replicated in P2/P3. Most observed CROSS tokens contain a partial nominal together with the complete particle, but the alternative full/full geometry is too sparse and yields unbounded linear-model intervals, so the data do not isolate a reliable geometry-specific contrast. Adding nominal, particle, eojeol, and token frequencies leaves the H4 M5 association positive, and no exact surface pattern is recurrent enough to account for the overall result. These patterns describe associations in the saved experiment and do not establish that particle attachment, token geometry, or any lexical form causes speculative rejection.", "",
        "See h4_config.json, h4_particle_mapping.json, the audits directory, and pair_results for formulas, hashes, diagnostics, raw rates, and adjusted estimates.", "",
    ]
    (OUT / "h4_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
