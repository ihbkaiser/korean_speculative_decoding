"""Token-level H2 analysis for the saved Korean speculative-decoding run."""

from __future__ import annotations

import json
import math
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


FRAGMENTATION_ORDER = ["2", "3", "4", "5", "6", "7", "8+"]
MORPH_CLASSES = ["EXACT", "WITHIN_SPLIT", "CROSS_MORPHEME", "CROSS_EOJEOL", "KIWI_COMPLEX"]
PRIMARY_CLASSES = ["WITHIN_SPLIT", "CROSS_MORPHEME"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _fragmentation_bin(count: int | None) -> str | None:
    if count is None or count < 2:
        return None if count is None else str(count)
    return str(count) if count <= 7 else "8+"


def _overlap(start: int, end: int, other_start: int, other_end: int) -> int:
    return max(0, min(end, other_end) - max(start, other_start))


def _span_tokenizer_ids(tokenizer: Any, full_ids: list[int], prompt_length: int) -> dict[str, Any]:
    """Return exact text offsets for re-tokenizable IDs and fallback spans otherwise."""
    from .alignment import token_ids_to_character_offsets, tokenize_with_offsets

    aligned = token_ids_to_character_offsets(tokenizer, full_ids)
    text = aligned["text"]
    encoded_ids, encoded_offsets = tokenize_with_offsets(tokenizer, text)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    expected_positions = [i for i, token_id in enumerate(full_ids) if token_id not in special_ids]
    expected_ids = [full_ids[i] for i in expected_positions]

    expected_to_encoded: dict[int, int] = {}
    matcher = SequenceMatcher(a=expected_ids, b=encoded_ids, autojunk=False)
    for block in matcher.get_matching_blocks():
        for delta in range(block.size):
            expected_to_encoded[block.a + delta] = block.b + delta

    token_spans: dict[int, tuple[int, int]] = {}
    exact_ids: set[int] = set()
    encoded_index_by_full_position = {full_pos: i for i, full_pos in enumerate(expected_positions)}
    for full_pos in range(prompt_length, len(full_ids)):
        if full_ids[full_pos] in special_ids:
            token_spans[full_pos] = (0, 0)
            continue
        expected_index = encoded_index_by_full_position[full_pos]
        encoded_index = expected_to_encoded.get(expected_index)
        if encoded_index is not None:
            assert encoded_ids[encoded_index] == full_ids[full_pos], (
                f"Retokenized ID mismatch at generated position {full_pos - prompt_length}"
            )
            token_spans[full_pos] = encoded_offsets[encoded_index]
            exact_ids.add(full_pos)
        else:
            # This occurs for byte-fallback fragments that decode to U+FFFD or
            # when truncation leaves an incomplete UTF-8 byte sequence. Keep
            # the generated row auditable, but classify it as ambiguous below.
            token_spans[full_pos] = tuple(aligned["offsets"][full_pos])

    if aligned["roundtrip_exact"]:
        assert encoded_ids == expected_ids, "Exact tokenizer round trip changed generated token IDs"
        assert len(exact_ids) == sum(full_ids[i] not in special_ids for i in range(prompt_length, len(full_ids)))
    return {
        "text": text,
        "spans": token_spans,
        "fallback_offsets": aligned["offsets"],
        "exact_ids": exact_ids,
        "roundtrip_exact": bool(aligned["roundtrip_exact"]),
        "expected_visible_ids": len(expected_ids),
        "retokenized_ids": len(encoded_ids),
        "generated_visible_ids": sum(full_ids[i] not in special_ids for i in range(prompt_length, len(full_ids))),
    }


def _classify_token(
    *,
    span_exact: bool,
    crosses_prompt: bool,
    whitespace_only: bool,
    token_start: int,
    token_end: int,
    eojeols: list[dict[str, Any]],
    morphemes: list[dict[str, Any]],
    bad_morpheme_indices: set[int],
) -> tuple[str, list[int], list[int], str]:
    if not span_exact:
        return "KIWI_COMPLEX", [], [], "token_id_not_retokenized_exactly"
    if crosses_prompt:
        return "KIWI_COMPLEX", [], [], "token_span_crosses_prompt_boundary"
    if whitespace_only or token_end <= token_start:
        return "KIWI_COMPLEX", [], [], "no_nonwhitespace_character_span"

    overlapping_eojeols = [
        int(row["eojeol_id"])
        for row in eojeols
        if _overlap(token_start, token_end, int(row["char_start"]), int(row["char_end"])) > 0
    ]
    if len(overlapping_eojeols) >= 2:
        return "CROSS_EOJEOL", overlapping_eojeols, [], "token_overlaps_multiple_eojeols"
    if len(overlapping_eojeols) != 1:
        return "KIWI_COMPLEX", overlapping_eojeols, [], "token_has_no_eojeol_alignment"

    overlapping_morphemes = [
        i for i, morph in enumerate(morphemes)
        if _overlap(token_start, token_end, morph["start"], morph["end"]) > 0
    ]
    if not overlapping_morphemes:
        return "KIWI_COMPLEX", overlapping_eojeols, [], "token_has_no_morpheme_overlap"
    if any(i in bad_morpheme_indices for i in overlapping_morphemes):
        return "KIWI_COMPLEX", overlapping_eojeols, overlapping_morphemes, "invalid_or_zero_width_kiwi_span"
    for left_idx, left in enumerate(overlapping_morphemes):
        for right in overlapping_morphemes[left_idx + 1:]:
            if _overlap(token_start, token_end, morphemes[left]["start"], morphemes[left]["end"]) and _overlap(
                token_start, token_end, morphemes[right]["start"], morphemes[right]["end"]
            ):
                if _overlap(
                    morphemes[left]["start"], morphemes[left]["end"],
                    morphemes[right]["start"], morphemes[right]["end"],
                ):
                    return "KIWI_COMPLEX", overlapping_eojeols, overlapping_morphemes, "overlapping_kiwi_morpheme_spans"

    union_chars: set[int] = set()
    for morph_idx in overlapping_morphemes:
        morph = morphemes[morph_idx]
        union_chars.update(range(max(token_start, morph["start"]), min(token_end, morph["end"])))
    if len(union_chars) != token_end - token_start:
        return "KIWI_COMPLEX", overlapping_eojeols, overlapping_morphemes, "token_span_not_fully_covered_by_morphemes"

    if len(overlapping_morphemes) >= 2:
        return "CROSS_MORPHEME", overlapping_eojeols, overlapping_morphemes, "overlaps_multiple_morphemes"
    morph = morphemes[overlapping_morphemes[0]]
    if token_start == morph["start"] and token_end == morph["end"]:
        return "EXACT", overlapping_eojeols, overlapping_morphemes, "exact_morpheme_span"
    if token_start >= morph["start"] and token_end <= morph["end"]:
        return "WITHIN_SPLIT", overlapping_eojeols, overlapping_morphemes, "strict_subspan_of_one_morpheme"
    return "KIWI_COMPLEX", overlapping_eojeols, overlapping_morphemes, "token_not_contained_in_one_morpheme"


def _build_token_table(run_dir: Path, prompt_cache: Path):
    import pandas as pd
    import re
    import yaml
    from kiwipiepy import Kiwi
    from transformers import AutoTokenizer

    from .data import encode_prompt, load_prompt_cache

    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["experiment"]
    tokenizer = AutoTokenizer.from_pretrained(config["target_model"], use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("H2 requires a fast tokenizer with character offsets")
    kiwi = Kiwi()
    records = load_prompt_cache(prompt_cache)
    record_by_source = {int(row["source_index"]): row for row in records}
    outputs = _read_jsonl(run_dir / "reference_outputs.jsonl")
    events = pd.read_parquet(run_dir / "sd_events.parquet")
    teacher = pd.read_parquet(run_dir / "teacher_forced_tokens.parquet")
    saved_eojeols = pd.read_parquet(run_dir / "eojeols.parquet")

    valid_events = events[events["rejected"].notna()].copy()
    if valid_events.duplicated(["prompt_id", "output_token_position"]).any():
        raise AssertionError("More than one valid SD event maps to an output token")
    valid_event_by_position = {
        (int(row.prompt_id), int(row.output_token_position)): row
        for row in valid_events.itertuples(index=False)
    }
    teacher_by_position = {
        (int(row.prompt_id), int(row.output_token_position)): row
        for row in teacher.itertuples(index=False)
    }
    saved_by_prompt = {
        int(prompt_id): group.sort_values("eojeol_index")
        for prompt_id, group in saved_eojeols.groupby("prompt_id", sort=False)
    }
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    token_rows: list[dict[str, Any]] = []
    alignment_audit: list[dict[str, Any]] = []

    for output in outputs:
        prompt_id = int(output["prompt_id"])
        source_index = int(output["source_index"])
        if source_index not in record_by_source:
            raise KeyError(f"No cached prompt for source_index={source_index}")
        prompt_ids = encode_prompt(tokenizer, record_by_source[source_index]["text"], int(config["max_prompt_tokens"]))
        generated_ids = [int(x) for x in output["reference_token_ids"]]
        full_ids = prompt_ids + generated_ids
        alignment = _span_tokenizer_ids(tokenizer, full_ids, len(prompt_ids))
        text = alignment["text"]
        prompt_offsets = []
        for full_pos in range(len(prompt_ids)):
            # Prompt IDs are not special in this experiment.
            start, end = alignment["fallback_offsets"][full_pos]
            if end > start:
                prompt_offsets.append((start, end))
        continuation_start = max((end for _, end in prompt_offsets), default=0)
        continuation_text = text[continuation_start:]
        if continuation_text != output["reference_text"]:
            raise AssertionError(f"Decoded continuation differs from saved text for prompt_id={prompt_id}")

        eojeol_matches = list(re.finditer(r"\S+", continuation_text, flags=re.UNICODE))
        eojeols = [
            {"eojeol_id": idx, "text": match.group(0), "char_start": match.start(), "char_end": match.end()}
            for idx, match in enumerate(eojeol_matches)
        ]
        saved = saved_by_prompt.get(prompt_id)
        saved_position_to_eojeol: dict[int, int] = {}
        if saved is not None:
            saved_spans = [
                (int(row.char_start), int(row.char_end), str(row.eojeol))
                for row in saved.itertuples(index=False)
            ]
            rebuilt_spans = [(r["char_start"], r["char_end"], r["text"]) for r in eojeols]
            if saved_spans != rebuilt_spans:
                raise AssertionError(f"Saved eojeol spans do not match reconstructed continuation for prompt_id={prompt_id}")
            for saved_row in saved.itertuples(index=False):
                positions = saved_row.output_token_positions
                if positions is None:
                    continue
                for position in positions:
                    saved_position_to_eojeol[int(position)] = int(saved_row.eojeol_index)

        morphemes: list[dict[str, Any]] = []
        bad_morpheme_indices: set[int] = set()
        for morph in kiwi.tokenize(continuation_text):
            start = int(morph.start)
            end = start + int(morph.len)
            index = len(morphemes)
            morphemes.append({
                "surface": str(morph.form),
                "pos": str(morph.tag),
                "start": start,
                "end": end,
            })
            if start < 0 or end <= start or end > len(continuation_text):
                bad_morpheme_indices.add(index)

        raw_rows: list[dict[str, Any]] = []
        for generation_pos, token_id in enumerate(generated_ids):
            full_pos = len(prompt_ids) + generation_pos
            absolute_start, absolute_end = alignment["spans"][full_pos]
            span_exact = full_pos in alignment["exact_ids"]
            crosses_prompt = absolute_start < continuation_start and absolute_end > 0 and token_id not in special_ids
            local_start = max(0, int(absolute_start) - continuation_start)
            local_end = max(0, int(absolute_end) - continuation_start)
            if local_start > len(continuation_text) or local_end > len(continuation_text):
                span_exact = False
                local_start = min(local_start, len(continuation_text))
                local_end = min(local_end, len(continuation_text))
            if token_id in special_ids:
                token_surface = str(tokenizer.convert_ids_to_tokens(token_id))
                nonwhite_start, nonwhite_end = local_start, local_end
                whitespace_only = True
            else:
                token_surface = continuation_text[local_start:local_end]
                visible_positions = [i for i in range(local_start, local_end) if not continuation_text[i].isspace()]
                whitespace_only = not visible_positions
                nonwhite_start = visible_positions[0] if visible_positions else local_start
                nonwhite_end = visible_positions[-1] + 1 if visible_positions else local_end
            morph_class, overlapping_eojeol_ids, overlapping_morpheme_ids, reason = _classify_token(
                span_exact=span_exact,
                crosses_prompt=crosses_prompt,
                whitespace_only=whitespace_only,
                token_start=nonwhite_start,
                token_end=nonwhite_end,
                eojeols=eojeols,
                morphemes=morphemes,
                bad_morpheme_indices=bad_morpheme_indices,
            )
            if token_id in special_ids:
                morph_class = "KIWI_COMPLEX"
                reason = "special_token_without_visible_character_span"
            if (
                not overlapping_eojeol_ids
                and not whitespace_only
                and local_end > local_start
                and generation_pos in saved_position_to_eojeol
            ):
                # The run's audited incremental decoder already assigned this
                # ambiguous byte-fallback token to an eojeol. Retain that
                # mapping for fragmentation/position controls while excluding
                # the token itself from the primary morphology contrast.
                overlapping_eojeol_ids = [saved_position_to_eojeol[generation_pos]]
            event = valid_event_by_position.get((prompt_id, generation_pos))
            if event is None:
                sd_valid = False
                sd_rejected = None
                draft_entropy = None
                target_entropy = None
                proposal_slot = None
            else:
                sd_valid = True
                sd_rejected = bool(event.rejected)
                draft_entropy = None if pd.isna(event.draft_entropy) else float(event.draft_entropy)
                target_entropy = None if pd.isna(event.target_entropy) else float(event.target_entropy)
                proposal_slot = int(event.proposal_position)
                if int(event.reference_target_token_id) != token_id:
                    raise AssertionError(
                        f"Generated token disagrees with SD reference at prompt={prompt_id}, position={generation_pos}"
                    )
            teacher_event = teacher_by_position.get((prompt_id, generation_pos))
            teacher_disagreement = None if teacher_event is None else bool(teacher_event.draft_target_disagreement)
            raw_rows.append({
                "prompt_id": prompt_id,
                "generation_pos": generation_pos,
                "token_id": token_id,
                "is_special_token": token_id in special_ids,
                "token_text": token_surface,
                "token_start_char": nonwhite_start,
                "token_end_char": nonwhite_end,
                "tokenizer_span_start_char": local_start,
                "tokenizer_span_end_char": local_end,
                "span_exact": bool(span_exact),
                "morph_class": morph_class,
                "classification_reason": reason,
                "overlapping_eojeol_ids": overlapping_eojeol_ids,
                "overlapping_morpheme_ids": overlapping_morpheme_ids,
                "overlapping_morpheme_surfaces": [morphemes[i]["surface"] for i in overlapping_morpheme_ids],
                "overlapping_morpheme_pos": [morphemes[i]["pos"] for i in overlapping_morpheme_ids],
                "overlapping_morpheme_spans": [[morphemes[i]["start"], morphemes[i]["end"]] for i in overlapping_morpheme_ids],
                "overlapping_morpheme_count": len(overlapping_morpheme_ids),
                "eojeol_id": overlapping_eojeol_ids[0] if len(overlapping_eojeol_ids) == 1 else None,
                "sd_valid": sd_valid,
                "sd_rejected": sd_rejected,
                "draft_entropy": draft_entropy,
                "target_entropy": target_entropy,
                "proposal_slot": proposal_slot,
                "teacher_disagreement": teacher_disagreement,
            })

        # Fragmentation and token position are computed from visible tokens in
        # the original continuation, after assigning each token by character
        # overlap. A cross-eojeol token contributes to each touched eojeol's
        # count but keeps its ambiguous class and no singular eojeol_id.
        token_positions_by_eojeol: dict[int, list[int]] = {row["eojeol_id"]: [] for row in eojeols}
        for row_idx, row in enumerate(raw_rows):
            for eojeol_id in row["overlapping_eojeol_ids"]:
                token_positions_by_eojeol[eojeol_id].append(row_idx)
        for eojeol in eojeols:
            positions = sorted(token_positions_by_eojeol[eojeol["eojeol_id"]], key=lambda i: (raw_rows[i]["token_start_char"], raw_rows[i]["generation_pos"]))
            fragmentation = len(positions)
            for token_pos, row_idx in enumerate(positions):
                row = raw_rows[row_idx]
                if row["eojeol_id"] != eojeol["eojeol_id"]:
                    continue
                row["fragmentation"] = fragmentation
                row["fragmentation_bin"] = _fragmentation_bin(fragmentation)
                row["token_pos_in_eojeol"] = token_pos
                row["relative_pos_in_eojeol"] = token_pos / (fragmentation - 1) if fragmentation > 1 else 0.0
                row["first_token"] = token_pos == 0
                row["last_token"] = token_pos == fragmentation - 1
                row["token_char_length"] = max(0, row["token_end_char"] - row["token_start_char"])
                row["eojeol_char_length"] = eojeol["char_end"] - eojeol["char_start"]
        for row in raw_rows:
            if "fragmentation" not in row:
                row.update({
                    "fragmentation": None,
                    "fragmentation_bin": None,
                    "token_pos_in_eojeol": None,
                    "relative_pos_in_eojeol": None,
                    "first_token": None,
                    "last_token": None,
                    "token_char_length": max(0, row["token_end_char"] - row["token_start_char"]),
                    "eojeol_char_length": None,
                })
            row["generation_position"] = row["generation_pos"]
            row["relative_position"] = row["relative_pos_in_eojeol"]
            row["entropy_gap"] = (
                None if row["draft_entropy"] is None or row["target_entropy"] is None
                else row["draft_entropy"] - row["target_entropy"]
            )
            token_rows.append(row)

        alignment_audit.append({
            "prompt_id": prompt_id,
            "generated_tokens": len(generated_ids),
            "generated_visible_ids": alignment["generated_visible_ids"],
            "generated_exact_id_spans": len(alignment["exact_ids"]),
            "retokenization_roundtrip_exact": alignment["roundtrip_exact"],
            "expected_visible_ids": alignment["expected_visible_ids"],
            "retokenized_ids": alignment["retokenized_ids"],
        })

    table = pd.DataFrame(token_rows)
    if table.duplicated(["prompt_id", "generation_pos"]).any():
        raise AssertionError("Token table must contain one row per generated token position")
    expected_tokens = sum(len(row["reference_token_ids"]) for row in outputs)
    if len(table) != expected_tokens:
        raise AssertionError(f"Wrote {len(table)} token rows; expected {expected_tokens}")
    primary_classes = table["morph_class"].isin(["EXACT", "WITHIN_SPLIT", "CROSS_MORPHEME", "CROSS_EOJEOL"])
    if not table.loc[primary_classes, "span_exact"].astype(bool).all():
        raise AssertionError("A token without an exact retokenized ID span entered a non-ambiguous class")
    return table, pd.DataFrame(alignment_audit), len(outputs)


def _fit_clustered_logit(data, outcome: str, formula: str) -> dict[str, Any]:
    import statsmodels.formula.api as smf

    clean = data.dropna(subset=[outcome, "prompt_id"]).copy()
    clean[outcome] = clean[outcome].astype(int)
    clean["fragmentation_bin"] = clean["fragmentation_bin"].astype("category").cat.remove_unused_categories()
    clean["morph_class"] = clean["morph_class"].astype("category").cat.remove_unused_categories()
    base = {
        "formula": formula,
        "n_tokens": int(len(clean)),
        "n_prompts": int(clean["prompt_id"].nunique()),
        "beta_cross": None,
        "beta_split": 0.0,
        "contrast_cross_minus_split": None,
        "std_error": None,
        "odds_ratio": None,
        "ci_low": None,
        "ci_high": None,
        "or_ci_low": None,
        "or_ci_high": None,
        "p_value": None,
        "status": "not_estimable",
    }
    if len(clean) < 10 or clean[outcome].nunique() < 2 or clean["morph_class"].nunique() < 2:
        base["status"] = "not_estimable:insufficient_classes_or_rows"
        return base
    try:
        fitted = smf.logit(formula, data=clean).fit(
            disp=False,
            maxiter=200,
            cov_type="cluster",
            cov_kwds={"groups": clean["prompt_id"], "use_correction": True},
        )
        term = next(
            name for name in fitted.params.index
            if "morph_class" in name and "CROSS_MORPHEME" in name
        )
        beta = float(fitted.params[term])
        interval = fitted.conf_int().loc[term]
        low, high = float(interval.iloc[0]), float(interval.iloc[1])
        p_value = float(fitted.pvalues[term])
        base.update({
            "beta_cross": beta,
            "contrast_cross_minus_split": beta,
            "std_error": float(fitted.bse[term]),
            "odds_ratio": math.exp(beta),
            "ci_low": low,
            "ci_high": high,
            "or_ci_low": math.exp(low),
            "or_ci_high": math.exp(high),
            "p_value": p_value,
            "status": "fit" if fitted.mle_retvals.get("converged", True) else "fit:nonconverged",
            "summary": fitted.summary().as_text(),
        })
    except Exception as exc:
        base["status"] = f"fit_failed:{type(exc).__name__}:{exc}"
    return base


def _fit_clustered_ols(data, outcome: str, formula: str, term_key: str = "morph_class") -> dict[str, Any]:
    import statsmodels.formula.api as smf

    clean = data.dropna(subset=[outcome, "prompt_id"]).copy()
    clean["fragmentation_bin"] = clean["fragmentation_bin"].astype("category").cat.remove_unused_categories()
    clean["morph_class"] = clean["morph_class"].astype("category").cat.remove_unused_categories()
    result: dict[str, Any] = {
        "formula": formula,
        "n_tokens": int(len(clean)),
        "n_prompts": int(clean["prompt_id"].nunique()),
        "coefficient_cross_minus_split": None,
        "std_error": None,
        "ci_low": None,
        "ci_high": None,
        "p_value": None,
        "status": "not_estimable",
    }
    if len(clean) < 10 or clean["morph_class"].nunique() < 2:
        result["status"] = "not_estimable:insufficient_rows_or_classes"
        return result
    try:
        fitted = smf.ols(formula, data=clean).fit(
            cov_type="cluster",
            cov_kwds={"groups": clean["prompt_id"], "use_correction": True},
        )
        term = next(name for name in fitted.params.index if term_key in name and "CROSS_MORPHEME" in name)
        interval = fitted.conf_int().loc[term]
        result.update({
            "coefficient_cross_minus_split": float(fitted.params[term]),
            "std_error": float(fitted.bse[term]),
            "ci_low": float(interval.iloc[0]),
            "ci_high": float(interval.iloc[1]),
            "p_value": float(fitted.pvalues[term]),
            "status": "fit",
            "summary": fitted.summary().as_text(),
        })
    except Exception as exc:
        result["status"] = f"fit_failed:{type(exc).__name__}:{exc}"
    return result


def _primary_formula(controls: list[str], outcome: str = "sd_rejected") -> str:
    terms = [
        "C(morph_class, Treatment(reference='WITHIN_SPLIT'))",
        "C(fragmentation_bin, Treatment(reference='2'))",
    ] + controls
    return outcome + " ~ " + " + ".join(terms)


def _bootstrap_rate_table(data, replicates: int = 2000, seed: int = 3090):
    import numpy as np
    import pandas as pd

    pairs = [(morph, frag) for morph in ["EXACT", "WITHIN_SPLIT", "CROSS_MORPHEME"] for frag in FRAGMENTATION_ORDER]
    prompts = sorted(data["prompt_id"].unique())
    prompt_index = {prompt: i for i, prompt in enumerate(prompts)}
    pair_index = {pair: i for i, pair in enumerate(pairs)}
    successes = np.zeros((len(prompts), len(pairs)), dtype=float)
    totals = np.zeros_like(successes)
    grouped = data.groupby(["prompt_id", "morph_class", "fragmentation_bin"], observed=True)["sd_rejected"].agg(["sum", "count"])
    for (prompt, morph, frag), row in grouped.iterrows():
        idx = pair_index.get((str(morph), str(frag)))
        if idx is not None:
            successes[prompt_index[int(prompt)], idx] = float(row["sum"])
            totals[prompt_index[int(prompt)], idx] = float(row["count"])
    rng = np.random.default_rng(seed)
    rates = np.full((replicates, len(pairs)), np.nan)
    for rep in range(replicates):
        sample = rng.integers(0, len(prompts), size=len(prompts))
        den = totals[sample].sum(axis=0)
        num = successes[sample].sum(axis=0)
        rates[rep] = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    rows = []
    point = data.groupby(["morph_class", "fragmentation_bin"], observed=True)["sd_rejected"].agg(["sum", "count"])
    for morph, frag in pairs:
        if (morph, frag) in point.index:
            num = float(point.loc[(morph, frag), "sum"])
            den = int(point.loc[(morph, frag), "count"])
            point_rate = num / den if den else None
        else:
            num, den, point_rate = 0.0, 0, None
        idx = pair_index[(morph, frag)]
        valid = rates[:, idx][np.isfinite(rates[:, idx])]
        ci_low, ci_high = (np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else (None, None))
        prompts_n = int(data.loc[(data["morph_class"] == morph) & (data["fragmentation_bin"] == frag), "prompt_id"].nunique())
        rows.append({
            "morph_class": morph,
            "fragmentation_bin": frag,
            "n_tokens": den,
            "n_prompts": prompts_n,
            "rejections": int(num),
            "rejection_rate": point_rate,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "bootstrap_replicates": replicates,
        })
    return pd.DataFrame(rows)


def _make_figures(run_dir: Path, rates, forest_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"EXACT": "#6b7280", "WITHIN_SPLIT": "#1976d2", "CROSS_MORPHEME": "#d84315"}
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    x = np.arange(len(FRAGMENTATION_ORDER))
    offsets = {"EXACT": -0.06, "WITHIN_SPLIT": 0.0, "CROSS_MORPHEME": 0.06}
    for morph in ["EXACT", "WITHIN_SPLIT", "CROSS_MORPHEME"]:
        group = rates[rates["morph_class"] == morph].set_index("fragmentation_bin").reindex(FRAGMENTATION_ORDER)
        y = group["rejection_rate"].to_numpy(dtype=float)
        low = group["bootstrap_ci_low"].to_numpy(dtype=float)
        high = group["bootstrap_ci_high"].to_numpy(dtype=float)
        valid = np.isfinite(y) & np.isfinite(low) & np.isfinite(high)
        ax.errorbar(
            x[valid] + offsets[morph], y[valid],
            yerr=np.vstack([y[valid] - low[valid], high[valid] - y[valid]]),
            marker="o", capsize=3, linewidth=1.6, color=colors[morph], label=morph.replace("_", " "),
        )
    ax.set_xticks(x, FRAGMENTATION_ORDER)
    ax.set_xlabel("Tokens per eojeol (fragmentation)")
    ax.set_ylabel("SD rejection rate")
    ax.set_title("Rejection by token–morpheme relation and exact fragmentation")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(run_dir / "h2_rejection_by_morphology_fragmentation.png", dpi=190)
    plt.close(fig)

    ordered = [row for row in forest_rows if row.get("status") == "fit" and row.get("odds_ratio") is not None]
    ordered = list(reversed(ordered))
    fig_height = max(4.2, 0.46 * len(ordered) + 1.4)
    fig, ax = plt.subplots(figsize=(8.2, fig_height), constrained_layout=True)
    y = np.arange(len(ordered))
    for pos, row in zip(y, ordered):
        low, high = row["or_ci_low"], row["or_ci_high"]
        ax.errorbar(
            row["odds_ratio"], pos,
            xerr=[[row["odds_ratio"] - low], [high - row["odds_ratio"]]],
            fmt="o", capsize=3, color="#5e35b1" if row["label"] == "Pooled (M1; fragmentation FE)" else "#00897b",
        )
    labels = [row["label"] for row in ordered]
    ax.set_yticks(y, labels)
    ax.axvline(1.0, color="#555", linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio for CROSS_MORPHEME vs WITHIN_SPLIT (log scale)")
    ax.set_title("Cross-morpheme rejection contrast")
    ax.grid(axis="x", which="both", alpha=0.22)
    fig.savefig(run_dir / "h2_forest_cross_vs_split.png", dpi=190)
    plt.close(fig)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    if isinstance(value, (int,)):
        return f"{value:,}"
    return f"{float(value):.{digits}g}"


def _write_reports(run_dir: Path, table, alignment_audit, n_prompts_total: int, rates, class_counts, fragmentation_summary, models, entropy_models, forest_rows):
    import numpy as np
    import pandas as pd

    ambiguous_n = int(table["morph_class"].isin(["CROSS_EOJEOL", "KIWI_COMPLEX"]).sum())
    ambiguous_reasons = table.loc[table["morph_class"].isin(["CROSS_EOJEOL", "KIWI_COMPLEX"]), "classification_reason"].value_counts().to_dict()
    all_n = int(len(table))
    ambiguous_rate = ambiguous_n / all_n if all_n else float("nan")
    visible_table = table[~table["is_special_token"].astype(bool)]
    exact_n = int(visible_table["span_exact"].sum())
    visible_n = int(len(visible_table))
    unmatched_visible_n = int((~visible_table["span_exact"].astype(bool)).sum())
    special_n = int(table["is_special_token"].astype(bool).sum())
    valid_n = int(table["sd_valid"].sum())
    primary = table[
        table["sd_valid"].astype(bool)
        & table["fragmentation_bin"].isin(FRAGMENTATION_ORDER)
        & table["morph_class"].isin(PRIMARY_CLASSES)
    ].copy()
    primary_rejection = float(primary["sd_rejected"].astype(bool).mean()) if len(primary) else float("nan")
    teacher_valid = table[table["sd_valid"].astype(bool) & table["teacher_disagreement"].notna()]
    teacher_match = float((teacher_valid["sd_rejected"].astype(bool) == teacher_valid["teacher_disagreement"].astype(bool)).mean()) if len(teacher_valid) else None

    summary_lines = [
        "# H2 analysis: cross-morpheme vs within-morpheme split tokens",
        "",
        f"Run: `{run_dir.name}`. This analysis reuses the saved 1,000-prompt run; speculative decoding was not rerun.",
        "",
        "## Data and alignment",
        "",
        f"- Generated token rows reconstructed: **{all_n:,}** across **{n_prompts_total:,} prompts**.",
        f"- SD-valid generated tokens: **{valid_n:,}**; primary contrast tokens: **{len(primary):,}** across **{primary['prompt_id'].nunique():,} prompts**.",
        f"- Visible token spans with exact retokenized ID alignment: **{exact_n:,}/{visible_n:,} ({exact_n / visible_n:.2%})**; **{special_n:,}** special tokens have no visible span.",
        f"- Ambiguous classification exclusions (`CROSS_EOJEOL` or `KIWI_COMPLEX`): **{ambiguous_n:,}/{all_n:,} ({ambiguous_rate:.2%})** of generated tokens.",
        f"- `CROSS_EOJEOL`: **{int((table.morph_class == 'CROSS_EOJEOL').sum()):,}**; `KIWI_COMPLEX`: **{int((table.morph_class == 'KIWI_COMPLEX').sum()):,}**.",
        "- Ambiguity reasons: " + "; ".join(f"`{reason}` {count:,}" for reason, count in sorted(ambiguous_reasons.items())) + ".",
        f"- Visible generated IDs not mapped by exact re-tokenization and retained as `KIWI_COMPLEX`: **{unmatched_visible_n:,}**. Contextual round-trip was exact for **{int(alignment_audit.retokenization_roundtrip_exact.sum()):,}/{len(alignment_audit):,} prompts**.",
        "- Exact fragmentation levels are retained as `2, 3, 4, 5, 6, 7, 8+`; no `5+` pooling was used.",
        "",
        "## Primary result",
        "",
        "All regressions use two-sided Wald tests with standard errors clustered by `prompt_id`. The class coefficient is parameterized as CROSS relative to the WITHIN_SPLIT reference (the SPLIT coefficient is 0 by reference coding); `CROSS - SPLIT` is that same log-odds contrast.",
        "",
        "| Model | CROSS coefficient (SPLIT ref.) | CROSS − SPLIT | OR (95% CI) | p | N tokens | N prompts |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["M1", "M2", "M3"]:
        r = models[name]
        or_text = "NA" if r["odds_ratio"] is None else f"{_fmt(r['odds_ratio'])} ({_fmt(r['or_ci_low'])}, {_fmt(r['or_ci_high'])})"
        summary_lines.append(
            f"| {name} | {_fmt(r['beta_cross'])} (SPLIT=0) | {_fmt(r['contrast_cross_minus_split'])} | {or_text} | {_fmt(r['p_value'])} | {r['n_tokens']:,} | {r['n_prompts']:,} |"
        )
    summary_lines.extend([
        "",
        "| Model | CROSS − SPLIT coefficient | 95% CI | p | N tokens | N prompts |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, r in entropy_models.items():
        ci = "NA" if r["ci_low"] is None else f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]"
        summary_lines.append(f"| {name} | {_fmt(r['coefficient_cross_minus_split'])} | {ci} | {_fmt(r['p_value'])} | {r['n_tokens']:,} | {r['n_prompts']:,} |")

    m1, m2, m3 = (models[name] for name in ["M1", "M2", "M3"])
    entropy_cross = entropy_models["Draft entropy"].get("coefficient_cross_minus_split")
    positive_sig = lambda r: r.get("status", "").startswith("fit") and r.get("beta_cross") is not None and r["beta_cross"] > 0 and r.get("p_value") is not None and r["p_value"] < 0.05
    if not positive_sig(m1):
        classification = "D. No H2 signal: CROSS is not higher than SPLIT in M1."
        rationale = "The unadjusted fragmentation-fixed model does not show a positive, two-sided p<0.05 CROSS-vs-SPLIT contrast."
    elif positive_sig(m2) and positive_sig(m3):
        classification = "A. H2 supported in M2 and M3."
        rationale = "The positive CROSS-vs-SPLIT contrast remains statistically distinguishable from zero after structural and entropy adjustment."
    elif positive_sig(m2) and entropy_cross is not None and entropy_cross > 0:
        classification = "B. Morphology-to-uncertainty candidate."
        rationale = "M2 shows a positive CROSS-vs-SPLIT contrast, the M3 rejection contrast is not significant, and CROSS has a positive adjusted draft-entropy contrast. This is an association pattern, not a causal pathway."
    elif positive_sig(m2):
        classification = "A. H2 supported after M2; M3 does not retain a significant positive contrast."
        rationale = "The positive structural-adjusted contrast is present in M2 but does not persist as a significant positive estimate in M3. The entropy pattern does not meet the positive-coefficient condition for category B."
    else:
        classification = "C. Structural confounding: the positive M1 contrast disappears after M2."
        rationale = "M1 has a positive significant contrast, but the estimate is no longer positive and significant after structural controls in M2."
    summary_lines.extend([
        "",
        "## Exact-fragmentation rates",
        "",
        "Rates below include SD-valid tokens only. Confidence intervals are percentile intervals from 2,000 bootstrap resamples of prompts with replacement.",
        "",
        "| Fragmentation | EXACT | WITHIN_SPLIT | CROSS_MORPHEME |",
        "|---|---:|---:|---:|",
    ])
    for frag in FRAGMENTATION_ORDER:
        cells = []
        for morph in ["EXACT", "WITHIN_SPLIT", "CROSS_MORPHEME"]:
            row = rates[(rates.morph_class == morph) & (rates.fragmentation_bin == frag)].iloc[0]
            value = "NA" if pd.isna(row.rejection_rate) else f"{row.rejection_rate:.1%} [{row.bootstrap_ci_low:.1%}, {row.bootstrap_ci_high:.1%}], n={int(row.n_tokens):,}"
            cells.append(value)
        summary_lines.append(f"| {frag} | {cells[0]} | {cells[1]} | {cells[2]} |")
    summary_lines.extend([
        "",
        "## Result classification",
        "",
        f"**{classification}** {rationale}",
        "",
        "## Secondary checks and limits",
        "",
        f"- Teacher-forced draft/target disagreement is included only as a sanity check; on {len(teacher_valid):,} SD-valid rows with both outcomes, its token-level agreement with the SD rejection label is {teacher_match:.2%} (not an independent replication)." if teacher_match is not None else "- Teacher-forced disagreement was unavailable for the primary rows.",
        f"- Primary token-level rejection rate across CROSS and WITHIN_SPLIT rows: {primary_rejection:.2%}.",
        "- The entropy regressions estimate conditional associations; they do not establish a mechanism or causal mediation.",
        "- The run uses the first 1,000 non-empty Korean Wikipedia articles in dataset order. Results describe this generated continuation sample and should not be read as causal or corpus-wide claims.",
        "",
        "## Files",
        "",
        "- `h2_token_table.parquet` — one row per saved generated token, with Kiwi surface/POS/span overlaps and SD/teacher outcomes.",
        "- `h2_class_counts.csv`, `h2_fragmentation_summary.csv`, `h2_regression.txt`.",
        "- `h2_rejection_by_morphology_fragmentation.png`, `h2_forest_cross_vs_split.png`.",
    ])
    (run_dir / "h2_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    report = [
        "H2 clustered regression report",
        f"Run: {run_dir.name}",
        "Outcome for primary models: sd_rejected; sample: sd_valid=True, fragmentation>=2, morph_class in {CROSS_MORPHEME, WITHIN_SPLIT}.",
        "Cluster covariance: prompt_id. Tests: two-sided Wald. SPLIT is the reference class; beta(CROSS) equals CROSS-SPLIT.",
        f"Ambiguous class exclusions: {ambiguous_n:,}/{all_n:,} ({ambiguous_rate:.4%}) of all generated tokens.",
        f"Visible retokenization mismatch / byte-fallback rows: {unmatched_visible_n:,}; special tokens without spans: {special_n:,}; exact contextual round trips: {int(alignment_audit.retokenization_roundtrip_exact.sum()):,}/{len(alignment_audit):,} prompts.",
        "",
    ]
    formulas = {
        "M1": _primary_formula([]),
        "M2": _primary_formula(["relative_position", "first_token", "last_token", "proposal_slot", "generation_position", "token_char_length", "eojeol_char_length"]),
        "M3": _primary_formula(["relative_position", "first_token", "last_token", "proposal_slot", "generation_position", "token_char_length", "eojeol_char_length", "draft_entropy", "target_entropy"]),
    }
    for name in ["M1", "M2", "M3"]:
        result = models[name]
        report.extend([
            f"{name}: {formulas[name]}",
            f"status={result['status']}; n_tokens={result['n_tokens']}; n_prompts={result['n_prompts']}",
            f"beta_CROSS={result['beta_cross']}; beta_SPLIT=0 [reference]; CROSS-SPLIT={result['contrast_cross_minus_split']}",
            f"OR={result['odds_ratio']}; OR_95_CI=[{result['or_ci_low']}, {result['or_ci_high']}]; beta_95_CI=[{result['ci_low']}, {result['ci_high']}]; two_sided_p={result['p_value']}",
            result.get("summary", ""),
            "",
        ])
    for name, result in entropy_models.items():
        report.extend([
            f"{name}: {result['formula']}",
            f"status={result['status']}; n_tokens={result['n_tokens']}; n_prompts={result['n_prompts']}",
            f"CROSS-SPLIT coefficient={result['coefficient_cross_minus_split']}; 95% CI=[{result['ci_low']}, {result['ci_high']}]; two_sided_p={result['p_value']}",
            result.get("summary", ""),
            "",
        ])
    report.append("Teacher-disagreement sanity check (secondary; formula mirrors M2):")
    report.append(json.dumps(models["Teacher disagreement M2"], indent=2, default=str))
    report.append("\nFragmentation-specific and pooled M1 odds ratios:")
    report.append(pd.DataFrame(forest_rows).to_string(index=False))
    (run_dir / "h2_regression.txt").write_text("\n".join(report), encoding="utf-8")


def analyze_h2(run_dir: str | Path, prompt_cache: str | Path, bootstrap_replicates: int = 2000) -> dict[str, Any]:
    """Reconstruct token spans and run H2 models without decoding any prompts."""
    import pandas as pd

    root = Path(run_dir).resolve()
    cache = Path(prompt_cache).resolve()
    table, alignment_audit, prompt_count = _build_token_table(root, cache)
    table_path = root / "h2_token_table.parquet"
    temporary = table_path.with_suffix(".parquet.tmp")
    table.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(table_path)
    alignment_audit.to_csv(root / "h2_tokenizer_alignment_audit.csv", index=False)

    class_counts = (
        table.groupby("morph_class", observed=False)
        .agg(
            n_tokens=("token_id", "size"),
            n_sd_valid=("sd_valid", "sum"),
            n_prompts=("prompt_id", "nunique"),
        )
        .reindex(MORPH_CLASSES, fill_value=0)
        .reset_index()
    )
    valid_table = table[table["sd_valid"].astype(bool)].copy()
    valid_table["sd_rejected"] = valid_table["sd_rejected"].astype(bool)
    valid_table["n_rejections"] = valid_table["sd_rejected"].astype(int)
    rates_all = (
        valid_table[valid_table["morph_class"].isin(["EXACT", "WITHIN_SPLIT", "CROSS_MORPHEME"]) & valid_table["fragmentation_bin"].isin(FRAGMENTATION_ORDER)]
        .groupby(["morph_class", "fragmentation_bin"], observed=True)
        .agg(n_tokens=("token_id", "size"), n_prompts=("prompt_id", "nunique"), rejections=("n_rejections", "sum"), rejection_rate=("sd_rejected", "mean"))
        .reset_index()
    )
    grid = pd.MultiIndex.from_product(
        [["EXACT", "WITHIN_SPLIT", "CROSS_MORPHEME"], FRAGMENTATION_ORDER],
        names=["morph_class", "fragmentation_bin"],
    ).to_frame(index=False)
    fragmentation_summary = grid.merge(rates_all, how="left", on=["morph_class", "fragmentation_bin"])
    for col in ["n_tokens", "n_prompts", "rejections"]:
        fragmentation_summary[col] = fragmentation_summary[col].fillna(0).astype(int)

    rates_input = valid_table[
        valid_table["morph_class"].isin(["EXACT", "WITHIN_SPLIT", "CROSS_MORPHEME"])
        & valid_table["fragmentation_bin"].isin(FRAGMENTATION_ORDER)
    ].copy()
    rates = _bootstrap_rate_table(rates_input, replicates=bootstrap_replicates)
    fragmentation_summary = fragmentation_summary.merge(
        rates[["morph_class", "fragmentation_bin", "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_replicates"]],
        on=["morph_class", "fragmentation_bin"], how="left",
    )
    class_counts["rejections"] = [
        int(table.loc[table.morph_class == morph, "sd_rejected"].fillna(False).astype(bool).sum()) for morph in class_counts.morph_class
    ]
    class_counts["rejection_rate"] = [
        (r / n if n else None) for r, n in zip(class_counts.rejections, class_counts.n_sd_valid)
    ]

    primary = valid_table[
        valid_table["morph_class"].isin(PRIMARY_CLASSES)
        & valid_table["fragmentation_bin"].isin(FRAGMENTATION_ORDER)
    ].copy()
    primary["generation_position"] = primary["generation_pos"]
    primary["sd_rejected"] = primary["sd_rejected"].astype(int)
    primary["first_token"] = primary["first_token"].astype(int)
    primary["last_token"] = primary["last_token"].astype(int)
    controls = ["relative_position", "first_token", "last_token", "proposal_slot", "generation_position", "token_char_length", "eojeol_char_length"]
    models = {
        "M1": _fit_clustered_logit(primary, "sd_rejected", _primary_formula([])),
        "M2": _fit_clustered_logit(primary, "sd_rejected", _primary_formula(controls)),
        "M3": _fit_clustered_logit(primary, "sd_rejected", _primary_formula(controls + ["draft_entropy", "target_entropy"])),
    }
    models["Teacher disagreement M2"] = _fit_clustered_logit(
        primary.assign(teacher_disagreement=primary["teacher_disagreement"].astype("float")),
        "teacher_disagreement",
        _primary_formula(controls, outcome="teacher_disagreement"),
    )
    entropy_formula = "draft_entropy ~ " + " + ".join([
        "C(morph_class, Treatment(reference='WITHIN_SPLIT'))",
        "C(fragmentation_bin, Treatment(reference='2'))",
    ] + controls)
    entropy_gap_formula = "entropy_gap ~ " + " + ".join([
        "C(morph_class, Treatment(reference='WITHIN_SPLIT'))",
        "C(fragmentation_bin, Treatment(reference='2'))",
    ] + controls)
    entropy_models = {
        "Draft entropy": _fit_clustered_ols(primary, "draft_entropy", entropy_formula),
        "Draft-target entropy gap": _fit_clustered_ols(primary, "entropy_gap", entropy_gap_formula),
    }

    forest_rows: list[dict[str, Any]] = []
    pooled = dict(models["M1"])
    pooled["label"] = "Pooled (M1; fragmentation FE)"
    forest_rows.append(pooled)
    for frag in FRAGMENTATION_ORDER:
        subset = primary[primary["fragmentation_bin"] == frag]
        result = _fit_clustered_logit(
            subset,
            "sd_rejected",
            "sd_rejected ~ C(morph_class, Treatment(reference='WITHIN_SPLIT'))",
        )
        result["label"] = f"Fragmentation {frag}"
        forest_rows.append(result)

    class_counts.to_csv(root / "h2_class_counts.csv", index=False)
    fragmentation_summary.to_csv(root / "h2_fragmentation_summary.csv", index=False)
    pd.DataFrame(forest_rows).to_csv(root / "h2_forest_estimates.csv", index=False)
    _make_figures(root, rates, forest_rows)
    _write_reports(
        root,
        table,
        alignment_audit,
        prompt_count,
        rates,
        class_counts,
        fragmentation_summary,
        models,
        entropy_models,
        forest_rows,
    )
    return {
        "run_dir": str(root),
        "token_rows": len(table),
        "sd_valid_rows": int(table["sd_valid"].sum()),
        "primary_rows": len(primary),
        "ambiguous_rows": int(table["morph_class"].isin(["CROSS_EOJEOL", "KIWI_COMPLEX"]).sum()),
        "models": {name: {key: value for key, value in result.items() if key != "summary"} for name, result in models.items()},
        "entropy_models": {name: {key: value for key, value in result.items() if key != "summary"} for name, result in entropy_models.items()},
    }
