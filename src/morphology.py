"""Kiwi morphology extraction and eojeol-level measurements."""

from __future__ import annotations

from typing import Any

from .alignment import boundary_f1, extract_text_eojeol_offsets, internal_token_boundaries, tokenize_with_offsets


def make_kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def analyze_eojeol(eojeol: str, kiwi: Any, tokenizer: Any) -> dict[str, Any]:
    morphemes = list(kiwi.tokenize(eojeol))
    morpheme_spans: list[tuple[int, int]] = []
    for morpheme in morphemes:
        start = int(morpheme.start)
        end = start + int(morpheme.len)
        if 0 <= start < end <= len(eojeol):
            morpheme_spans.append((start, end))
    morpheme_boundaries = sorted({end for _, end in morpheme_spans if end < len(eojeol)})
    token_ids, token_offsets = tokenize_with_offsets(tokenizer, eojeol)
    tokenizer_boundaries = internal_token_boundaries(token_offsets, len(eojeol))
    f1 = boundary_f1(morpheme_boundaries, tokenizer_boundaries)
    token_count = len(token_ids)
    return {
        "morpheme_count": len(morphemes),
        "llm_token_count": token_count,
        "fragmentation": float(token_count),
        "morpheme_boundaries": morpheme_boundaries,
        "tokenizer_boundaries": tokenizer_boundaries,
        "boundary_f1": f1,
        "misalignment": 1.0 - f1,
        "morpheme_spans": morpheme_spans,
        "tokenizer_offsets": token_offsets,
    }


def fragmentation_bin(token_count: int) -> str | None:
    if token_count == 2:
        return "2"
    if token_count == 3:
        return "3"
    if token_count == 4:
        return "4"
    if token_count >= 5:
        return "5+"
    return None


def extract_eojeol_rows(
    text: str,
    kiwi: Any,
    tokenizer: Any,
    prompt_id: int | str,
    output_token_to_eojeol: dict[int, int | None] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for eojeol_index, (surface, char_start, char_end) in enumerate(extract_text_eojeol_offsets(text)):
        analysis = analyze_eojeol(surface, kiwi, tokenizer)
        rows.append({
            "prompt_id": prompt_id,
            "eojeol_index": eojeol_index,
            "eojeol": surface,
            "char_start": char_start,
            "char_end": char_end,
            "morpheme_count": analysis["morpheme_count"],
            "llm_token_count": analysis["llm_token_count"],
            "fragmentation": analysis["fragmentation"],
            "fragmentation_bin": fragmentation_bin(analysis["llm_token_count"]),
            "morpheme_boundaries": analysis["morpheme_boundaries"],
            "tokenizer_boundaries": analysis["tokenizer_boundaries"],
            "boundary_f1": analysis["boundary_f1"],
            "misalignment": analysis["misalignment"],
            "morpheme_spans": analysis["morpheme_spans"],
        })
    return rows
