"""Character-offset and boundary-alignment helpers."""

from __future__ import annotations

from typing import Any, Iterable


def boundary_f1(morpheme_boundaries: Iterable[int], tokenizer_boundaries: Iterable[int]) -> float:
    gold = set(int(x) for x in morpheme_boundaries)
    predicted = set(int(x) for x in tokenizer_boundaries)
    if not gold and not predicted:
        return 1.0
    if not gold or not predicted:
        return 0.0
    true_positive = len(gold & predicted)
    precision = true_positive / len(predicted)
    recall = true_positive / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    if len(ids) != len(offsets):
        raise ValueError(f"Tokenizer returned {len(ids)} IDs but {len(offsets)} offsets")
    for start, end in offsets:
        if start < 0 or end < start or end > len(text):
            raise ValueError(f"Invalid character offset {(start, end)} for string of length {len(text)}")
    return ids, offsets


def internal_token_boundaries(offsets: Iterable[tuple[int, int]], text_length: int) -> list[int]:
    return sorted({end for start, end in offsets if 0 < end < text_length and end > start})


def token_ids_to_character_offsets(tokenizer: Any, token_ids: list[int]) -> dict[str, Any]:
    """Decode IDs and retain an offset per input ID.

    Fast-tokenizer offsets are preferred when decoded text re-encodes exactly.
    At a generation limit, a byte-level token sequence can end in an incomplete
    UTF-8 character; then the final decoded replacement character cannot
    re-encode to the original byte token. For that case, incremental prefix
    decoding attributes changed suffix characters to the token that completed
    or changed them. Special IDs skipped by decoding receive zero-width spans.
    """
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    text = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    visible_positions = [i for i, token_id in enumerate(token_ids) if token_id not in special_ids]
    visible_ids = [token_ids[i] for i in visible_positions]
    encoded_ids, visible_offsets = tokenize_with_offsets(tokenizer, text)
    offsets: list[tuple[int, int]] = [(0, 0)] * len(token_ids)
    roundtrip_exact = encoded_ids == visible_ids
    if roundtrip_exact:
        for original_position, offset in zip(visible_positions, visible_offsets):
            offsets[original_position] = offset
    else:
        previous_text = ""
        for original_position in visible_positions:
            prefix_text = tokenizer.decode(
                token_ids[:original_position + 1],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            shared = 0
            shared_limit = min(len(previous_text), len(prefix_text))
            while shared < shared_limit and previous_text[shared] == prefix_text[shared]:
                shared += 1
            # A partial byte sequence can decode as U+FFFD, then be replaced
            # when a later token completes the character. Clear stale spans.
            if shared < len(previous_text):
                for earlier_position in visible_positions:
                    if earlier_position >= original_position:
                        break
                    start, end = offsets[earlier_position]
                    if end > shared:
                        offsets[earlier_position] = (shared, shared)
            offsets[original_position] = (shared, max(shared, len(prefix_text)))
            previous_text = prefix_text
        if previous_text != text:
            raise ValueError("Incremental tokenizer decode did not reproduce the full decoded text")
    mismatch_index = None
    if not roundtrip_exact:
        mismatch_index = next((i for i, (a, b) in enumerate(zip(encoded_ids, visible_ids)) if a != b), min(len(encoded_ids), len(visible_ids)))
    return {
        "text": text,
        "offsets": offsets,
        "visible_token_positions": visible_positions,
        "roundtrip_exact": roundtrip_exact,
        "roundtrip_mismatch_index": mismatch_index,
    }


def map_token_positions_to_spans(
    token_offsets: list[tuple[int, int]],
    token_start: int,
    span_rows: list[dict[str, Any]],
    text_origin: int = 0,
) -> dict[int, int | None]:
    """Map absolute token positions to the eojeol span with greatest overlap."""
    result: dict[int, int | None] = {}
    for position in range(token_start, len(token_offsets)):
        start, end = token_offsets[position]
        if end <= start:
            result[position - token_start] = None
            continue
        local_start = max(0, start - text_origin)
        local_end = max(0, end - text_origin)
        best_id: int | None = None
        best_overlap = 0
        for row in span_rows:
            overlap = max(0, min(local_end, row["char_end"]) - max(local_start, row["char_start"]))
            if overlap > best_overlap:
                best_id = int(row["eojeol_index"])
                best_overlap = overlap
        result[position - token_start] = best_id
    return result


def extract_text_eojeol_offsets(text: str) -> list[tuple[str, int, int]]:
    import re

    return [(match.group(0), match.start(), match.end()) for match in re.finditer(r"\S+", text, flags=re.UNICODE)]
