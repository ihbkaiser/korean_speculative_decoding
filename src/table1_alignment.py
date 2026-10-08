"""Deterministic token/morpheme alignment helpers for the NAACL Table 1.

The functions in this module deliberately operate only on already materialized
text, token spans, and Kiwi-like morpheme records.  They do not import a
tokenizer, a morphology package, a model, or a network client.

Character offsets are Python string offsets and all spans use the half-open
``[start, end)`` convention.  ``relative_position_in_eojeol`` is normalized
over token boundaries: the first token is 0.0 and the last token is 1.0 (and a
single-token eojeol is 0.0).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from typing import Any


ALIGNMENT_CLASSES = ("EXACT", "WITHIN_SPLIT", "CROSS", "OTHER/AMBIGUOUS")
ALIGNED_CLASSES = frozenset({"EXACT", "WITHIN_SPLIT", "CROSS"})


def _empty_alignment() -> dict[str, Any]:
    """Return the complete alignment-field schema with safe null defaults."""

    return {
        "alignment_class": "OTHER/AMBIGUOUS",
        "crossed_boundary_count": 0,
        "crossed_boundary_positions": [],
        "fine_boundaries": [],
        "alignment_status": "EXCLUDED",
        "alignment_exclusion_reason": None,
        "left_pos": None,
        "right_pos": None,
        "fine_boundary": None,
        "coarse_boundary": None,
        "coarse_boundaries": [],
        "left_chars_in_token": None,
        "right_chars_in_token": None,
        "boundary_position_ratio": None,
        "boundary_ratio": None,
        "nearest_boundary_position": None,
        "nearest_boundary_positions": [],
        "nearest_left_pos": None,
        "nearest_right_pos": None,
        "nearest_fine_boundary": None,
        "nearest_coarse_boundary": None,
        "nearest_boundary_distance": None,
        "nearest_boundary_ambiguous": False,
        "nearest_left_pos_candidates": [],
        "nearest_right_pos_candidates": [],
        "nearest_fine_boundary_candidates": [],
        "nearest_coarse_boundary_candidates": [],
    }


_CANONICAL_ALIGNMENT_FIELDS = frozenset(_empty_alignment())
_CANONICAL_ROW_FIELDS = _CANONICAL_ALIGNMENT_FIELDS | frozenset(
    {
        "eojeol_text",
        "eojeol_char_len",
        "eojeol_token_count",
        "token_index_in_eojeol",
        "relative_position_in_eojeol",
        "is_first_token",
        "is_last_token",
        "morpheme_surfaces",
        "morpheme_pos_tags",
        "morpheme_char_spans",
        "token_char_span",
        "token_start",
        "token_end",
        "token_raw_text",
        "token_core_text",
        "visible_candidate",
        "token_eojeol_count",
        "prompt_id",
    }
)


def _as_span(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (str, bytes)):
        return None
    try:
        values = list(value)
    except (TypeError, ValueError):
        return None
    if len(values) != 2:
        return None
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in values):
        return None
    try:
        start, end = int(values[0]), int(values[1])
    except (TypeError, ValueError, OverflowError):
        return None
    return start, end


def _normalise_morphemes(
    morphemes: Sequence[Mapping[str, Any]],
    eojeol_text: str,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Validate ordered morpheme records and retain their original fields."""

    try:
        records = list(morphemes)
    except TypeError:
        return None, "ambiguous_morphology_span"

    normalized: list[dict[str, Any]] = []
    previous_start = -1
    previous_end = -1
    for record in records:
        if not isinstance(record, Mapping):
            return None, "ambiguous_morphology_span"
        if not {"surface", "pos", "start", "end"}.issubset(record):
            return None, "ambiguous_morphology_span"
        if (
            isinstance(record["start"], bool)
            or isinstance(record["end"], bool)
            or not isinstance(record["start"], Integral)
            or not isinstance(record["end"], Integral)
        ):
            return None, "ambiguous_morphology_span"
        start, end = int(record["start"]), int(record["end"])
        surface = record["surface"]
        pos = record["pos"]
        if (
            not isinstance(surface, str)
            or not isinstance(pos, str)
            or start < 0
            or start >= end
            or end > len(eojeol_text)
            or start < previous_start
            or start < previous_end
            or eojeol_text[start:end] != surface
        ):
            return None, "ambiguous_morphology_span"
        normalized.append({**dict(record), "start": start, "end": end})
        previous_start, previous_end = start, end
    return normalized, None


_COARSE_POS_GROUPS = {
    **{tag: "NOMINAL" for tag in ("NNG", "NNP", "NNB", "NP", "NR")},
    **{tag: "PARTICLE" for tag in ("JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC")},
    **{tag: "PREDICATE" for tag in ("VV", "VA", "VX", "VCP", "VCN", "VV-I", "VV-R", "VA-I", "VA-R")},
    **{tag: "ENDING" for tag in ("EP", "EF", "EC", "ETN", "ETM")},
    **{tag: "MODIFIER" for tag in ("MM", "MAG", "MAJ")},
    **{tag: "DERIVATIONAL" for tag in ("XPN", "XSN", "XSV", "XSA", "XSA-I", "XSM")},
    **{tag: "OTHER_CONTENT" for tag in ("XR", "SL", "SH", "SN", "IC", "W_URL", "W_EMAIL", "W_HASHTAG", "W_MENTION", "W_SERIAL", "W_EMOJI")},
    **{tag: "OTHER_NONCONTENT" for tag in ("SF", "SP", "SS", "SSO", "SSC", "SE", "SO", "SW", "SB", "UN", "Z_CODA", "Z_SIOT")},
}
_CONTENT_GROUPS = frozenset({"NOMINAL", "PREDICATE", "MODIFIER", "DERIVATIONAL", "OTHER_CONTENT"})


def _coarse_pos(pos: str | None) -> str | None:
    """Map a Kiwi POS tag to the repo's coarse morphology group."""

    if not pos:
        return None
    return _COARSE_POS_GROUPS.get(str(pos), "OTHER_NONCONTENT")


def _boundary_label(left_pos: str | None, right_pos: str | None) -> str | None:
    if left_pos is None or right_pos is None:
        return None
    return f"{left_pos}→{right_pos}"


def _coarse_boundary(left_pos: str | None, right_pos: str | None) -> str | None:
    left, right = _coarse_pos(left_pos), _coarse_pos(right_pos)
    if left is None or right is None:
        return None
    if left == "NOMINAL" and right == "PARTICLE":
        return "NOMINAL_TO_PARTICLE"
    if left == "PREDICATE" and right == "ENDING":
        return "PREDICATE_TO_ENDING"
    if left == "ENDING" and right == "ENDING":
        return "ENDING_TO_ENDING"
    if left in _CONTENT_GROUPS and right in _CONTENT_GROUPS:
        return "LEXICAL_TO_LEXICAL"
    return "OTHER"


def _internal_boundaries(
    morphemes: Sequence[Mapping[str, Any]],
    eojeol_char_len: int,
) -> list[dict[str, Any]]:
    boundaries: list[dict[str, Any]] = []
    for left, right in zip(morphemes, morphemes[1:]):
        position = int(left["end"])
        if position <= 0 or position >= eojeol_char_len or position != int(right["start"]):
            continue
        left_pos = str(left["pos"])
        right_pos = str(right["pos"])
        boundaries.append(
            {
                "position": position,
                "left_pos": left_pos,
                "right_pos": right_pos,
                "fine_boundary": _boundary_label(left_pos, right_pos),
                "coarse_boundary": _coarse_boundary(left_pos, right_pos),
            }
        )
    return boundaries


def _distance_to_span(position: int, start: int, end: int) -> int:
    if position < start:
        return start - position
    if position > end:
        return position - end
    return 0


def _add_nearest_boundary_fields(
    result: dict[str, Any],
    boundaries: Sequence[Mapping[str, Any]],
    token_start: int,
    token_end: int,
) -> None:
    if not boundaries:
        return
    distances = [_distance_to_span(int(boundary["position"]), token_start, token_end) for boundary in boundaries]
    nearest_distance = min(distances)
    nearest = [
        boundary
        for boundary, distance in zip(boundaries, distances)
        if distance == nearest_distance
    ]
    positions = [int(boundary["position"]) for boundary in nearest]
    result["nearest_boundary_positions"] = positions
    result["nearest_boundary_distance"] = nearest_distance
    result["nearest_boundary_ambiguous"] = len(nearest) > 1
    result["nearest_left_pos_candidates"] = [str(boundary["left_pos"]) for boundary in nearest]
    result["nearest_right_pos_candidates"] = [str(boundary["right_pos"]) for boundary in nearest]
    result["nearest_fine_boundary_candidates"] = [str(boundary["fine_boundary"]) for boundary in nearest]
    result["nearest_coarse_boundary_candidates"] = [str(boundary["coarse_boundary"]) for boundary in nearest]
    if len(nearest) != 1:
        return
    boundary = nearest[0]
    result["nearest_boundary_position"] = int(boundary["position"])
    result["nearest_left_pos"] = str(boundary["left_pos"])
    result["nearest_right_pos"] = str(boundary["right_pos"])
    result["nearest_fine_boundary"] = str(boundary["fine_boundary"])
    result["nearest_coarse_boundary"] = str(boundary["coarse_boundary"])


def classify_token_alignment(
    token_span: Sequence[int],
    morphemes: Sequence[Mapping[str, Any]],
    eojeol_text: str,
    token_raw_text: str | None = None,
    token_core_text: str | None = None,
) -> dict[str, Any]:
    """Classify one visible token span against ordered morpheme spans.

    ``token_core_text`` is checked against the eojeol substring when supplied;
    ``token_raw_text`` is retained for audit but is not compared because it may
    contain a tokenizer whitespace marker.  A malformed input is represented
    as ``OTHER/AMBIGUOUS`` with an explicit exclusion reason.
    """

    result = _empty_alignment()
    span = _as_span(token_span)
    if span is None:
        result["alignment_exclusion_reason"] = "invalid_token_span"
        return result
    token_start, token_end = span
    if token_start < 0 or token_start >= token_end or token_end > len(eojeol_text):
        result["alignment_exclusion_reason"] = "invalid_token_span"
        return result

    normalized, morphology_error = _normalise_morphemes(morphemes, eojeol_text)
    if morphology_error is not None or normalized is None:
        result["alignment_exclusion_reason"] = morphology_error or "ambiguous_morphology_span"
        return result

    if token_core_text is not None and token_core_text != eojeol_text[token_start:token_end]:
        result["alignment_exclusion_reason"] = "token_core_text_mismatch"
        return result

    overlaps = [
        morpheme
        for morpheme in normalized
        if min(token_end, int(morpheme["end"])) > max(token_start, int(morpheme["start"]))
    ]
    if not overlaps:
        result["alignment_exclusion_reason"] = "morphology_no_overlap"
        return result

    boundaries = _internal_boundaries(normalized, len(eojeol_text))
    if len(overlaps) == 1:
        morpheme = overlaps[0]
        morpheme_start, morpheme_end = int(morpheme["start"]), int(morpheme["end"])
        if token_start < morpheme_start or token_end > morpheme_end:
            result["alignment_exclusion_reason"] = "partial_morpheme_overlap"
            return result
        if token_start == morpheme_start and token_end == morpheme_end:
            result["alignment_class"] = "EXACT"
        else:
            result["alignment_class"] = "WITHIN_SPLIT"
            _add_nearest_boundary_fields(result, boundaries, token_start, token_end)
        result["alignment_status"] = "ALIGNED"
        return result

    crossed = [
        boundary
        for boundary in boundaries
        if token_start < int(boundary["position"]) < token_end
    ]
    if not crossed:
        result["alignment_exclusion_reason"] = "ambiguous_morphology_span"
        return result

    result["alignment_class"] = "CROSS"
    result["alignment_status"] = "ALIGNED"
    result["crossed_boundary_count"] = len(crossed)
    result["crossed_boundary_positions"] = [int(boundary["position"]) for boundary in crossed]
    result["fine_boundaries"] = [str(boundary["fine_boundary"]) for boundary in crossed]
    result["coarse_boundaries"] = [str(boundary["coarse_boundary"]) for boundary in crossed]
    if len(crossed) == 1:
        boundary = crossed[0]
        boundary_position = int(boundary["position"])
        left_chars = boundary_position - token_start
        right_chars = token_end - boundary_position
        result["left_pos"] = str(boundary["left_pos"])
        result["right_pos"] = str(boundary["right_pos"])
        result["fine_boundary"] = str(boundary["fine_boundary"])
        result["coarse_boundary"] = str(boundary["coarse_boundary"])
        result["left_chars_in_token"] = left_chars
        result["right_chars_in_token"] = right_chars
        result["boundary_position_ratio"] = left_chars / (left_chars + right_chars)
        result["boundary_ratio"] = result["boundary_position_ratio"]
    return result


def _serialise_morpheme_field(morphemes: Iterable[Mapping[str, Any]], field: str) -> list[Any]:
    values: list[Any] = []
    for morpheme in morphemes:
        if field == "span":
            try:
                span = _as_span((morpheme["start"], morpheme["end"]))
            except (KeyError, TypeError, ValueError, OverflowError):
                span = None
            if span is None:
                values.append(None)
            else:
                values.append([span[0], span[1]])
        else:
            values.append(morpheme.get(field))
    return values


def build_aligned_token_row(
    eojeol_text: str,
    token_span: Sequence[int],
    morphemes: Sequence[Mapping[str, Any]],
    token_raw_text: str | None = None,
    token_core_text: str | None = None,
    token_index_in_eojeol: int = 0,
    eojeol_token_count: int = 1,
    *,
    prompt_id: str | int | None = None,
    visible_candidate: bool = True,
    token_eojeol_count: int = 1,
    exclusion_reason: str | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """Build one flat, serialization-friendly aligned-token row."""

    if not isinstance(eojeol_text, str):
        raise TypeError("eojeol_text must be str")
    if type(eojeol_token_count) is not int:
        raise TypeError("token_eojeol_count/eojeol_token_count must be int")
    if type(token_index_in_eojeol) is not int:
        raise TypeError("token_index_in_eojeol must be int")
    if type(visible_candidate) is not bool:
        raise TypeError("visible_candidate must be bool")
    if type(token_eojeol_count) is not int:
        raise TypeError("token_eojeol_count must be int")
    conflicts = sorted(set(metadata) & _CANONICAL_ROW_FIELDS)
    if conflicts:
        raise ValueError(f"metadata cannot overwrite canonical fields: {', '.join(conflicts)}")

    token_count = eojeol_token_count
    token_index = token_index_in_eojeol
    if token_count <= 0 or token_index < 0 or token_index >= token_count:
        raise ValueError("token_index_in_eojeol must be inside eojeol_token_count")

    classification = classify_token_alignment(
        token_span,
        morphemes,
        eojeol_text,
        token_raw_text=token_raw_text,
        token_core_text=token_core_text,
    )
    forced_reason = exclusion_reason
    if forced_reason is None and token_eojeol_count != 1:
        forced_reason = "multi_eojeol_token"
    if forced_reason is None and not visible_candidate:
        forced_reason = "special_or_non_visible_token"
    if forced_reason is not None:
        classification["alignment_class"] = "OTHER/AMBIGUOUS"
        classification["alignment_status"] = "EXCLUDED"
        classification["alignment_exclusion_reason"] = forced_reason

    span = _as_span(token_span)
    token_char_span = list(span) if span is not None else None
    relative_position = token_index / (token_count - 1) if token_count > 1 else 0.0
    row: dict[str, Any] = {
        "eojeol_text": eojeol_text,
        "eojeol_char_len": len(eojeol_text),
        "eojeol_token_count": token_count,
        "token_index_in_eojeol": token_index,
        "relative_position_in_eojeol": relative_position,
        "is_first_token": token_index == 0,
        "is_last_token": token_index == token_count - 1,
        "morpheme_surfaces": _serialise_morpheme_field(morphemes, "surface"),
        "morpheme_pos_tags": _serialise_morpheme_field(morphemes, "pos"),
        "morpheme_char_spans": _serialise_morpheme_field(morphemes, "span"),
        "token_char_span": token_char_span,
        "token_start": token_char_span[0] if token_char_span is not None else None,
        "token_end": token_char_span[1] if token_char_span is not None else None,
        "token_raw_text": token_raw_text,
        "token_core_text": token_core_text,
        "visible_candidate": visible_candidate,
        "token_eojeol_count": token_eojeol_count,
        "prompt_id": prompt_id,
    }
    row.update(metadata)
    row.update(classification)
    return row


_COMPLETION_STATUSES = frozenset({"COMPLETED", "INCOMPLETE", "FAILED", "PENDING"})


def _validate_prompt_id(value: Any) -> str | int:
    if value is None:
        raise ValueError("prompt_id must be an int or str")
    if isinstance(value, bool):
        raise TypeError("prompt_id must be a non-boolean int or non-empty str")
    if isinstance(value, Integral):
        normalized = int(value)
        if normalized < 0:
            raise ValueError("prompt_id must be non-negative")
        return normalized
    if isinstance(value, str):
        if not value:
            raise ValueError("prompt_id must be non-empty")
        return value
    raise TypeError("prompt_id must be an int or str")


def _record_list(value: Any, field: str) -> list[Mapping[str, Any]]:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            records = value.to_dict(orient="records")
        except TypeError as exc:
            raise TypeError(f"{field} must provide record mappings") from exc
    elif isinstance(value, Mapping):
        records = [value]
    else:
        try:
            records = list(value)
        except TypeError as exc:
            raise TypeError(f"{field} must be iterable") from exc
    if not isinstance(records, list):
        raise TypeError(f"{field} must provide record mappings")
    return records


def _strict_token_count(value: Any, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be int")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _normalise_completion_records(
    completed_prompt_metadata: Iterable[Mapping[str, Any]] | Mapping[Any, Any],
) -> list[dict[str, Any]]:
    if isinstance(completed_prompt_metadata, Mapping):
        record_keys = {"prompt_id", "completed", "inference_completed", "status", "generated_token_count", "generated_tokens"}
        if record_keys.intersection(completed_prompt_metadata):
            records: list[Any] = [dict(completed_prompt_metadata)]
        else:
            records = []
            for raw_prompt_id, value in completed_prompt_metadata.items():
                prompt_id = _validate_prompt_id(raw_prompt_id)
                if isinstance(value, Mapping):
                    record = dict(value)
                    if "prompt_id" in record and _validate_prompt_id(record["prompt_id"]) != prompt_id:
                        raise ValueError("prompt_id conflicts with completion metadata key")
                    record["prompt_id"] = prompt_id
                elif type(value) is bool:
                    record = {"prompt_id": prompt_id, "completed": value}
                else:
                    raise TypeError("completion metadata mapping values must be bool or mappings")
                records.append(record)
    else:
        records = _record_list(completed_prompt_metadata, "completed_prompt_metadata")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str | int] = set()
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise TypeError("each completion metadata record must be a mapping")
        record = dict(raw_record)
        if "prompt_id" not in record:
            raise ValueError("completion metadata requires prompt_id")
        prompt_id = _validate_prompt_id(record["prompt_id"])
        if prompt_id in seen_ids:
            raise ValueError(f"duplicate prompt_id in completion metadata: {prompt_id}")
        seen_ids.add(prompt_id)

        has_completed = "completed" in record
        has_inference_completed = "inference_completed" in record
        has_status = "status" in record
        completion_values: list[bool] = []
        if has_completed:
            if type(record["completed"]) is not bool:
                raise TypeError("completed must be bool")
            completion_values.append(record["completed"])
        if has_inference_completed:
            if type(record["inference_completed"]) is not bool:
                raise TypeError("inference_completed must be bool")
            completion_values.append(record["inference_completed"])
        status_completed: bool | None = None
        if has_status:
            status = record["status"]
            if not isinstance(status, str) or status not in _COMPLETION_STATUSES:
                raise ValueError(f"invalid completion status: {status!r}")
            status_completed = status == "COMPLETED"
            completion_values.append(status_completed)
        if not completion_values:
            raise ValueError("completion metadata requires completed, inference_completed, or status")
        if any(value != completion_values[0] for value in completion_values[1:]):
            raise ValueError(f"conflicting completion status for prompt_id {prompt_id}")

        count_values: list[int] = []
        for field in ("generated_token_count", "generated_tokens"):
            if field in record:
                count_values.append(_strict_token_count(record[field], "generated token count"))
        if len(count_values) == 2 and count_values[0] != count_values[1]:
            raise ValueError(f"conflicting generated token counts for prompt_id {prompt_id}")
        generated_count = count_values[0] if count_values else 0
        normalized.append(
            {
                "prompt_id": prompt_id,
                "completed": completion_values[0],
                "generated_token_count": generated_count,
            }
        )
    return normalized


def _completed_prompt_ids(completed_prompt_metadata: Iterable[Mapping[str, Any]] | Mapping[Any, Any] | None) -> tuple[set[Any] | None, int | None, int | None]:
    if completed_prompt_metadata is None:
        return None, None, None
    records = _normalise_completion_records(completed_prompt_metadata)
    completed_ids = {record["prompt_id"] for record in records if record["completed"]}
    generated_tokens = sum(record["generated_token_count"] for record in records if record["completed"])
    return completed_ids, len(completed_ids), generated_tokens


def _row_is_visible(row: Mapping[str, Any]) -> bool:
    if "visible_candidate" in row:
        value = row["visible_candidate"]
    elif "is_visible" in row:
        value = row["is_visible"]
    else:
        return True
    if type(value) is not bool:
        raise TypeError("visible_candidate must be bool")
    return value


def _validate_row_count(value: Any, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be int")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _row_is_eligible(row: Mapping[str, Any]) -> bool:
    if "token_eojeol_count" in row:
        _validate_row_count(row["token_eojeol_count"], "token_eojeol_count")
    return (
        _row_is_visible(row)
        and row.get("alignment_status", "ALIGNED") == "ALIGNED"
        and row.get("alignment_class") in ALIGNED_CLASSES
    )


def _percent(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def aggregate_alignment_rows(
    aligned_rows: Iterable[Mapping[str, Any]],
    completed_prompt_metadata: Iterable[Mapping[str, Any]] | Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate Table 1 counts and percentages from aligned-token rows.

    When completed-prompt metadata is supplied, rows with a ``prompt_id`` are
    restricted to completed prompt IDs. Rows without a prompt ID remain
    eligible for local/unit-test aggregation.
    """

    rows = _record_list(aligned_rows, "aligned_rows")
    completed_ids, completed_prompt_count, generated_token_count = _completed_prompt_ids(completed_prompt_metadata)
    if completed_ids is not None:
        rows = [row for row in rows if "prompt_id" not in row or row.get("prompt_id") in completed_ids]

    visible_rows = [row for row in rows if _row_is_visible(row)]
    eligible_rows = [row for row in visible_rows if _row_is_eligible(row)]
    exact_count = sum(row.get("alignment_class") == "EXACT" for row in eligible_rows)
    cross_count = sum(row.get("alignment_class") == "CROSS" for row in eligible_rows)
    within_count = sum(row.get("alignment_class") == "WITHIN_SPLIT" for row in eligible_rows)
    single_boundary_cross_count = sum(
        row.get("alignment_class") == "CROSS" and int(row.get("crossed_boundary_count", 0)) == 1
        for row in eligible_rows
    )
    exclusion_breakdown = Counter(
        str(row.get("alignment_exclusion_reason") or "unspecified")
        for row in visible_rows
        if not _row_is_eligible(row)
    )
    visible_count = len(visible_rows)
    eligible_count = len(eligible_rows)
    excluded_count = visible_count - eligible_count
    result: dict[str, Any] = {
        "completed_prompt_count": completed_prompt_count,
        "generated_token_count": generated_token_count,
        "visible_candidate_count": visible_count,
        "eligible_count": eligible_count,
        "exact_count": exact_count,
        "cross_count": cross_count,
        "within_count": within_count,
        "single_boundary_cross_count": single_boundary_cross_count,
        "exclusion_count": excluded_count,
        "exclusion_breakdown": dict(sorted(exclusion_breakdown.items())),
        "aligned_percent": _percent(eligible_count, visible_count),
        "exclusion_percent": _percent(excluded_count, visible_count),
        "exact_percent": _percent(exact_count, eligible_count),
        "cross_percent": _percent(cross_count, eligible_count),
        "within_percent": _percent(within_count, eligible_count),
    }
    result["aligned_percentage"] = result["aligned_percent"]
    result["exclusion_percentage"] = result["exclusion_percent"]
    result["exclusion_breakdown_total"] = sum(exclusion_breakdown.values())
    return result


def support_counts_by_fine_boundary(
    aligned_rows: Iterable[Mapping[str, Any]],
    min_cross: int = 200,
    min_within: int = 200,
    *,
    cross_threshold: int | None = None,
    within_threshold: int | None = None,
) -> dict[str, Any]:
    """Count supported fine POS transitions from eligible local evidence.

    CROSS evidence is limited to single-boundary rows. WITHIN evidence is
    limited to rows with a unique nearest boundary; tied nearest boundaries are
    intentionally excluded from boundary-specific support.
    """

    if cross_threshold is not None:
        min_cross = cross_threshold
    if within_threshold is not None:
        min_within = within_threshold
    if min_cross < 0 or min_within < 0:
        raise ValueError("support thresholds must be non-negative")

    cross_counts: Counter[str] = Counter()
    within_counts: Counter[str] = Counter()
    for row in _record_list(aligned_rows, "aligned_rows"):
        if not _row_is_eligible(row):
            continue
        if row.get("alignment_class") == "CROSS" and int(row.get("crossed_boundary_count", 0)) == 1:
            fine_boundary = row.get("fine_boundary")
            if fine_boundary:
                cross_counts[str(fine_boundary)] += 1
        elif row.get("alignment_class") == "WITHIN_SPLIT" and not bool(row.get("nearest_boundary_ambiguous", False)):
            fine_boundary = row.get("nearest_fine_boundary")
            if fine_boundary:
                within_counts[str(fine_boundary)] += 1

    boundaries = sorted(set(cross_counts) | set(within_counts))
    by_fine_boundary = {
        boundary: {
            "cross_count": cross_counts[boundary],
            "within_count": within_counts[boundary],
            "supported": cross_counts[boundary] >= min_cross and within_counts[boundary] >= min_within,
        }
        for boundary in boundaries
    }
    supported_boundaries = [boundary for boundary in boundaries if by_fine_boundary[boundary]["supported"]]
    return {
        "by_fine_boundary": by_fine_boundary,
        "supported_boundaries": supported_boundaries,
        "supported_boundary_count": len(supported_boundaries),
        "cross_threshold": min_cross,
        "within_threshold": min_within,
    }


def _row_span(row: Mapping[str, Any]) -> tuple[int, int] | None:
    if row.get("token_char_span") is not None:
        return _as_span(row.get("token_char_span"))
    if row.get("token_start") is not None and row.get("token_end") is not None:
        return _as_span((row.get("token_start"), row.get("token_end")))
    return None


def assert_alignment_invariants(
    aligned_rows: Iterable[Mapping[str, Any]],
    summary: Mapping[str, Any] | None = None,
    completed_prompt_metadata: Iterable[Mapping[str, Any]] | Mapping[Any, Any] | None = None,
) -> bool:
    """Raise ``AssertionError`` when Table 1 alignment invariants are broken."""

    rows = _record_list(aligned_rows, "aligned_rows")
    recomputed = aggregate_alignment_rows(rows, completed_prompt_metadata=completed_prompt_metadata)
    if summary is not None:
        assert isinstance(summary, Mapping), "summary must be a mapping"
        for field, supplied_value in summary.items():
            if field not in recomputed:
                continue
            if field in {"completed_prompt_count", "generated_token_count"} and completed_prompt_metadata is None:
                continue
            assert supplied_value == recomputed[field], f"aggregate mismatch for {field}"
    single = recomputed["single_boundary_cross_count"]
    cross = recomputed["cross_count"]
    eligible = recomputed["eligible_count"]
    within = recomputed["within_count"]
    assert single <= cross <= eligible, "single_boundary_cross <= cross <= eligible"
    assert within <= eligible, "within <= eligible"

    visible = recomputed["visible_candidate_count"]
    excluded = recomputed["exclusion_count"]
    breakdown = recomputed["exclusion_breakdown"]
    assert excluded == visible - eligible, "exclusions reconcile with visible and eligible"
    assert sum(int(value) for value in breakdown.values()) == excluded, "exclusions reconcile by reason"

    if completed_prompt_metadata is not None:
        completion_records = _normalise_completion_records(completed_prompt_metadata)
        row_counts = Counter(row.get("prompt_id") for row in rows if "prompt_id" in row)
        for completion in completion_records:
            if not completion["completed"]:
                continue
            prompt_id = completion["prompt_id"]
            expected_count = int(completion["generated_token_count"])
            actual_count = int(row_counts.get(prompt_id, 0))
            assert actual_count == expected_count, (
                f"generated token count mismatch for prompt_id={prompt_id!r}: "
                f"rows={actual_count}, reference={expected_count}"
            )

    for row in rows:
        eojeol_text = row.get("eojeol_text")
        eojeol_length = row.get("eojeol_char_len", len(eojeol_text) if isinstance(eojeol_text, str) else None)
        assert isinstance(eojeol_text, str), "eojeol_text is a string"
        assert isinstance(eojeol_length, int) and not isinstance(eojeol_length, bool) and eojeol_length >= 0, "eojeol length is valid"
        assert eojeol_length == len(eojeol_text), "eojeol_char_len equals len(eojeol_text)"

        visible_candidate = _row_is_visible(row)
        token_span = None
        if "token_char_span" in row:
            token_span = _as_span(row["token_char_span"])
            assert token_span is not None, "token span representation is valid"
            if visible_candidate:
                assert token_span[0] < token_span[1], "token span has positive width"
            else:
                assert token_span[0] <= token_span[1], "non-visible token span is ordered"
        separate_span = None
        if "token_start" in row or "token_end" in row:
            assert "token_start" in row and "token_end" in row, "token span representation is valid"
            separate_span = _as_span((row["token_start"], row["token_end"]))
            assert separate_span is not None, "token span representation is valid"
            if visible_candidate:
                assert separate_span[0] < separate_span[1], "token span has positive width"
            else:
                assert separate_span[0] <= separate_span[1], "non-visible token span is ordered"
        if token_span is not None and separate_span is not None:
            assert token_span == separate_span, "token_char_span reconciles with token_start/token_end"
        if token_span is None:
            token_span = separate_span
        if token_span is None and not visible_candidate:
            continue
        assert token_span is not None, "token span representation is valid"
        start, end = token_span
        if visible_candidate:
            assert start < end, "token span has positive width"
            assert 0 <= start < end <= eojeol_length, "token span inside eojeol"
        else:
            assert 0 <= start <= end <= eojeol_length, "non-visible token span inside eojeol"

        assert "morpheme_char_spans" in row, "morpheme span representation is present"
        morpheme_spans = row["morpheme_char_spans"]
        assert not isinstance(morpheme_spans, (str, bytes)), "morpheme span representation is valid"
        try:
            morpheme_spans = list(morpheme_spans)
        except TypeError as exc:
            raise AssertionError("morpheme span representation is valid") from exc
        for morpheme_span in morpheme_spans:
            assert morpheme_span is not None, "morpheme span representation is valid"
            span = _as_span(morpheme_span)
            assert span is not None and 0 <= span[0] < span[1] <= eojeol_length, "morpheme span inside eojeol"
    return True


# Small aliases make the helpers convenient to discover without duplicating
# behavior or creating a second API surface.
classify_alignment = classify_token_alignment
build_token_alignment_row = build_aligned_token_row
aggregate_alignment = aggregate_alignment_rows
count_supported_boundaries = support_counts_by_fine_boundary


__all__ = [
    "ALIGNMENT_CLASSES",
    "ALIGNED_CLASSES",
    "aggregate_alignment",
    "aggregate_alignment_rows",
    "assert_alignment_invariants",
    "build_aligned_token_row",
    "build_token_alignment_row",
    "classify_alignment",
    "classify_token_alignment",
    "count_supported_boundaries",
    "support_counts_by_fine_boundary",
]
