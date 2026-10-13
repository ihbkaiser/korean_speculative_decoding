"""Shared-text position sampling and statistics for the cross-target DP study."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata

from .table1_data import text_sha256


WORD_RE = re.compile(r"\S+", flags=re.UNICODE)
PRIMARY_SET = "eojeol_boundary"
MORPH_SET = "morph_extended"


def position_id(doc_id: str, char_offset: int) -> str:
    value = f"{doc_id}\0{int(char_offset)}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def prefix_hash(prefix: str) -> str:
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()


def _context_start(starts: Sequence[int], offset: int, context_window_chars: int) -> int:
    target = int(offset) - int(context_window_chars)
    index = bisect_right(starts, target) - 1
    return int(starts[index]) if index >= 0 else 0


def _position_row(
    *,
    document: Mapping[str, Any],
    text: str,
    word_starts: Sequence[int],
    char_offset: int,
    analysis_set: str,
    position_type: str,
    context_window_chars: int,
    **extra: Any,
) -> dict[str, Any]:
    starts = list(word_starts)
    context_start = _context_start(starts, char_offset, context_window_chars)
    prefix = text[context_start:char_offset]
    if not prefix:
        raise ValueError(f"Empty prefix for {document.get('doc_id')} at {char_offset}")
    return {
        "position_id": position_id(str(document["doc_id"]), char_offset),
        "doc_id": str(document["doc_id"]),
        "source_index": int(document.get("source_index", -1)),
        "pool_index": int(document.get("pool_index", -1)),
        "document_raw_text_hash": str(document.get("raw_text_hash", text_sha256(text))),
        "char_offset": int(char_offset),
        "context_start": context_start,
        "context_chars": len(prefix),
        "shared_prefix_sha256": prefix_hash(prefix),
        "analysis_set": analysis_set,
        "position_type": position_type,
        "prefix_text": prefix,
        **extra,
    }


def sample_eojeol_positions(
    documents: Sequence[Mapping[str, Any]],
    *,
    max_documents: int = 2000,
    positions_per_document: int = 20,
    context_window_chars: int = 256,
    min_context_chars: int = 128,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Mapping[str, Any]]]:
    """Select evenly spaced shared eojeol-onset positions from frozen documents.

    Text is expected to have already been normalized by the frozen FineWeb2
    pipeline. Each selected document contributes exactly the requested number
    of positions, which makes within-document label permutations well-defined.
    """
    if max_documents < 1 or positions_per_document < 1:
        raise ValueError("max_documents and positions_per_document must be positive")
    if context_window_chars < min_context_chars or min_context_chars < 1:
        raise ValueError("context window must be at least the positive minimum context")

    positions: list[dict[str, Any]] = []
    selected_documents: dict[str, Mapping[str, Any]] = {}
    excluded_short = 0
    candidate_total = 0
    considered = list(documents[:max_documents])
    for document in considered:
        doc_id = str(document["doc_id"])
        text = str(document["text"])
        if unicodedata.normalize("NFC", text).strip() != text:
            raise ValueError(f"Frozen text is not NFC/stripped for doc_id={doc_id}")
        expected_hash = document.get("raw_text_hash")
        if expected_hash and text_sha256(text) != str(expected_hash):
            raise ValueError(f"Frozen text hash mismatch for doc_id={doc_id}")

        matches = list(WORD_RE.finditer(text))
        starts = [match.start() for match in matches]
        candidates = [index for index, match in enumerate(matches) if match.start() >= min_context_chars]
        candidate_total += len(candidates)
        if len(candidates) < positions_per_document:
            excluded_short += 1
            continue

        # Even spacing preserves coverage of each document without overweighting
        # unusually long web pages. Endpoints are included deterministically.
        chosen = np.linspace(0, len(candidates) - 1, positions_per_document, dtype=np.int64)
        selected_documents[doc_id] = document
        for within_doc_index, candidate_index in enumerate(chosen.tolist()):
            word_index = candidates[int(candidate_index)]
            match = matches[word_index]
            offset = int(match.start())
            row = _position_row(
                document=document,
                text=text,
                word_starts=starts,
                char_offset=offset,
                analysis_set=PRIMARY_SET,
                position_type="eojeol_onset",
                context_window_chars=context_window_chars,
                eojeol_index=int(word_index),
                eojeol_start=int(match.start()),
                eojeol_end=int(match.end()),
                eojeol=str(match.group(0)),
                position_index_in_document=within_doc_index,
                n_eligible_eojeol_onsets=len(candidates),
            )
            if row["context_chars"] < min_context_chars:
                raise AssertionError("selected position has insufficient shared text context")
            positions.append(row)

    metadata = {
        "documents_considered": len(considered),
        "documents_included": len(selected_documents),
        "documents_excluded_insufficient_positions": excluded_short,
        "eligible_eojeol_onsets_before_sampling": candidate_total,
        "positions_per_document": positions_per_document,
        "primary_position_count": len(positions),
        "context_window_chars": context_window_chars,
        "min_context_chars": min_context_chars,
        "position_rule": "Evenly spaced eojeol starts at offsets >= min_context_chars; one identical Unicode prefix is reconstructed for all targets.",
        "normalization": "Frozen FineWeb2 text is NFC-normalized and stripped; this script verifies and does not modify it.",
    }
    return positions, metadata, selected_documents


def add_matched_morphology_positions(
    primary_positions: Sequence[Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
    kiwi: Any,
    *,
    anchor_words_per_document: int = 2,
    context_window_chars: int = 256,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add one Kiwi boundary and a nearest within-morpheme offset per anchor word."""
    by_doc: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in primary_positions:
        by_doc[str(row["doc_id"])].append(row)

    output: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = defaultdict(int)
    anchor_count = 0
    pair_count = 0
    for doc_id, rows in by_doc.items():
        rows = sorted(rows, key=lambda row: int(row["char_offset"]))
        if anchor_words_per_document > len(rows):
            reason_counts["not_enough_primary_positions"] += 1
            continue
        indices = np.linspace(0, len(rows) - 1, anchor_words_per_document, dtype=np.int64)
        document = documents[doc_id]
        text = str(document["text"])
        all_starts = [match.start() for match in WORD_RE.finditer(text)]
        for selected_index in indices.tolist():
            primary = rows[int(selected_index)]
            anchor_count += 1
            eojeol = str(primary["eojeol"])
            eojeol_start = int(primary["eojeol_start"])
            try:
                tokens = list(kiwi.tokenize(eojeol))
            except Exception:
                reason_counts["kiwi_tokenizer_error"] += 1
                continue
            spans: list[tuple[int, int, str, str]] = []
            valid = bool(tokens)
            cursor = 0
            for token in tokens:
                start = int(token.start)
                end = start + int(token.len)
                form = str(token.form)
                tag = str(token.tag)
                if start != cursor or end <= start or end > len(eojeol) or eojeol[start:end] != form:
                    valid = False
                    break
                spans.append((start, end, form, tag))
                cursor = end
            if not valid or cursor != len(eojeol):
                reason_counts["kiwi_span_not_exact_cover"] += 1
                continue
            boundaries = [
                spans[i][1]
                for i in range(len(spans) - 1)
                if spans[i][1] == spans[i + 1][0] and 0 < spans[i][1] < len(eojeol)
            ]
            if not boundaries:
                reason_counts["no_internal_morpheme_boundary"] += 1
                continue

            boundary = int(boundaries[len(boundaries) // 2])
            left = next((span for span in spans if span[1] == boundary), None)
            right = next((span for span in spans if span[0] == boundary), None)
            if left is None or right is None:
                reason_counts["boundary_span_missing"] += 1
                continue

            internal_candidates: list[tuple[int, int, tuple[int, int, str, str]]] = []
            for span in spans:
                start, end, _, _ = span
                for offset in range(start + 1, end):
                    if offset not in boundaries:
                        internal_candidates.append((abs(offset - boundary), offset, span))
            if not internal_candidates:
                reason_counts["no_within_morpheme_character_offset"] += 1
                continue
            _, internal_offset, internal_span = min(internal_candidates, key=lambda item: (item[0], item[1]))

            pair_id = position_id(doc_id, eojeol_start) + ":morph"
            common = {
                "matched_word_id": pair_id,
                "eojeol_index": int(primary["eojeol_index"]),
                "eojeol_start": eojeol_start,
                "eojeol_end": int(primary["eojeol_end"]),
                "eojeol": eojeol,
                "morph_boundary_char_offset": eojeol_start + boundary,
                "morph_left_morpheme": left[2],
                "morph_left_pos": left[3],
                "morph_right_morpheme": right[2],
                "morph_right_pos": right[3],
            }
            for offset, position_type, extra in (
                (
                    eojeol_start + boundary,
                    "morpheme_boundary",
                    {"morpheme_form": None, "morpheme_pos": None},
                ),
                (
                    eojeol_start + internal_offset,
                    "morpheme_internal",
                    {"morpheme_form": internal_span[2], "morpheme_pos": internal_span[3]},
                ),
            ):
                position = _position_row(
                    document=document,
                    text=text,
                    word_starts=all_starts,
                    char_offset=offset,
                    analysis_set=MORPH_SET,
                    position_type=position_type,
                    context_window_chars=context_window_chars,
                    **common,
                    **extra,
                )
                output.append(position)
            pair_count += 1

    metadata = {
        "anchor_words_requested": len(by_doc) * anchor_words_per_document,
        "anchor_words_checked": anchor_count,
        "matched_word_pairs": pair_count,
        "morph_position_count": len(output),
        "morph_position_rule": "For two evenly spaced sampled eojeols per document, select the middle internal Kiwi boundary and the nearest character gap strictly inside one Kiwi morpheme; retain only exact-cover analyses with a matched pair.",
        "exclusions": dict(sorted(reason_counts.items())),
    }
    return output, metadata


def exact_bottom_labels(values: Sequence[float], fraction: float) -> np.ndarray:
    """Mark exactly the lowest fraction, with deterministic stable tie handling."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between zero and one")
    count = max(1, int(math.ceil(fraction * len(values))))
    order = np.argsort(values, kind="stable")
    labels = np.zeros(len(values), dtype=bool)
    labels[order[:count]] = True
    return labels


def pairwise_overlap(labels_a: Sequence[bool], labels_b: Sequence[bool]) -> dict[str, float | int]:
    a = np.asarray(labels_a, dtype=bool)
    b = np.asarray(labels_b, dtype=bool)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("decision labels must be non-empty vectors with equal shape")
    intersection = int(np.sum(a & b))
    union = int(np.sum(a | b))
    prevalence_a = float(np.mean(a))
    prevalence_b = float(np.mean(b))
    overlap_fraction = intersection / len(a)
    overlap_pct = 100.0 * overlap_fraction
    expected = prevalence_a * prevalence_b
    return {
        "n_positions": int(len(a)),
        "n_a": int(np.sum(a)),
        "n_b": int(np.sum(b)),
        "intersection_size": intersection,
        "union_size": union,
        "overlap_fraction": float(overlap_fraction),
        "overlap_pct": float(overlap_pct),
        "jaccard": float(intersection / union) if union else 0.0,
        "conditional_b_given_a": float(intersection / np.sum(a)) if np.sum(a) else float("nan"),
        "independence_expected_overlap_fraction": float(expected),
        "independence_expected_overlap_pct": float(100.0 * expected),
        "overlap_enrichment": float(overlap_fraction / expected) if expected else float("nan"),
    }


def rank_correlation(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError("correlation inputs must be equal one-dimensional vectors")
    ar = rankdata(a, method="average")
    br = rankdata(b, method="average")
    if np.std(ar) == 0 or np.std(br) == 0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def document_bootstrap_indices(doc_ids: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    """Return row indices from a document bootstrap, preserving sampled clusters."""
    doc_ids = np.asarray(doc_ids, dtype=object)
    unique_docs = np.unique(doc_ids)
    if len(unique_docs) < 2:
        raise ValueError("document bootstrap needs at least two documents")
    sampled = rng.choice(unique_docs, size=len(unique_docs), replace=True)
    by_doc: dict[str, np.ndarray] = {
        str(doc): np.flatnonzero(doc_ids == doc) for doc in unique_docs
    }
    return np.concatenate([by_doc[str(doc)] for doc in sampled])


def within_document_permutation(
    labels: np.ndarray,
    doc_ids: Sequence[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Independently shuffle each model's binary labels within every document."""
    values = np.asarray(labels, dtype=bool)
    docs = np.asarray(doc_ids, dtype=object)
    if values.ndim != 2 or values.shape[0] != len(docs):
        raise ValueError("labels must be positions x models and align with doc_ids")
    inverse = np.unique(docs, return_inverse=True)[1]
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    grouped = values[order]
    if len(counts) and np.all(counts == counts[0]):
        blocks = grouped.reshape(len(counts), int(counts[0]), values.shape[1])
        random_order = np.argsort(rng.random(blocks.shape), axis=1, kind="stable")
        shuffled = np.take_along_axis(blocks, random_order, axis=1).reshape(values.shape)
    else:
        shuffled = grouped.copy()
        boundaries = np.concatenate(([0], np.cumsum(counts)))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            for model_index in range(values.shape[1]):
                shuffled[start:end, model_index] = rng.permutation(
                    shuffled[start:end, model_index]
                )
    output = np.empty_like(values)
    output[order] = shuffled
    return output

