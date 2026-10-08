from __future__ import annotations

from src.table1_data import (
    build_prompt_pool,
    document_hangul_ratio,
    normalize_document_text,
)


def test_normalization_and_hangul_filter_are_deterministic():
    text = "  한국어\n문서  "
    assert normalize_document_text(text) == "한국어\n문서"
    assert document_hangul_ratio("한국어 문서") == 1.0
    assert document_hangul_ratio("English only") == 0.0


def test_pool_deduplicates_normalized_text_and_uses_seeded_shuffle():
    records = [
        {"id": "a", "text": "  한국어 문서 A  "},
        {"id": "duplicate", "text": "한국어 문서 A"},
        {"id": "b", "text": "한국어 문서 B"},
        {"id": "c", "text": "한국어 문서 C"},
    ]
    first = build_prompt_pool(records, limit=3, seed=42, min_chars=2, min_hangul_ratio=0.5)
    second = build_prompt_pool(records, limit=3, seed=42, min_chars=2, min_hangul_ratio=0.5)
    assert first == second
    assert len(first) == 3
    assert len({row["raw_text_hash"] for row in first}) == 3
    assert sum(row["text"] == "한국어 문서 A" for row in first) == 1
    assert all(row["doc_id"] for row in first)


def test_pool_rejects_insufficient_eligible_documents():
    records = [{"id": "a", "text": "English text"}]
    try:
        build_prompt_pool(records, limit=1, min_chars=2, min_hangul_ratio=0.5)
    except ValueError as exc:
        assert "eligible" in str(exc)
    else:
        raise AssertionError("Expected insufficient pool to fail")
