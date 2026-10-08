from __future__ import annotations

import pytest

from src.table1_alignment import (
    aggregate_alignment_rows,
    assert_alignment_invariants,
    build_aligned_token_row,
    classify_token_alignment,
    support_counts_by_fine_boundary,
)


MORPHEMES = [
    {"surface": "학", "pos": "NNG", "start": 0, "end": 1},
    {"surface": "생", "pos": "JKS", "start": 1, "end": 2},
    {"surface": "들", "pos": "XSN", "start": 2, "end": 3},
]

WITHIN_MORPHEMES = [
    {"surface": "학생", "pos": "NNG", "start": 0, "end": 2},
    {"surface": "들", "pos": "XSN", "start": 2, "end": 3},
]


def test_classifies_exact_within_and_single_boundary_cross():
    exact = classify_token_alignment((0, 1), MORPHEMES, "학생들", "학", "학")
    assert exact["alignment_class"] == "EXACT"
    assert exact["crossed_boundary_count"] == 0
    assert exact["alignment_status"] == "ALIGNED"

    within = classify_token_alignment((0, 1), [{"surface": "학생", "pos": "NNG", "start": 0, "end": 2}], "학생", "학", "학")
    assert within["alignment_class"] == "WITHIN_SPLIT"
    assert within["nearest_boundary_position"] is None
    assert within["nearest_boundary_distance"] is None

    cross = classify_token_alignment((0, 2), MORPHEMES, "학생들", "학생", "학생")
    assert cross["alignment_class"] == "CROSS"
    assert cross["crossed_boundary_count"] == 1
    assert cross["left_pos"] == "NNG"
    assert cross["right_pos"] == "JKS"
    assert cross["fine_boundary"] == "NNG→JKS"
    assert cross["coarse_boundary"] == "NOMINAL_TO_PARTICLE"
    assert cross["left_chars_in_token"] == 1
    assert cross["right_chars_in_token"] == 1
    assert cross["boundary_position_ratio"] == pytest.approx(0.5)


def test_coarse_boundary_groups_are_named_deterministically():
    cases = [
        ("VV", "EF", "PREDICATE_TO_ENDING"),
        ("EP", "EF", "ENDING_TO_ENDING"),
        ("NNG", "VV", "LEXICAL_TO_LEXICAL"),
        ("SF", "JKS", "OTHER"),
    ]
    for left_pos, right_pos, expected in cases:
        text = "가나"
        result = classify_token_alignment(
            (0, 2),
            [
                {"surface": "가", "pos": left_pos, "start": 0, "end": 1},
                {"surface": "나", "pos": right_pos, "start": 1, "end": 2},
            ],
            text,
            "가나",
            "가나",
        )
        assert result["coarse_boundary"] == expected


def test_multi_boundary_cross_exposes_coarse_boundaries_without_dropping_scalar_fields():
    result = classify_token_alignment(
        (0, 3),
        [
            {"surface": "가", "pos": "NNG", "start": 0, "end": 1},
            {"surface": "나", "pos": "VV", "start": 1, "end": 2},
            {"surface": "다", "pos": "EF", "start": 2, "end": 3},
        ],
        "가나다",
        "가나다",
        "가나다",
    )
    assert result["alignment_class"] == "CROSS"
    assert result["crossed_boundary_count"] == 2
    assert result["fine_boundaries"] == ["NNG→VV", "VV→EF"]
    assert result["coarse_boundaries"] == ["LEXICAL_TO_LEXICAL", "PREDICATE_TO_ENDING"]
    assert result["fine_boundary"] is None
    assert result["coarse_boundary"] is None


def test_alignment_invariants_reject_inconsistent_lengths_and_malformed_spans():
    good = build_aligned_token_row(
        "학생",
        (0, 1),
        [{"surface": "학", "pos": "NNG", "start": 0, "end": 1}, {"surface": "생", "pos": "NNG", "start": 1, "end": 2}],
    )

    wrong_length = dict(good)
    wrong_length["eojeol_char_len"] = 99
    with pytest.raises(AssertionError, match="eojeol_char_len"):
        assert_alignment_invariants([wrong_length])

    zero_width = dict(good)
    zero_width["token_char_span"] = [1, 1]
    with pytest.raises(AssertionError, match="positive"):
        assert_alignment_invariants([zero_width])

    malformed_token_span = dict(good)
    malformed_token_span["token_char_span"] = "0:1"
    with pytest.raises(AssertionError, match="token span"):
        assert_alignment_invariants([malformed_token_span])

    malformed_morpheme_span = dict(good)
    malformed_morpheme_span["morpheme_char_spans"] = [[0, "bad"]]
    with pytest.raises(AssertionError, match="morpheme span"):
        assert_alignment_invariants([malformed_morpheme_span])


def test_alignment_invariants_allow_zero_width_non_visible_special_rows():
    special = build_aligned_token_row(
        "",
        (0, 0),
        [],
        visible_candidate=False,
        exclusion_reason="special_or_non_visible_token",
    )

    assert special["token_char_span"] == [0, 0]
    assert_alignment_invariants([special])


def test_alignment_invariants_reject_generated_token_count_mismatch():
    row = build_aligned_token_row(
        "í•™",
        (0, 1),
        [{"surface": "í•™", "pos": "NNG", "start": 0, "end": 1}],
        prompt_id="p1",
    )

    with pytest.raises(AssertionError, match="generated token count"):
        assert_alignment_invariants(
            [row],
            completed_prompt_metadata=[
                {"prompt_id": "p1", "completed": True, "generated_token_count": 2}
            ],
        )


def test_within_split_nearest_boundary_tie_is_not_resolved():
    morphemes = [
        {"surface": "가", "pos": "NNG", "start": 0, "end": 1},
        {"surface": "나다라", "pos": "VV", "start": 1, "end": 4},
        {"surface": "마", "pos": "EF", "start": 4, "end": 5},
    ]
    result = classify_token_alignment((2, 3), morphemes, "가나다라마", "다", "다")
    assert result["alignment_class"] == "WITHIN_SPLIT"
    assert result["nearest_boundary_ambiguous"] is True
    assert result["nearest_boundary_positions"] == [1, 4]
    assert result["nearest_boundary_distance"] == 1
    assert result["nearest_left_pos"] is None
    assert result["nearest_right_pos"] is None
    assert result["nearest_fine_boundary"] is None
    assert result["nearest_coarse_boundary_candidates"] == ["LEXICAL_TO_LEXICAL", "PREDICATE_TO_ENDING"]


def test_invalid_alignment_is_explicitly_excluded():
    mismatch = classify_token_alignment((0, 1), MORPHEMES, "학생들", "x", "x")
    assert mismatch["alignment_class"] == "OTHER/AMBIGUOUS"
    assert mismatch["alignment_status"] == "EXCLUDED"
    assert mismatch["alignment_exclusion_reason"] == "token_core_text_mismatch"

    malformed = classify_token_alignment(
        (0, 1),
        [{"surface": "학", "pos": "NNG", "start": 1, "end": 0}],
        "학",
        "학",
        "학",
    )
    assert malformed["alignment_class"] == "OTHER/AMBIGUOUS"
    assert malformed["alignment_exclusion_reason"] == "ambiguous_morphology_span"

    non_integral_morpheme = classify_token_alignment(
        (0, 1),
        [{"surface": "학", "pos": "NNG", "start": 0.9, "end": 1}],
        "학",
        "학",
        "학",
    )
    assert non_integral_morpheme["alignment_class"] == "OTHER/AMBIGUOUS"
    assert non_integral_morpheme["alignment_exclusion_reason"] == "ambiguous_morphology_span"

    malformed_row = build_aligned_token_row(
        "학",
        (0, 1),
        [{"surface": "학", "pos": "NNG", "start": 0.9, "end": 1}],
    )
    with pytest.raises(AssertionError, match="morpheme span"):
        assert_alignment_invariants([malformed_row])


def test_build_row_contains_table1_schema_and_position_fields():
    row = build_aligned_token_row(
        "학생들",
        (1, 3),
        MORPHEMES,
        token_raw_text="▁생들",
        token_core_text="생들",
        token_index_in_eojeol=1,
        eojeol_token_count=3,
        prompt_id="p1",
    )
    assert row["eojeol_text"] == "학생들"
    assert row["eojeol_char_len"] == 3
    assert row["eojeol_token_count"] == 3
    assert row["token_index_in_eojeol"] == 1
    assert row["relative_position_in_eojeol"] == pytest.approx(0.5)
    assert row["is_first_token"] is False
    assert row["is_last_token"] is False
    assert row["morpheme_surfaces"] == ["학", "생", "들"]
    assert row["morpheme_pos_tags"] == ["NNG", "JKS", "XSN"]
    assert row["morpheme_char_spans"] == [[0, 1], [1, 2], [2, 3]]
    assert row["alignment_class"] == "CROSS"
    assert row["alignment_status"] == "ALIGNED"
    assert row["alignment_exclusion_reason"] is None
    assert row["token_raw_text"] == "▁생들"
    assert row["token_core_text"] == "생들"


def test_aggregate_reconciles_visible_eligible_and_exclusions_for_completed_prompts():
    rows = [
        build_aligned_token_row("학생들", (0, 1), MORPHEMES, token_index_in_eojeol=0, eojeol_token_count=3, prompt_id="p1"),
        build_aligned_token_row("학생들", (0, 2), MORPHEMES, token_index_in_eojeol=0, eojeol_token_count=3, prompt_id="p1"),
        build_aligned_token_row(
            "학생들",
            (1, 2),
            WITHIN_MORPHEMES,
            token_raw_text="생",
            token_core_text="생",
            token_index_in_eojeol=1,
            eojeol_token_count=3,
            prompt_id="p1",
        ),
        build_aligned_token_row(
            "학생들",
            (0, 1),
            MORPHEMES,
            token_index_in_eojeol=0,
            eojeol_token_count=3,
            prompt_id="p1",
            token_eojeol_count=2,
        ),
        build_aligned_token_row(
            "학생들",
            (0, 1),
            MORPHEMES,
            token_raw_text="wrong",
            token_core_text="wrong",
            token_index_in_eojeol=0,
            eojeol_token_count=3,
            prompt_id="p1",
        ),
        build_aligned_token_row(
            "학생들",
            (0, 1),
            MORPHEMES,
            token_index_in_eojeol=0,
            eojeol_token_count=3,
            prompt_id="p2",
            visible_candidate=False,
            exclusion_reason="special_or_non_visible_token",
        ),
    ]
    completed_metadata = [
        {"prompt_id": "p1", "completed": True, "generated_token_count": 5},
        {"prompt_id": "p2", "completed": False, "generated_token_count": 9},
    ]
    summary = aggregate_alignment_rows(rows, completed_prompt_metadata=completed_metadata)
    assert summary["completed_prompt_count"] == 1
    assert summary["generated_token_count"] == 5
    assert summary["visible_candidate_count"] == 5
    assert summary["eligible_count"] == 3
    assert summary["exact_count"] == 1
    assert summary["cross_count"] == 1
    assert summary["within_count"] == 1
    assert summary["single_boundary_cross_count"] == 1
    assert summary["exclusion_count"] == 2
    assert summary["exclusion_breakdown"] == {
        "multi_eojeol_token": 1,
        "token_core_text_mismatch": 1,
    }
    assert summary["aligned_percent"] == pytest.approx(60.0)
    assert summary["exclusion_percent"] == pytest.approx(40.0)
    assert summary["cross_percent"] == pytest.approx(100.0 / 3.0)
    assert summary["within_percent"] == pytest.approx(100.0 / 3.0)
    assert_alignment_invariants(rows, summary=summary, completed_prompt_metadata=completed_metadata)


def test_support_thresholds_distinguish_200_200_from_100_100():
    cross = {"alignment_class": "CROSS", "crossed_boundary_count": 1, "fine_boundary": "NNG→JKS", "alignment_status": "ALIGNED", "visible_candidate": True}
    within = {"alignment_class": "WITHIN_SPLIT", "nearest_fine_boundary": "NNG→JKS", "nearest_boundary_ambiguous": False, "alignment_status": "ALIGNED", "visible_candidate": True}
    rows_200 = [cross.copy() for _ in range(200)] + [within.copy() for _ in range(200)]
    rows_100 = [cross.copy() for _ in range(100)] + [within.copy() for _ in range(100)]

    q2 = support_counts_by_fine_boundary(rows_200, min_cross=200, min_within=200)
    other = support_counts_by_fine_boundary(rows_100, min_cross=100, min_within=100)
    too_strict = support_counts_by_fine_boundary(rows_100, min_cross=200, min_within=200)

    assert q2["by_fine_boundary"]["NNG→JKS"] == {
        "cross_count": 200,
        "within_count": 200,
        "supported": True,
    }
    assert other["supported_boundaries"] == ["NNG→JKS"]
    assert too_strict["supported_boundaries"] == []


def test_invariants_reject_bad_span_and_unreconciled_summary():
    bad_row = {
        "eojeol_text": "학",
        "eojeol_char_len": 1,
        "token_char_span": [0, 2],
        "morpheme_char_spans": [[0, 1]],
        "alignment_class": "EXACT",
        "alignment_status": "ALIGNED",
        "visible_candidate": True,
    }
    with pytest.raises(AssertionError, match="inside eojeol"):
        assert_alignment_invariants([bad_row])

    good_row = build_aligned_token_row("학", (0, 1), [{"surface": "학", "pos": "NNG", "start": 0, "end": 1}])
    bad_summary = aggregate_alignment_rows([good_row])
    bad_summary["eligible_count"] = 0
    with pytest.raises(AssertionError, match="aggregate mismatch"):
        assert_alignment_invariants([good_row], summary=bad_summary)


def test_invariants_reconcile_duplicate_token_span_fields_and_summary_denominators():
    row = build_aligned_token_row("학", (0, 1), [{"surface": "학", "pos": "NNG", "start": 0, "end": 1}])
    inconsistent_span = dict(row)
    inconsistent_span["token_start"] = 1
    inconsistent_span["token_end"] = 2
    with pytest.raises(AssertionError, match="reconcile"):
        assert_alignment_invariants([inconsistent_span])

    summary = aggregate_alignment_rows([row])
    summary["visible_candidate_count"] = 99
    with pytest.raises(AssertionError, match="aggregate mismatch"):
        assert_alignment_invariants([row], summary=summary)


def test_builder_rejects_non_boolean_visibility_and_non_int_counts():
    morphemes = [{"surface": "학", "pos": "NNG", "start": 0, "end": 1}]
    with pytest.raises(TypeError, match="visible_candidate"):
        build_aligned_token_row("학", (0, 1), morphemes, visible_candidate="yes")
    with pytest.raises(TypeError, match="token_eojeol_count"):
        build_aligned_token_row("학", (0, 1), morphemes, token_eojeol_count="1")
    with pytest.raises(TypeError, match="token_index_in_eojeol"):
        build_aligned_token_row("학", (0, 1), morphemes, token_index_in_eojeol=0.9)
    with pytest.raises(TypeError, match="token_eojeol_count"):
        build_aligned_token_row("학", (0, 1), morphemes, token_eojeol_count=True)


def test_builder_rejects_metadata_that_conflicts_with_canonical_fields():
    morphemes = [{"surface": "학", "pos": "NNG", "start": 0, "end": 1}]
    with pytest.raises(ValueError, match="canonical"):
        build_aligned_token_row(
            "학",
            (0, 1),
            morphemes,
            alignment_class="CROSS",
        )


def test_completion_metadata_validates_ids_status_counts_and_duplicates():
    row = build_aligned_token_row("학", (0, 1), [{"surface": "학", "pos": "NNG", "start": 0, "end": 1}], prompt_id="p1")
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_alignment_rows(
            [row],
            completed_prompt_metadata=[
                {"prompt_id": "p1", "completed": True, "generated_token_count": 1},
                {"prompt_id": "p1", "completed": False, "generated_token_count": 0},
            ],
        )
    with pytest.raises(ValueError, match="status"):
        aggregate_alignment_rows([row], completed_prompt_metadata=[{"prompt_id": "p1", "status": "done", "generated_token_count": 1}])
    with pytest.raises(TypeError, match="generated token count"):
        aggregate_alignment_rows([row], completed_prompt_metadata=[{"prompt_id": "p1", "completed": True, "generated_token_count": 0.9}])
    with pytest.raises(ValueError, match="prompt_id"):
        aggregate_alignment_rows([row], completed_prompt_metadata=[{"prompt_id": None, "completed": True, "generated_token_count": 1}])
