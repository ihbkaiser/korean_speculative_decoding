from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from src.table1_alignment import build_aligned_token_row
from src.table1_morphology import _continuation_window, build_table1_outputs


@pytest.mark.parametrize("prompt,generated", [("a ", " b"), ("a", "b c")])
def test_excluded_unmatched_tokens_keep_valid_spans_and_denominator(prompt, generated):
    from types import SimpleNamespace
    from src.table1_morphology import _make_prompt_alignment_rows
    from src.table1_alignment import assert_alignment_invariants, aggregate_alignment_rows
    from tests.test_pipeline import CharTokenizer

    tokenizer = CharTokenizer()
    class KiwiStub:
        def tokenize(self, text):
            return [SimpleNamespace(form=text, tag="NNG", start=0, len=len(text))]

    reference = {"doc_id": "p1", "pair_id": "Q1", "raw_text_hash": "h1",
                 "prompt_token_ids": tokenizer(prompt)["input_ids"],
                 "target_continuation_token_ids": tokenizer(generated)["input_ids"]}
    rows = _make_prompt_alignment_rows(reference, tokenizer=tokenizer, kiwi=KiwiStub())
    assert len(rows) == len(generated)
    summary = aggregate_alignment_rows(rows)
    assert summary["visible_candidate_count"] == len(generated)
    assert any(row["alignment_status"] == "EXCLUDED" for row in rows)
    assert_alignment_invariants(rows)


def test_zero_width_kiwi_analysis_is_preserved_raw_and_stays_excluded():
    from types import SimpleNamespace
    from src.table1_morphology import _make_prompt_alignment_rows
    from src.table1_alignment import assert_alignment_invariants, aggregate_alignment_rows
    from tests.test_pipeline import CharTokenizer

    class KiwiStub:
        def tokenize(self, text):
            return [SimpleNamespace(form="a", tag="NNG", start=0, len=1),
                    SimpleNamespace(form="이", tag="VCP", start=1, len=0)]

    tok = CharTokenizer()
    reference = {"doc_id": "p1", "pair_id": "Q1", "raw_text_hash": "h1",
                 "prompt_token_ids": tok(" ")["input_ids"],
                 "target_continuation_token_ids": tok("a")["input_ids"]}
    rows = _make_prompt_alignment_rows(reference, tokenizer=tok, kiwi=KiwiStub())
    assert rows[0]["alignment_exclusion_reason"] == "ambiguous_morphology_span"
    assert rows[0]["morpheme_char_spans"] == []
    assert rows[0]["raw_morpheme_char_spans"] == [[0, 1], [1, 1]]
    summary = aggregate_alignment_rows(rows)
    assert summary["visible_candidate_count"] == 1 and summary["eligible_count"] == 0
    assert_alignment_invariants(rows)


def test_table_row_handles_multitoken_numpy_array_from_parquet():
    from src.table1_morphology import _table_row

    reference = {"doc_id": "p1", "completed": True, "exact_sd_target": True,
                 "target_continuation_token_ids": np.array([2, 3])}
    aligned = []
    for position in range(2):
        row = build_aligned_token_row(
            "학", (0, 1), [{"surface": "학", "pos": "NNG", "start": 0, "end": 1}],
            prompt_id="p1",
        )
        row.update({"doc_id": "p1", "pair_id": "Q1", "generated_token_position": position,
                    "visible_candidate": True})
        aligned.append(row)
    row, _, _ = _table_row("Q1", _config()["pairs"]["Q1"], [reference], aligned, True, [])
    assert row["Generated tokens"] == 2


def _config():
    pairs = {}
    for pair_id in ("Q1", "Q2", "Q3", "M1", "G1"):
        pairs[pair_id] = {
            "family": "Test",
            "draft": f"draft/{pair_id}",
            "target": f"target/{pair_id}",
            "prompt_split": "common_20k",
            "prompt_count": 1,
            "support_threshold_cross": 100,
            "support_threshold_within": 100,
        }
    return {
        "paths": {
            "prompts": "data/prompts_40k.parquet",
            "runs": "runs/table1",
            "results": "results",
        },
        "pairs": pairs,
    }


def test_build_table1_writes_five_rows_and_complete_status(tmp_path: Path):
    (tmp_path / "data").mkdir()
    pd.DataFrame([{
        "doc_id": "doc-1",
        "raw_text_hash": "hash-1",
        "text": "한국어 문서",
        "pool_index": 0,
    }]).to_parquet(tmp_path / "data/prompts_40k.parquet", index=False)
    config = _config()
    for pair_id in config["pairs"]:
        shard = tmp_path / "runs/table1" / pair_id / "shards/shard-00000-of-00001"
        shard.mkdir(parents=True)
        reference = {
            "doc_id": "doc-1",
            "raw_text_hash": "hash-1",
            "pair_id": pair_id,
            "prompt_token_ids": [1],
            "target_continuation_token_ids": [2],
            "target_continuation_text": "학",
            "completed": True,
            "exact_sd_target": True,
        }
        aligned = build_aligned_token_row(
            "학", (0, 1), [{"surface": "학", "pos": "NNG", "start": 0, "end": 1}],
            prompt_id="doc-1",
        )
        aligned.update({"pair_id": pair_id, "doc_id": "doc-1", "raw_text_hash": "hash-1", "visible_candidate": True})
        pd.DataFrame([reference]).to_parquet(shard / "references.parquet", index=False)
        pd.DataFrame([aligned]).to_parquet(shard / "aligned_tokens.parquet", index=False)
        (shard / "COMPLETE").write_text("ok\n", encoding="utf-8")
        (shard / "ALIGNMENT_COMPLETE").write_text("ok\n", encoding="utf-8")
    result = build_table1_outputs(root=tmp_path, config=config)
    assert result["status"] == "COMPLETE"
    table = pd.read_csv(tmp_path / "results/table1_models_data_alignment.csv")
    assert len(table) == 5
    assert set(table["Status"]) == {"COMPLETE"}
    assert (tmp_path / "results/table1_models_data_alignment.md").exists()
    assert (tmp_path / "results/table1_models_data_alignment.tex").exists()
    assert (tmp_path / "results/table1_exclusion_breakdown.csv").exists()
    assert (tmp_path / "results/boundary_support_by_pair.csv").exists()


def test_continuation_window_drops_prompt_partial_eojeol():
    start, continuation = _continuation_window("학생은 학교다", prompt_char_end=2)

    assert start == 4
    assert continuation == "학교다"


def test_continuation_window_keeps_boundary_aligned_prompt():
    start, continuation = _continuation_window("학생은 학교다", prompt_char_end=4)

    assert start == 4
    assert continuation == "학교다"
