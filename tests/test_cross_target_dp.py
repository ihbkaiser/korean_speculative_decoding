from __future__ import annotations

import hashlib
import unittest

import numpy as np

from src.cross_target_dp import (
    add_matched_morphology_positions,
    exact_bottom_labels,
    pairwise_overlap,
    prefix_hash,
    rank_correlation,
    sample_eojeol_positions,
    within_document_permutation,
)
from scripts.analyze_cross_target_dp import _permutation_main


class _Token:
    def __init__(self, form, tag, start):
        self.form = form
        self.tag = tag
        self.start = start
        self.len = len(form)


class _FakeKiwi:
    def tokenize(self, text):
        # The synthetic test eojeol below is "가나다는".
        if text == "가나다는":
            return [_Token("가나다", "NNG", 0), _Token("는", "JX", 3)]
        return [_Token(text, "NNG", 0)]


class CrossTargetPositionTests(unittest.TestCase):
    def test_shared_prefix_hash_and_deterministic_eojeol_sample(self):
        text = "가나다는 가나다는 가나다는"
        doc = {
            "doc_id": "doc-1",
            "source_index": 4,
            "pool_index": 5,
            "text": text,
            "raw_text_hash": hashlib.sha256(text.encode()).hexdigest(),
        }
        positions, metadata, included = sample_eojeol_positions(
            [doc],
            max_documents=1,
            positions_per_document=2,
            context_window_chars=5,
            min_context_chars=1,
        )
        self.assertEqual(len(positions), 2)
        self.assertEqual(metadata["documents_included"], 1)
        self.assertEqual(set(included), {"doc-1"})
        for row in positions:
            expected = text[row["context_start"] : row["char_offset"]]
            self.assertEqual(row["shared_prefix_sha256"], prefix_hash(expected))
            self.assertEqual(row["context_chars"], len(expected))

    def test_morphology_positions_are_matched_within_same_eojeol(self):
        text = "가나다는 가나다는 가나다는"
        doc = {"doc_id": "doc-1", "text": text, "raw_text_hash": hashlib.sha256(text.encode()).hexdigest()}
        positions, _, included = sample_eojeol_positions(
            [doc], max_documents=1, positions_per_document=2,
            context_window_chars=5, min_context_chars=1,
        )
        morph, meta = add_matched_morphology_positions(
            positions, included, _FakeKiwi(), anchor_words_per_document=1, context_window_chars=5
        )
        self.assertEqual(meta["matched_word_pairs"], 1)
        self.assertEqual([row["position_type"] for row in morph], ["morpheme_boundary", "morpheme_internal"])
        self.assertEqual(morph[0]["char_offset"], 8)
        self.assertEqual(morph[1]["char_offset"], 7)
        self.assertEqual(morph[0]["matched_word_id"], morph[1]["matched_word_id"])

    def test_exact_bottom_labels_and_overlap_metrics(self):
        labels = exact_bottom_labels([4, 1, 3, 2, 0], 0.6)
        self.assertEqual(labels.tolist(), [False, True, False, True, True])
        metrics = pairwise_overlap(labels, [False, True, True, False, True])
        self.assertEqual(metrics["intersection_size"], 2)
        self.assertAlmostEqual(metrics["jaccard"], 2 / 4)
        self.assertAlmostEqual(metrics["overlap_fraction"], 2 / 5)
        self.assertAlmostEqual(metrics["overlap_pct"], 40.0)
        self.assertAlmostEqual(metrics["conditional_b_given_a"], 2 / 3)
        self.assertAlmostEqual(metrics["independence_expected_overlap_fraction"], 0.36)
        self.assertAlmostEqual(metrics["overlap_enrichment"], (2 / 5) / 0.36)

    def test_rank_correlation_is_spearman_and_document_permutation_preserves_counts(self):
        self.assertAlmostEqual(rank_correlation([1, 2, 3], [4, 5, 6]), 1.0)
        docs = np.array(["a"] * 4 + ["b"] * 4)
        labels = np.array([
            [1, 0], [1, 1], [0, 0], [0, 1],
            [1, 1], [0, 0], [1, 0], [0, 1],
        ], dtype=bool)
        permuted = within_document_permutation(labels, docs, np.random.default_rng(4))
        for model in range(labels.shape[1]):
            for doc in ("a", "b"):
                mask = docs == doc
                self.assertEqual(int(permuted[mask, model].sum()), int(labels[mask, model].sum()))

    def test_analysis_permutation_reports_percent_and_three_plus_bucket(self):
        documents = 20
        positions_per_document = 10
        doc_ids = np.repeat([f"doc-{index}" for index in range(documents)], positions_per_document)
        labels = np.zeros((len(doc_ids), 5), dtype=bool)
        for doc_index in range(documents):
            start = doc_index * positions_per_document
            for model_index in range(labels.shape[1]):
                labels[start + model_index % positions_per_document, model_index] = True
                labels[start + (model_index + 1) % positions_per_document, model_index] = True
        pair_rows, bucket_rows, exact_rows = _permutation_main(
            labels, doc_ids, ("qwen4", "qwen8", "qwen14", "gemma12", "ministral8"),
            replicates=20, seed=11,
        )
        self.assertEqual(len(pair_rows), 20)
        self.assertAlmostEqual(pair_rows[1]["observed"], 10.0)
        self.assertEqual([row["target_count_bucket"] for row in bucket_rows], ["0", "exactly_1", "exactly_2", "3+"])
        self.assertAlmostEqual(sum(row["observed_fraction"] for row in bucket_rows), 1.0)
        self.assertAlmostEqual(sum(row["permutation_mean_fraction"] for row in bucket_rows), 1.0)
        self.assertEqual(len(exact_rows), 6)


if __name__ == "__main__":
    unittest.main()
