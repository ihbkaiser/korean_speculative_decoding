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
        self.assertAlmostEqual(metrics["overlap_pct"], 2 / 5)

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


if __name__ == "__main__":
    unittest.main()
