#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from retrieval.scoring import cosine_similarity, merge_candidate_indices


class ScoringTestCase(unittest.TestCase):
    def test_cosine_similarity_identical_vectors(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_merge_candidate_indices_merges_vector_and_keyword_pools(self) -> None:
        ranked = [(0, 0.9, 0.9, 0.0), (1, 0.8, 0.8, 0.0), (2, 0.7, 0.7, 0.0)]
        candidate_indices = [10, 11, 12]
        keyword_scores = [0.1, 0.9, 0.2]
        merged = merge_candidate_indices(ranked, candidate_indices, keyword_scores, pool_size=2)
        self.assertEqual(merged[:2], [0, 1])
        self.assertIn(11, merged)


if __name__ == "__main__":
    unittest.main()
