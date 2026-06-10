#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List

import numpy as np


def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
    v1 = np.array(vector1, dtype=np.float32)
    v2 = np.array(vector2, dtype=np.float32)
    if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
        return 0.0
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    if magnitude == 0:
        return 0.0
    return float(dot_product / magnitude)


def batch_cosine_similarity(
    query_vector: List[float],
    candidate_vectors,
) -> np.ndarray:
    if isinstance(candidate_vectors, np.ndarray):
        matrix = candidate_vectors.astype(np.float32)
        if matrix.size == 0:
            return np.array([], dtype=np.float32)
    else:
        if not candidate_vectors:
            return np.array([], dtype=np.float32)
        matrix = np.array(candidate_vectors, dtype=np.float32)

    query = np.array(query_vector, dtype=np.float32)

    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(matrix)):
        return np.array(
            [cosine_similarity(query_vector, vec) for vec in candidate_vectors],
            dtype=np.float32,
        )

    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(len(candidate_vectors), dtype=np.float32)

    matrix_norms = np.linalg.norm(matrix, axis=1)
    valid = matrix_norms > 0
    scores = np.zeros(len(candidate_vectors), dtype=np.float32)

    if np.any(valid):
        normalized_query = query / query_norm
        normalized_matrix = matrix[valid] / matrix_norms[valid, np.newaxis]
        scores[valid] = normalized_matrix @ normalized_query

    return scores
