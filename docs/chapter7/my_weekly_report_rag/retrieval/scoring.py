#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import re
from collections import Counter
from typing import Iterable, List

import numpy as np

from models import ChunkMetadata


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


def batch_cosine_similarity(query_vector: List[float], candidate_vectors) -> np.ndarray:
    if isinstance(candidate_vectors, np.ndarray):
        matrix = candidate_vectors.astype(np.float32)
        if matrix.size == 0:
            return np.array([], dtype=np.float32)
    else:
        if len(candidate_vectors) == 0:
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
        return np.zeros(len(matrix), dtype=np.float32)

    matrix_norms = np.linalg.norm(matrix, axis=1)
    valid = matrix_norms > 0
    scores = np.zeros(len(matrix), dtype=np.float32)
    if np.any(valid):
        normalized_query = query / query_norm
        normalized_matrix = matrix[valid] / matrix_norms[valid, np.newaxis]
        scores[valid] = normalized_matrix @ normalized_query
    return scores


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text.lower())
    return [token for token in tokens if len(token) >= 2]


def bm25_score(query: str, document: str, corpus_avg_len: float, k1: float = 1.5, b: float = 0.75) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    doc_tokens = _tokenize(document)
    if not doc_tokens:
        return 0.0
    doc_len = len(doc_tokens)
    term_freq = Counter(doc_tokens)
    score = 0.0
    for token in set(query_tokens):
        tf = term_freq.get(token, 0)
        if tf == 0:
            continue
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_len / max(corpus_avg_len, 1.0))
        score += math.log(1 + numerator / denominator)
    return score


def build_search_text(metadata: ChunkMetadata, body_text: str) -> str:
    return " ".join([
        metadata.project,
        metadata.author,
        metadata.quarter,
        metadata.section_type,
        body_text,
    ])


def score_candidates(
    query: str,
    texts: Iterable[str],
    metadata_list: Iterable[ChunkMetadata],
) -> List[float]:
    search_docs = [
        build_search_text(meta, text.split("\n", 1)[-1] if text else "")
        for text, meta in zip(texts, metadata_list)
    ]
    if not search_docs:
        return []
    avg_len = sum(len(_tokenize(doc)) for doc in search_docs) / len(search_docs)
    raw_scores = [bm25_score(query, doc, avg_len) for doc in search_docs]
    max_score = max(raw_scores) if raw_scores else 0.0
    if max_score <= 0:
        return [0.0 for _ in raw_scores]
    return [score / max_score for score in raw_scores]


def merge_candidate_indices(
    ranked: List[tuple[int, float, float, float]],
    candidate_indices: List[int],
    keyword_scores: List[float],
    pool_size: int,
) -> List[int]:
    vector_top = [index for index, _, _, _ in ranked[:pool_size]]
    keyword_sorted = sorted(
        [(candidate_indices[offset], keyword_scores[offset]) for offset in range(len(candidate_indices))],
        key=lambda item: item[1],
        reverse=True,
    )
    keyword_top = [index for index, score in keyword_sorted[:pool_size] if score > 0]
    merged = list(dict.fromkeys(vector_top + keyword_top))
    return merged or vector_top
