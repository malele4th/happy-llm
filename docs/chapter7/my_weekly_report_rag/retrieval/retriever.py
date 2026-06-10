#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import List, Optional

from config import (
    BM25_SCORE_WEIGHT,
    DEFAULT_K,
    MODE_THRESHOLDS,
    SEARCH_CANDIDATE_POOL_FACTOR,
    VECTOR_SCORE_WEIGHT,
)
from indexing.store import IndexStore
from models import DEFAULT_SEARCH_MODE, SearchMode, SearchResult
from providers.embeddings import EmbeddingProvider
from retrieval.scoring import batch_cosine_similarity, merge_candidate_indices, score_candidates
from retrieval.search_modes import apply_search_mode

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, store: IndexStore) -> None:
        self.store = store

    def _filter_indices(self, year: Optional[int] = None, month: Optional[int] = None) -> List[int]:
        if year is None:
            return list(range(len(self.store.records)))

        indices = []
        for index, record in enumerate(self.store.records):
            report_date = record.metadata.report_date
            if len(report_date) < 6:
                continue
            if int(report_date[:4]) != year:
                continue
            if month is not None and int(report_date[4:6]) != month:
                continue
            indices.append(index)
        return indices

    def query(
        self,
        query: str,
        embedding_model: EmbeddingProvider,
        k: int = DEFAULT_K,
        year: Optional[int] = None,
        month: Optional[int] = None,
        mode: SearchMode = DEFAULT_SEARCH_MODE,
    ) -> List[SearchResult]:
        if not self.store.records:
            return []

        threshold = MODE_THRESHOLDS.get(mode, MODE_THRESHOLDS["latest"])
        candidate_indices = self._filter_indices(year=year, month=month)
        if year is not None and not candidate_indices:
            return []

        query_vector = embedding_model.get_embedding(query, kind="query")
        vector_scores = batch_cosine_similarity(
            query_vector,
            self.store.vector_matrix[candidate_indices],
        )

        texts = [self.store.records[index].text for index in candidate_indices]
        metadata_list = [self.store.records[index].metadata for index in candidate_indices]
        keyword_scores = score_candidates(query, texts, metadata_list)

        ranked = []
        for offset, index in enumerate(candidate_indices):
            vector_score = float(vector_scores[offset])
            keyword_score = float(keyword_scores[offset])
            combined = VECTOR_SCORE_WEIGHT * vector_score + BM25_SCORE_WEIGHT * keyword_score
            ranked.append((index, combined, vector_score, keyword_score))
        ranked.sort(key=lambda item: item[1], reverse=True)

        pool_size = max(k * SEARCH_CANDIDATE_POOL_FACTOR * 2, k)
        merged_indices = merge_candidate_indices(ranked, candidate_indices, keyword_scores, pool_size)
        score_map = {
            index: (combined, vector_score, keyword_score)
            for index, combined, vector_score, keyword_score in ranked
        }

        results: List[SearchResult] = []
        for index in merged_indices:
            combined, vector_score, keyword_score = score_map[index]
            if combined < threshold:
                continue
            record = self.store.records[index]
            results.append(
                SearchResult(
                    text=record.text,
                    score=combined,
                    metadata=record.metadata,
                    vector_score=vector_score,
                    keyword_score=keyword_score,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        logger.debug(
            "检索 mode=%s threshold=%.2f 候选=%s 命中=%s",
            mode,
            threshold,
            len(merged_indices),
            len(results),
        )
        return apply_search_mode(results, mode=mode, k=k)
