#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import List, Optional, Tuple

from config import (
    BM25_SCORE_WEIGHT,
    DEFAULT_K,
    MODE_THRESHOLDS,
    SEARCH_CANDIDATE_POOL_FACTOR,
    VECTOR_SCORE_WEIGHT,
)
from embeddings import EmbeddingProvider
from index_store import IndexStore
from keyword_search import score_candidates
from models import DEFAULT_SEARCH_MODE, ChunkMetadata, SearchMode, SearchResult
from similarity import batch_cosine_similarity

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, store: IndexStore) -> None:
        self.store = store

    def _filter_indices(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None,
    ) -> List[int]:
        if year is None:
            return list(range(len(self.store.records)))

        indices = []
        for index, record in enumerate(self.store.records):
            report_date = record.metadata.report_date
            if len(report_date) < 6:
                continue
            doc_year = int(report_date[:4])
            doc_month = int(report_date[4:6])
            if doc_year != year:
                continue
            if month is not None and doc_month != month:
                continue
            indices.append(index)
        return indices

    @staticmethod
    def _dedupe_latest(results: List[SearchResult]) -> List[SearchResult]:
        by_project: dict[str, SearchResult] = {}
        for result in results:
            project = result.metadata.project
            report_date = result.metadata.report_date
            if project not in by_project:
                by_project[project] = result
                continue
            existing = by_project[project]
            if report_date > existing.metadata.report_date or (
                report_date == existing.metadata.report_date
                and result.score > existing.score
            ):
                by_project[project] = result
        return sorted(by_project.values(), key=lambda item: item.score, reverse=True)

    @staticmethod
    def _dedupe_compare(results: List[SearchResult]) -> List[SearchResult]:
        by_key: dict[Tuple[str, str], SearchResult] = {}
        for result in results:
            key = result.metadata.compare_key()
            if key not in by_key or result.score > by_key[key].score:
                by_key[key] = result
        return sorted(by_key.values(), key=lambda item: item.metadata.sort_key())

    @staticmethod
    def _apply_search_mode(
        results: List[SearchResult],
        mode: SearchMode,
        k: int,
    ) -> List[SearchResult]:
        pool_size = max(k * SEARCH_CANDIDATE_POOL_FACTOR, k)
        candidates = results[:pool_size]

        if mode == "timeline":
            ordered = sorted(candidates, key=lambda item: item.metadata.sort_key())
            return ordered[:k]
        if mode == "compare":
            return Retriever._dedupe_compare(candidates)[:k]
        return Retriever._dedupe_latest(candidates)[:k]

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
        candidate_vectors = self.store.vector_matrix[candidate_indices]
        vector_scores = batch_cosine_similarity(query_vector, candidate_vectors)

        texts = [self.store.records[index].text for index in candidate_indices]
        metadata_list = [self.store.records[index].metadata for index in candidate_indices]
        keyword_scores = score_candidates(query, texts, metadata_list)

        ranked: List[Tuple[int, float, float, float]] = []
        for offset, index in enumerate(candidate_indices):
            vector_score = float(vector_scores[offset])
            keyword_score = float(keyword_scores[offset])
            combined = (
                VECTOR_SCORE_WEIGHT * vector_score
                + BM25_SCORE_WEIGHT * keyword_score
            )
            ranked.append((index, combined, vector_score, keyword_score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        pool_size = max(k * SEARCH_CANDIDATE_POOL_FACTOR * 2, k)

        vector_top = [index for index, _, _, _ in ranked[:pool_size]]
        keyword_sorted = sorted(
            [
                (candidate_indices[offset], keyword_scores[offset])
                for offset in range(len(candidate_indices))
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        keyword_top = [index for index, score in keyword_sorted[:pool_size] if score > 0]
        merged_indices = list(dict.fromkeys(vector_top + keyword_top))
        if not merged_indices:
            merged_indices = vector_top

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
        return self._apply_search_mode(results, mode=mode, k=k)
