#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""混合检索引擎：向量相似度 + BM25 融合打分。"""

import logging
from typing import Dict, List, Optional, Tuple

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

RankedItem = Tuple[int, float, float, float]  # index, combined, vector, keyword


class SearchEngine:
    """向量 + BM25 混合检索，支持年月过滤与多种检索模式。"""

    def __init__(self, store: IndexStore) -> None:
        self.store = store

    def _filter_indices(self, year: Optional[int] = None, month: Optional[int] = None) -> List[int]:
        """按年月缩小候选集，未指定年份则返回全部索引。"""
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

    def _rank_candidates(
        self,
        query: str,
        candidate_indices: List[int],
        embedding_model: EmbeddingProvider,
    ) -> Tuple[List[RankedItem], List[float]]:
        """对候选 chunk 分别计算向量相似度与 BM25，加权融合后排序。"""
        query_vector = embedding_model.get_embedding(query, kind="query")
        vector_scores = batch_cosine_similarity(
            query_vector,
            self.store.vector_matrix[candidate_indices],
        )

        texts = [self.store.records[i].text for i in candidate_indices]
        metadata_list = [self.store.records[i].metadata for i in candidate_indices]
        keyword_scores = score_candidates(query, texts, metadata_list)

        ranked: List[RankedItem] = []
        for offset, index in enumerate(candidate_indices):
            vector_score = float(vector_scores[offset])
            keyword_score = float(keyword_scores[offset])
            combined = VECTOR_SCORE_WEIGHT * vector_score + BM25_SCORE_WEIGHT * keyword_score
            ranked.append((index, combined, vector_score, keyword_score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked, keyword_scores

    def _build_results(
        self,
        ranked: List[RankedItem],
        candidate_indices: List[int],
        keyword_scores: List[float],
        threshold: float,
        k: int,
    ) -> List[SearchResult]:
        """合并向量/BM25 两路 top 候选，过滤低于阈值的结果。"""
        pool_size = max(k * SEARCH_CANDIDATE_POOL_FACTOR * 2, k)
        # 取两路检索的并集，避免单路漏召回
        merged_indices = merge_candidate_indices(ranked, candidate_indices, keyword_scores, pool_size)
        score_map: Dict[int, Tuple[float, float, float]] = {
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
        return sorted(results, key=lambda item: item.score, reverse=True)

    def query(
        self,
        query: str,
        embedding_model: EmbeddingProvider,
        k: int = DEFAULT_K,
        year: Optional[int] = None,
        month: Optional[int] = None,
        mode: SearchMode = DEFAULT_SEARCH_MODE,
    ) -> List[SearchResult]:
        """检索主流程：过滤 → 打分 → 阈值截断 → 按模式去重/排序。"""
        if not self.store.records:
            return []

        threshold = MODE_THRESHOLDS.get(mode, MODE_THRESHOLDS["latest"])
        candidate_indices = self._filter_indices(year=year, month=month)
        if year is not None and not candidate_indices:
            return []

        ranked, keyword_scores = self._rank_candidates(query, candidate_indices, embedding_model)
        results = self._build_results(ranked, candidate_indices, keyword_scores, threshold, k)

        logger.debug(
            "检索 mode=%s threshold=%.2f 候选=%s 命中=%s",
            mode,
            threshold,
            len(ranked),
            len(results),
        )
        return apply_search_mode(results, mode=mode, k=k)
