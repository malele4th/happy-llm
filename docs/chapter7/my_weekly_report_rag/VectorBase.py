#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

SearchMode = Literal["latest", "timeline", "compare"]

import numpy as np
from tqdm import tqdm

from config import (
    DEFAULT_K,
    KEYWORD_BOOST_MAX,
    KEYWORD_PROJECT_BOOST,
    KEYWORD_TOKEN_BOOST,
    PROJECT_KEYWORDS,
    SIMILARITY_THRESHOLD,
    STORAGE_PATH,
)
from Embeddings import BaseEmbeddings, cosine_similarity
from utils import extract_query_tokens


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict


class VectorStore:
    def __init__(
        self,
        document: List[str] = None,
        metadata: List[dict] = None,
    ) -> None:
        self.document = document or []
        self.metadata = metadata or []
        self.vectors: List[List[float]] = []

    def get_vector(
        self,
        embedding_model: BaseEmbeddings,
        texts: Optional[List[str]] = None,
    ) -> List[List[float]]:
        docs = texts if texts is not None else self.document
        embeddings = embedding_model.get_embeddings(docs)
        if texts is None:
            self.vectors = embeddings
        return embeddings

    def persist(self, path: str = STORAGE_PATH) -> None:
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/document.json", "w", encoding="utf-8") as f:
            json.dump(self.document, f, ensure_ascii=False)
        with open(f"{path}/metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)
        if self.vectors:
            with open(f"{path}/vectors.json", "w", encoding="utf-8") as f:
                json.dump(self.vectors, f)

    def load_vector(self, path: str = STORAGE_PATH) -> None:
        with open(f"{path}/vectors.json", "r", encoding="utf-8") as f:
            self.vectors = json.load(f)
        with open(f"{path}/document.json", "r", encoding="utf-8") as f:
            self.document = json.load(f)
        metadata_path = f"{path}/metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = [{} for _ in self.document]

    def _filter_indices(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None,
    ) -> List[int]:
        if year is None:
            return list(range(len(self.document)))

        indices = []
        for i, meta in enumerate(self.metadata):
            report_date = meta.get("report_date", "")
            if len(report_date) < 6:
                continue
            doc_year = int(report_date[:4])
            doc_month = int(report_date[4:6])
            if doc_year != year:
                continue
            if month is not None and doc_month != month:
                continue
            indices.append(i)
        return indices

    @staticmethod
    def _keyword_boost(query: str, meta: dict) -> float:
        project = meta.get("project", "")
        text = meta.get("source", "")
        boost = 0.0
        query_lower = query.lower()
        project_lower = project.lower()

        for keyword in PROJECT_KEYWORDS:
            kw = keyword.lower()
            if kw in query_lower and kw in project_lower:
                boost += KEYWORD_PROJECT_BOOST

        for token in extract_query_tokens(query):
            if token in project_lower or token in text.lower():
                boost += KEYWORD_TOKEN_BOOST

        return min(boost, KEYWORD_BOOST_MAX)

    @staticmethod
    def _batch_cosine_similarity(
        query_vector: List[float],
        candidate_vectors: List[List[float]],
    ) -> np.ndarray:
        if not candidate_vectors:
            return np.array([], dtype=np.float32)

        query = np.array(query_vector, dtype=np.float32)
        matrix = np.array(candidate_vectors, dtype=np.float32)

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

    @staticmethod
    def _dedupe_latest(results: List[SearchResult]) -> List[SearchResult]:
        """同项目保留最新周报，日期相同则取分数更高者。"""
        by_project: dict[str, SearchResult] = {}
        for result in results:
            project = result.metadata.get("project", "")
            report_date = result.metadata.get("report_date", "")
            if project not in by_project:
                by_project[project] = result
                continue
            existing = by_project[project]
            existing_date = existing.metadata.get("report_date", "")
            if report_date > existing_date or (
                report_date == existing_date and result.score > existing.score
            ):
                by_project[project] = result
        return sorted(by_project.values(), key=lambda item: item.score, reverse=True)

    @staticmethod
    def _dedupe_compare(results: List[SearchResult]) -> List[SearchResult]:
        """每月每项目保留分数最高的一条，按时间排序。"""
        by_month_project: dict[Tuple[str, str], SearchResult] = {}
        for result in results:
            report_date = result.metadata.get("report_date", "")
            year_month = report_date[:6] if len(report_date) >= 6 else "unknown"
            project = result.metadata.get("project", "")
            key = (year_month, project)
            if key not in by_month_project or result.score > by_month_project[key].score:
                by_month_project[key] = result
        return sorted(
            by_month_project.values(),
            key=lambda item: item.metadata.get("report_date", ""),
        )

    @staticmethod
    def _apply_search_mode(
        results: List[SearchResult],
        mode: SearchMode,
        k: int,
    ) -> List[SearchResult]:
        # 先取高分候选池，再按模式重排，避免弱相关旧数据占满 timeline/compare
        pool_size = max(k * 8, k)
        candidates = results[:pool_size]

        if mode == "timeline":
            ordered = sorted(
                candidates,
                key=lambda item: item.metadata.get("report_date", ""),
            )
            return ordered[:k]
        if mode == "compare":
            return VectorStore._dedupe_compare(candidates)[:k]
        return VectorStore._dedupe_latest(results)[:k]

    def query(
        self,
        query: str,
        embedding_model: BaseEmbeddings,
        k: int = DEFAULT_K,
        year: Optional[int] = None,
        month: Optional[int] = None,
        threshold: float = SIMILARITY_THRESHOLD,
        mode: SearchMode = "latest",
    ) -> List[SearchResult]:
        query_vector = embedding_model.get_embedding(query)
        candidate_indices = self._filter_indices(year=year, month=month)

        if year is not None and not candidate_indices:
            return []

        if not candidate_indices:
            candidate_indices = list(range(len(self.document)))

        candidate_vectors = [self.vectors[i] for i in candidate_indices]
        similarities = self._batch_cosine_similarity(query_vector, candidate_vectors)

        scores: List[Tuple[int, float]] = []
        for offset, idx in enumerate(candidate_indices):
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            hybrid_score = float(similarities[offset]) + self._keyword_boost(query, meta)
            scores.append((idx, hybrid_score))

        scores.sort(key=lambda item: item[1], reverse=True)

        results: List[SearchResult] = []
        for idx, score in scores:
            if score < threshold:
                continue
            results.append(
                SearchResult(
                    text=self.document[idx],
                    score=score,
                    metadata=self.metadata[idx] if idx < len(self.metadata) else {},
                )
            )

        return self._apply_search_mode(results, mode=mode, k=k)
