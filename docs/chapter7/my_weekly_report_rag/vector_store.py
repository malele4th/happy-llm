#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import shutil
from typing import List, Optional, Tuple

from config import (
    DEFAULT_K,
    KEYWORD_BOOST_MAX,
    KEYWORD_PROJECT_BOOST,
    KEYWORD_TOKEN_BOOST,
    PROJECT_KEYWORDS,
    SEARCH_CANDIDATE_POOL_FACTOR,
    SIMILARITY_THRESHOLD,
    STORAGE_PATH,
)
from embeddings import BaseEmbeddings
from exceptions import StorageCorruptError
from similarity import batch_cosine_similarity
from models import ChunkMetadata, DEFAULT_SEARCH_MODE, SearchMode, SearchResult
from utils import extract_query_tokens


class VectorStore:
    def __init__(
        self,
        document: List[str] = None,
        metadata: List[dict] = None,
    ) -> None:
        self.document = document or []
        self.metadata = metadata or []
        self.vectors: List[List[float]] = []

    def compute_embeddings(
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
        parent_dir = os.path.dirname(os.path.abspath(path)) or "."
        tmp_path = os.path.join(parent_dir, ".storage_tmp")
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        os.makedirs(tmp_path, exist_ok=True)

        with open(os.path.join(tmp_path, "document.json"), "w", encoding="utf-8") as handle:
            json.dump(self.document, handle, ensure_ascii=False)
        with open(os.path.join(tmp_path, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, ensure_ascii=False)
        if self.vectors:
            with open(os.path.join(tmp_path, "vectors.json"), "w", encoding="utf-8") as handle:
                json.dump(self.vectors, handle)

        self._validate_lengths()
        if os.path.exists(path):
            shutil.rmtree(path)
        os.rename(tmp_path, path)

    def load_from_disk(self, path: str = STORAGE_PATH) -> None:
        with open(os.path.join(path, "vectors.json"), "r", encoding="utf-8") as handle:
            self.vectors = json.load(handle)
        with open(os.path.join(path, "document.json"), "r", encoding="utf-8") as handle:
            self.document = json.load(handle)
        metadata_path = os.path.join(path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as handle:
                self.metadata = json.load(handle)
        else:
            self.metadata = [{} for _ in self.document]
        self._validate_lengths()

    def _validate_lengths(self) -> None:
        doc_count = len(self.document)
        meta_count = len(self.metadata)
        vector_count = len(self.vectors)
        if doc_count != meta_count or (self.vectors and doc_count != vector_count):
            raise StorageCorruptError(
                f"索引数据不一致: document={doc_count}, metadata={meta_count}, vectors={vector_count}"
            )

    def _meta_at(self, index: int) -> ChunkMetadata:
        if index < len(self.metadata):
            return ChunkMetadata.from_dict(self.metadata[index])
        return ChunkMetadata(source="")

    def _filter_indices(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None,
    ) -> List[int]:
        if year is None:
            return list(range(len(self.document)))

        indices = []
        for index, meta in enumerate(self.metadata):
            chunk_meta = ChunkMetadata.from_dict(meta)
            report_date = chunk_meta.report_date
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
    def _keyword_boost(query: str, meta: ChunkMetadata) -> float:
        boost = 0.0
        query_lower = query.lower()
        project_lower = meta.project.lower()
        source_path_lower = meta.source.lower()

        for keyword in PROJECT_KEYWORDS:
            keyword_lower = keyword.lower()
            if keyword_lower in query_lower and keyword_lower in project_lower:
                boost += KEYWORD_PROJECT_BOOST

        for token in extract_query_tokens(query):
            if token in project_lower or token in source_path_lower:
                boost += KEYWORD_TOKEN_BOOST

        return min(boost, KEYWORD_BOOST_MAX)

    @staticmethod
    def _dedupe_latest(results: List[SearchResult]) -> List[SearchResult]:
        """同项目保留最新周报；同周同项目多段 chunk 保留分数更高者。"""
        by_project: dict[str, SearchResult] = {}
        for result in results:
            project = result.metadata.project
            report_date = result.metadata.report_date
            if project not in by_project:
                by_project[project] = result
                continue
            existing = by_project[project]
            existing_date = existing.metadata.report_date
            if report_date > existing_date or (
                report_date == existing_date and result.score > existing.score
            ):
                by_project[project] = result
        return sorted(by_project.values(), key=lambda item: item.score, reverse=True)

    @staticmethod
    def _dedupe_compare(results: List[SearchResult]) -> List[SearchResult]:
        """每月每项目保留分数最高的一条；同月同项目多段 chunk 不互相覆盖。"""
        by_key: dict[Tuple[str, str, int], SearchResult] = {}
        for result in results:
            meta = result.metadata
            key = (meta.year_month(), meta.project, meta.chunk_index)
            if key not in by_key or result.score > by_key[key].score:
                by_key[key] = result
        return sorted(
            by_key.values(),
            key=lambda item: (item.metadata.report_date, item.metadata.chunk_index),
        )

    @staticmethod
    def _apply_search_mode(
        results: List[SearchResult],
        mode: SearchMode,
        k: int,
    ) -> List[SearchResult]:
        pool_size = max(k * SEARCH_CANDIDATE_POOL_FACTOR, k)
        candidates = results[:pool_size]

        if mode == "timeline":
            ordered = sorted(
                candidates,
                key=lambda item: (item.metadata.report_date, item.metadata.chunk_index),
            )
            return ordered[:k]
        if mode == "compare":
            return VectorStore._dedupe_compare(candidates)[:k]
        return VectorStore._dedupe_latest(candidates)[:k]

    def query(
        self,
        query: str,
        embedding_model: BaseEmbeddings,
        k: int = DEFAULT_K,
        year: Optional[int] = None,
        month: Optional[int] = None,
        threshold: float = SIMILARITY_THRESHOLD,
        mode: SearchMode = DEFAULT_SEARCH_MODE,
    ) -> List[SearchResult]:
        if not self.document:
            return []

        query_vector = embedding_model.get_embedding(query)
        candidate_indices = self._filter_indices(year=year, month=month)
        if year is not None and not candidate_indices:
            return []

        candidate_vectors = [self.vectors[i] for i in candidate_indices]
        similarities = batch_cosine_similarity(query_vector, candidate_vectors)

        scores: List[Tuple[int, float]] = []
        for offset, index in enumerate(candidate_indices):
            chunk_meta = self._meta_at(index)
            hybrid_score = float(similarities[offset]) + self._keyword_boost(query, chunk_meta)
            scores.append((index, hybrid_score))

        scores.sort(key=lambda item: item[1], reverse=True)

        results: List[SearchResult] = []
        for index, score in scores:
            if score < threshold:
                continue
            results.append(
                SearchResult(
                    text=self.document[index],
                    score=score,
                    metadata=self._meta_at(index),
                )
            )

        return self._apply_search_mode(results, mode=mode, k=k)
