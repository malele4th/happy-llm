#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from config import DEFAULT_K, SIMILARITY_THRESHOLD, STORAGE_PATH
from Embeddings import BaseEmbeddings


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

    def get_vector(self, embedding_model: BaseEmbeddings) -> List[List[float]]:
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(embedding_model.get_embedding(doc))
        return self.vectors

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

    def query(
        self,
        query: str,
        embedding_model: BaseEmbeddings,
        k: int = DEFAULT_K,
        year: Optional[int] = None,
        month: Optional[int] = None,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> List[SearchResult]:
        query_vector = embedding_model.get_embedding(query)
        candidate_indices = self._filter_indices(year=year, month=month)

        if year is not None and not candidate_indices:
            return []

        if not candidate_indices:
            candidate_indices = list(range(len(self.document)))

        scores = []
        for i in candidate_indices:
            score = BaseEmbeddings.cosine_similarity(query_vector, self.vectors[i])
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:k]:
            if score < threshold:
                continue
            results.append(
                SearchResult(
                    text=self.document[idx],
                    score=score,
                    metadata=self.metadata[idx] if idx < len(self.metadata) else {},
                )
            )
        return results
