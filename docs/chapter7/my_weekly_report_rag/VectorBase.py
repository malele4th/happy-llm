#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from typing import List

import numpy as np
from tqdm import tqdm

from config import DEFAULT_K, STORAGE_PATH
from Embeddings import BaseEmbeddings


class VectorStore:
    def __init__(self, document: List[str] = None) -> None:
        self.document = document or []
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
        if self.vectors:
            with open(f"{path}/vectors.json", "w", encoding="utf-8") as f:
                json.dump(self.vectors, f)

    def load_vector(self, path: str = STORAGE_PATH) -> None:
        with open(f"{path}/vectors.json", "r", encoding="utf-8") as f:
            self.vectors = json.load(f)
        with open(f"{path}/document.json", "r", encoding="utf-8") as f:
            self.document = json.load(f)

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        return BaseEmbeddings.cosine_similarity(vector1, vector2)

    def query(self, query: str, embedding_model: BaseEmbeddings, k: int = DEFAULT_K) -> List[str]:
        query_vector = embedding_model.get_embedding(query)
        result = np.array([self.get_similarity(query_vector, vector) for vector in self.vectors])
        return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()
