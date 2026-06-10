#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from typing import List, Optional

import numpy as np
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from config import EMBEDDING_BATCH_SIZE, EMBEDDING_MAX_RETRIES, EMBEDDING_MODEL

_ = load_dotenv(find_dotenv())


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


class BaseEmbeddings:
    def get_embedding(self, text: str, model: str = EMBEDDING_MODEL) -> List[float]:
        return self.get_embeddings([text], model=model)[0]

    def get_embeddings(self, texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
        raise NotImplementedError


class OpenAIEmbedding(BaseEmbeddings):
    def __init__(self) -> None:
        self.client = OpenAI()
        self.client.api_key = os.getenv("OPENAI_API_KEY")
        self.client.base_url = os.getenv("OPENAI_BASE_URL")

    def get_embeddings(self, texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
        if not texts:
            return []

        results: List[List[float]] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + EMBEDDING_BATCH_SIZE]
            results.extend(self._embed_batch_with_retry(batch, model))
        return results

    def _embed_batch_with_retry(self, batch: List[str], model: str) -> List[List[float]]:
        last_error: Optional[Exception] = None
        for attempt in range(EMBEDDING_MAX_RETRIES):
            try:
                response = self.client.embeddings.create(input=batch, model=model)
                return [item.embedding for item in response.data]
            except Exception as exc:
                last_error = exc
                if attempt < EMBEDDING_MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        raise last_error  # type: ignore[misc]
