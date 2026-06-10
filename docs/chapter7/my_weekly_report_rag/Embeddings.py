#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from typing import List, Optional

from clients import get_openai_client
from config import EMBEDDING_BATCH_SIZE, EMBEDDING_MAX_RETRIES, EMBEDDING_MODEL


class BaseEmbeddings:
    def get_embedding(self, text: str, model: str = EMBEDDING_MODEL) -> List[float]:
        return self.get_embeddings([text], model=model)[0]

    def get_embeddings(self, texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
        raise NotImplementedError


class OpenAIEmbedding(BaseEmbeddings):
    def __init__(self) -> None:
        self.client = get_openai_client()

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
