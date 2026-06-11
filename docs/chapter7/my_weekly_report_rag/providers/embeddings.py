#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Embedding 提供方：OpenAI 兼容 API + 本地缓存。"""

import logging
import time
from typing import List, Literal, Optional, Protocol

from config import (
    BGE_PASSAGE_PREFIX,
    BGE_QUERY_PREFIX,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_MODEL,
)
from exceptions import EmbeddingError
from providers.embedding_cache import EmbeddingCache
from providers.openai_client import get_openai_client

logger = logging.getLogger(__name__)
EmbeddingKind = Literal["query", "passage"]


class EmbeddingProvider(Protocol):
    """Embedding 提供方接口，供检索引擎依赖注入。"""

    def get_embedding(self, text: str, kind: EmbeddingKind = "passage") -> List[float]: ...

    def get_embeddings(self, texts: List[str], kind: EmbeddingKind = "passage") -> List[List[float]]: ...


class OpenAIEmbedding:
    """OpenAI 兼容 API + SQLite 本地缓存的 Embedding 实现。"""

    def __init__(self, use_cache: bool = True) -> None:
        self.client = get_openai_client()
        self.cache = EmbeddingCache() if use_cache else None

    def _format_text(self, text: str, kind: EmbeddingKind) -> str:
        """按 BGE 模型要求为 query/passage 添加前缀。"""
        if kind == "query" and BGE_QUERY_PREFIX:
            return f"{BGE_QUERY_PREFIX}{text}"
        if kind == "passage" and BGE_PASSAGE_PREFIX:
            return f"{BGE_PASSAGE_PREFIX}{text}"
        return text

    def get_embedding(self, text: str, kind: EmbeddingKind = "passage") -> List[float]:
        return self.get_embeddings([text], kind=kind)[0]

    def get_embeddings(self, texts: List[str], kind: EmbeddingKind = "passage") -> List[List[float]]:
        """批量获取 embedding，优先读缓存，缺失部分调 API。"""
        if not texts:
            return []

        formatted = [self._format_text(text, kind) for text in texts]
        results: List[Optional[List[float]]] = [None] * len(formatted)
        missing_texts: List[str] = []
        missing_indexes: List[int] = []

        if self.cache is not None:
            cached = self.cache.get_many(formatted)
            for index, embedding in enumerate(cached):
                if embedding is not None:
                    results[index] = embedding
                else:
                    missing_indexes.append(index)
                    missing_texts.append(formatted[index])
        else:
            missing_indexes = list(range(len(formatted)))
            missing_texts = formatted

        if missing_texts:
            fetched = self._fetch_embeddings(missing_texts)
            for index, embedding in zip(missing_indexes, fetched):
                results[index] = embedding
            if self.cache is not None:
                self.cache.set_many(missing_texts, fetched)

        if any(embedding is None for embedding in results):
            raise EmbeddingError("embedding 结果不完整")
        return results  # type: ignore[return-value]

    def _fetch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """按批次调用 API 获取 embedding。"""
        results: List[List[float]] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + EMBEDDING_BATCH_SIZE]
            results.extend(self._embed_batch_with_retry(batch))
        return results

    def _embed_batch_with_retry(self, batch: List[str]) -> List[List[float]]:
        """单批次 embedding 请求，失败时指数退避重试。"""
        last_error: Optional[Exception] = None
        for attempt in range(EMBEDDING_MAX_RETRIES):
            try:
                response = self.client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
                return [item.embedding for item in response.data]
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "embedding 请求失败，重试 %s/%s: %s",
                    attempt + 1,
                    EMBEDDING_MAX_RETRIES,
                    exc,
                )
                if attempt < EMBEDDING_MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        raise EmbeddingError(f"embedding 请求失败（已重试 {EMBEDDING_MAX_RETRIES} 次）: {last_error}") from last_error
