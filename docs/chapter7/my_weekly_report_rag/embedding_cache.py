#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib
import json
import logging
import sqlite3
from typing import List, Optional

from config import EMBEDDING_CACHE_PATH, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingCache:
    def __init__(self, db_path: str = EMBEDDING_CACHE_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    text_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    PRIMARY KEY (text_hash, model)
                )
                """
            )

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str, model: str = EMBEDDING_MODEL) -> Optional[List[float]]:
        text_hash = self._hash_text(text)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT embedding FROM embedding_cache WHERE text_hash = ? AND model = ?",
                (text_hash, model),
            ).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def set(self, text: str, embedding: List[float], model: str = EMBEDDING_MODEL) -> None:
        text_hash = self._hash_text(text)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embedding_cache (text_hash, model, embedding)
                VALUES (?, ?, ?)
                """,
                (text_hash, model, json.dumps(embedding)),
            )

    def get_many(self, texts: List[str], model: str = EMBEDDING_MODEL) -> List[Optional[List[float]]]:
        return [self.get(text, model=model) for text in texts]

    def set_many(self, texts: List[str], embeddings: List[List[float]], model: str = EMBEDDING_MODEL) -> None:
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding, model=model)
        logger.info("embedding 缓存写入 %s 条", len(texts))
