#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import List, Optional, Tuple

from config import DEFAULT_K, STORAGE_PATH
from embeddings import OpenAIEmbedding
from exceptions import EnvConfigError, StorageNotFoundError
from llm import OpenAIChat
from models import DEFAULT_SEARCH_MODE, SearchMode, SearchResult
from utils import parse_date_filter
from vector_store import VectorStore


def check_env() -> None:
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        raise EnvConfigError("请在 .env 中配置 OPENAI_API_KEY 和 OPENAI_BASE_URL")


def load_index(storage_path: str = STORAGE_PATH) -> VectorStore:
    vectors_file = os.path.join(storage_path, "vectors.json")
    document_file = os.path.join(storage_path, "document.json")
    if not os.path.exists(vectors_file) or not os.path.exists(document_file):
        raise StorageNotFoundError(
            "storage 不存在或不完整，请先运行: python weekly_report_rag.py --build"
        )
    vector = VectorStore()
    vector.load_from_disk(storage_path)
    return vector


class RAGSession:
    """复用向量库与模型客户端，避免交互模式下重复加载。"""

    def __init__(self, storage_path: str = STORAGE_PATH) -> None:
        check_env()
        self.storage_path = storage_path
        self.vector = load_index(storage_path)
        self.embedding = OpenAIEmbedding()
        self.chat = OpenAIChat()


def resolve_date_filter(
    question: str,
    year: Optional[int],
    month: Optional[int],
    auto_date: bool,
) -> Tuple[Optional[int], Optional[int]]:
    if year is not None:
        return year, month
    if auto_date:
        return parse_date_filter(question)
    return None, None


def search(
    question: str,
    storage_path: str = STORAGE_PATH,
    k: int = DEFAULT_K,
    year: Optional[int] = None,
    month: Optional[int] = None,
    auto_date: bool = False,
    mode: SearchMode = DEFAULT_SEARCH_MODE,
    session: Optional[RAGSession] = None,
) -> List[SearchResult]:
    filter_year, filter_month = resolve_date_filter(question, year, month, auto_date)

    if session is not None:
        return session.vector.query(
            question,
            embedding_model=session.embedding,
            k=k,
            year=filter_year,
            month=filter_month,
            mode=mode,
        )

    vector = load_index(storage_path)
    embedding = OpenAIEmbedding()
    return vector.query(
        question,
        embedding_model=embedding,
        k=k,
        year=filter_year,
        month=filter_month,
        mode=mode,
    )
