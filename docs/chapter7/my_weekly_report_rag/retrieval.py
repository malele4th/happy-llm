#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple

from bootstrap import check_env, load_index
from config import DEFAULT_K, STORAGE_PATH
from embeddings import OpenAIEmbedding
from index_store import IndexStore
from llm import OpenAIChat
from models import DEFAULT_SEARCH_MODE, SearchMode, SearchResult
from retriever import Retriever
from utils import parse_date_filter


class RAGSession:
    """复用索引、检索器与模型客户端。"""

    def __init__(self, storage_path: str = STORAGE_PATH) -> None:
        check_env()
        self.storage_path = storage_path
        self.store = load_index(storage_path)
        self.retriever = Retriever(self.store)
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
    k: int = DEFAULT_K,
    year: Optional[int] = None,
    month: Optional[int] = None,
    mode: SearchMode = DEFAULT_SEARCH_MODE,
    session: Optional[RAGSession] = None,
    storage_path: str = STORAGE_PATH,
) -> List[SearchResult]:
    if session is None:
        session = RAGSession(storage_path)

    return session.retriever.query(
        question,
        embedding_model=session.embedding,
        k=k,
        year=year,
        month=month,
        mode=mode,
    )
