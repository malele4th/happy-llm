#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple

from config import DEFAULT_K, STORAGE_PATH
from indexing.store import IndexStore, load_index
from models import DEFAULT_SEARCH_MODE, SearchMode, SearchResult
from providers.embeddings import OpenAIEmbedding
from generation.llm import OpenAIChat
from retrieval.retriever import Retriever
from utils import parse_date_filter


class RAGSession:
    """复用索引、检索器与模型客户端。"""

    def __init__(self, storage_path: str = STORAGE_PATH) -> None:
        self.storage_path = storage_path
        self.store: IndexStore = load_index(storage_path)
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
    active_session = session or RAGSession(storage_path)
    return active_session.retriever.query(
        question,
        embedding_model=active_session.embedding,
        k=k,
        year=year,
        month=month,
        mode=mode,
    )
