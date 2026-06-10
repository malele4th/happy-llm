#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Optional

from config import DEFAULT_K, INDEX_PATH
from generation.llm import OpenAIChat
from indexing.store import IndexStore, load_index
from models import DEFAULT_SEARCH_MODE, SearchMode, SearchResult
from providers.embeddings import OpenAIEmbedding
from retrieval.engine import SearchEngine


class RAGSession:
    """复用索引、检索引擎与模型客户端。"""

    def __init__(self, index_path: str = INDEX_PATH) -> None:
        self.index_path = index_path
        self.store: IndexStore = load_index(index_path)
        self.search_engine = SearchEngine(self.store)
        self.embedding = OpenAIEmbedding()
        self.chat = OpenAIChat()


def search(
    question: str,
    k: int = DEFAULT_K,
    year: Optional[int] = None,
    month: Optional[int] = None,
    mode: SearchMode = DEFAULT_SEARCH_MODE,
    session: Optional[RAGSession] = None,
    index_path: str = INDEX_PATH,
) -> List[SearchResult]:
    if session is None:
        session = RAGSession(index_path)
    return session.search_engine.query(
        question,
        embedding_model=session.embedding,
        k=k,
        year=year,
        month=month,
        mode=mode,
    )
