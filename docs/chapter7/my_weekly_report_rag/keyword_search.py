#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import re
from collections import Counter
from typing import Iterable, List

from models import ChunkMetadata
from utils import extract_query_tokens


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text.lower())
    return [token for token in tokens if len(token) >= 2]


def bm25_score(query: str, document: str, corpus_avg_len: float, k1: float = 1.5, b: float = 0.75) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    doc_tokens = _tokenize(document)
    if not doc_tokens:
        return 0.0

    doc_len = len(doc_tokens)
    term_freq = Counter(doc_tokens)
    score = 0.0
    for token in set(query_tokens):
        tf = term_freq.get(token, 0)
        if tf == 0:
            continue
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_len / max(corpus_avg_len, 1.0))
        score += math.log(1 + numerator / denominator)
    return score


def build_search_text(metadata: ChunkMetadata, body_text: str) -> str:
    return " ".join([
        metadata.project,
        metadata.author,
        metadata.quarter,
        metadata.section_type,
        body_text,
    ])


def score_candidates(
    query: str,
    texts: Iterable[str],
    metadata_list: Iterable[ChunkMetadata],
) -> List[float]:
    search_docs = [
        build_search_text(meta, text.split("\n", 1)[-1] if text else "")
        for text, meta in zip(texts, metadata_list)
    ]
    if not search_docs:
        return []

    avg_len = sum(len(_tokenize(doc)) for doc in search_docs) / len(search_docs)
    raw_scores = [bm25_score(query, doc, avg_len) for doc in search_docs]
    max_score = max(raw_scores) if raw_scores else 0.0
    if max_score <= 0:
        return [0.0 for _ in raw_scores]
    return [score / max_score for score in raw_scores]
