#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

_BASE_DIR = Path(__file__).resolve().parent
PARSER_RULES_PATH = str(_BASE_DIR / "parser_rules.json")

REPORT_DATA_PATH = os.getenv(
    "REPORT_DATA_PATH",
    "/Users/bigo/Desktop/bigo/bigo工作周报",
)
STORAGE_PATH = os.getenv("REPORT_STORAGE_PATH", "./storage")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen2.5-32B-Instruct")

MAX_TOKEN_LEN = 600
COVER_CONTENT = 150
DEFAULT_K = 5
SEARCH_CANDIDATE_POOL_FACTOR = 8

MODE_THRESHOLDS = {
    "latest": float(os.getenv("THRESHOLD_LATEST", "0.35")),
    "timeline": float(os.getenv("THRESHOLD_TIMELINE", "0.28")),
    "compare": float(os.getenv("THRESHOLD_COMPARE", "0.30")),
}

VECTOR_SCORE_WEIGHT = float(os.getenv("VECTOR_SCORE_WEIGHT", "0.7"))
BM25_SCORE_WEIGHT = float(os.getenv("BM25_SCORE_WEIGHT", "0.3"))

EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
EMBEDDING_CACHE_PATH = os.getenv(
    "EMBEDDING_CACHE_PATH",
    str(_BASE_DIR / ".embedding_cache.sqlite"),
)

BGE_QUERY_PREFIX = os.getenv(
    "BGE_QUERY_PREFIX",
    "为这个句子生成表示以用于检索相关文章：",
)
BGE_PASSAGE_PREFIX = os.getenv("BGE_PASSAGE_PREFIX", "")

MANIFEST_FILE = "manifest.json"


def setup_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
