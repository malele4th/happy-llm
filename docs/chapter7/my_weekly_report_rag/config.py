#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

_BASE_DIR = Path(__file__).resolve().parent
_RULES_PATH = _BASE_DIR / "parser_rules.json"

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
HEADING_MAX_LEN = 40
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
PARSER_RULES_PATH = str(_RULES_PATH)


def _load_parser_rules() -> dict:
    with open(_RULES_PATH, encoding="utf-8") as handle:
        return json.load(handle)


_PARSER_RULES = _load_parser_rules()
PARSER_PROJECT_HINTS = _PARSER_RULES.get("parser_project_hints", [])
PARSER_SECTION_PREFIXES = _PARSER_RULES.get("section_prefixes", {})
