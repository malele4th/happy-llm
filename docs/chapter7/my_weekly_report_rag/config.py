#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from exceptions import EnvConfigError

_ = load_dotenv(find_dotenv())

_BASE_DIR = Path(__file__).resolve().parent
PARSER_RULES_PATH = str(_BASE_DIR / "parsing" / "parser_rules.json")
LOG_DIR = _BASE_DIR / "log"
INDEX_PATH = os.getenv("INDEX_PATH", "./data")

REPORT_DATA_PATH = os.getenv("REPORT_DATA_PATH", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen2.5-32B-Instruct")

MAX_TOKEN_LEN = 600
COVER_CONTENT = 150
DEFAULT_K = 5
DEFAULT_AUTO_DATE = os.getenv("DEFAULT_AUTO_DATE", "true").lower() in ("1", "true", "yes")
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
INDEX_TMP_DIR = ".index_tmp"
LEGACY_TMP_DIR = ".storage_tmp"

WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", "1203"))
WEB_ACCESS_TOKEN = os.getenv("WEB_ACCESS_TOKEN", "")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    log_file = LOG_DIR / f"app_{datetime.now():%Y%m%d}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def check_env() -> None:
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        raise EnvConfigError("请在 .env 中配置 OPENAI_API_KEY 和 OPENAI_BASE_URL")


def cleanup_index_tmp(base_dir: str | None = None) -> None:
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(INDEX_PATH)) or "."
    for tmp_name in (INDEX_TMP_DIR, LEGACY_TMP_DIR):
        tmp_path = os.path.join(base_dir, tmp_name)
        if os.path.isdir(tmp_path):
            shutil.rmtree(tmp_path, ignore_errors=True)
