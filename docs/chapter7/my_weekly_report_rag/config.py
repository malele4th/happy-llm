#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

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
SIMILARITY_THRESHOLD = 0.35
HEADING_MAX_LEN = 40
SEARCH_CANDIDATE_POOL_FACTOR = 8

EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
KEYWORD_BOOST_MAX = 0.3
KEYWORD_PROJECT_BOOST = 0.15
KEYWORD_TOKEN_BOOST = 0.05

_DEFAULT_KEYWORDS = "catchii,rank,家族房,bge,imo,helloyo,likee,bigo"
PROJECT_KEYWORDS = [
    kw.strip()
    for kw in os.getenv("REPORT_PROJECT_KEYWORDS", _DEFAULT_KEYWORDS).split(",")
    if kw.strip()
]

MANIFEST_FILE = "manifest.json"
