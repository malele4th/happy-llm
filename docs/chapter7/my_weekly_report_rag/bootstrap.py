#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil

from config import STORAGE_PATH
from exceptions import EnvConfigError, StorageNotFoundError
from index_store import IndexStore


def check_env() -> None:
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        raise EnvConfigError("请在 .env 中配置 OPENAI_API_KEY 和 OPENAI_BASE_URL")


def cleanup_storage_tmp(base_dir: str | None = None) -> None:
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(STORAGE_PATH)) or "."
    tmp_path = os.path.join(base_dir, ".storage_tmp")
    if os.path.isdir(tmp_path):
        shutil.rmtree(tmp_path, ignore_errors=True)


def load_index(storage_path: str = STORAGE_PATH) -> IndexStore:
    if not IndexStore.exists(storage_path):
        raise StorageNotFoundError(
            "storage 不存在或不完整，请先运行: python weekly_report_rag.py --build"
        )
    store = IndexStore()
    store.load_from_disk(storage_path)
    return store
