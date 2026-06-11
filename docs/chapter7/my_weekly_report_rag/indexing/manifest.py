#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""索引 manifest：文件哈希与解析规则版本，用于增量构建。"""

import hashlib
import json
import os

from config import COVER_CONTENT, MANIFEST_FILE, MAX_TOKEN_LEN, PARSER_RULES_PATH
from parsing.reader import DocxReportReader
from parsing.rules import heading_max_len, load_parser_rules


def compute_index_version() -> str:
    """根据切块参数与解析规则生成版本号，规则变更时触发全量重建。"""
    with open(PARSER_RULES_PATH, encoding="utf-8") as handle:
        rules_hash = hashlib.md5(handle.read().encode()).hexdigest()[:8]
    payload = "|".join([
        str(MAX_TOKEN_LEN),
        str(COVER_CONTENT),
        str(heading_max_len()),
        rules_hash,
    ])
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def file_hash(file_path: str) -> str:
    """计算文件 MD5，用于增量构建时检测变更。"""
    digest = hashlib.md5()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(index_path: str) -> str:
    """返回 manifest.json 的完整路径。"""
    return os.path.join(index_path, MANIFEST_FILE)


def build_manifest(reader: DocxReportReader) -> dict:
    """构建 manifest 字典（与索引一并原子写入）。"""
    return {
        "index_version": compute_index_version(),
        "parser_rules": load_parser_rules(),
        "files": {
            os.path.relpath(file_path, reader.data_path): {"hash": file_hash(file_path)}
            for file_path in reader.file_list
        },
    }


def load_manifest(index_path: str) -> dict:
    """读取 manifest，不存在时返回空结构。"""
    path = manifest_path(index_path)
    if not os.path.exists(path):
        return {"files": {}, "index_version": ""}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(index_path: str, reader: DocxReportReader) -> None:
    """独立写入 manifest（仅用于兼容；构建流程应通过 IndexStore.persist 一并写入）。"""
    manifest = build_manifest(reader)
    with open(manifest_path(index_path), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
