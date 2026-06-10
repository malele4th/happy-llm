#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib
import json
import logging
import os
import shutil
from typing import List

from bootstrap import check_env, load_index
from config import (
    COVER_CONTENT,
    HEADING_MAX_LEN,
    MANIFEST_FILE,
    MAX_TOKEN_LEN,
    PARSER_RULES_PATH,
    REPORT_DATA_PATH,
    STORAGE_PATH,
)
from embeddings import OpenAIEmbedding
from exceptions import NoDataError
from index_store import IndexStore
from parser import ReadFiles, load_parser_rules
from models import DocumentChunk

logger = logging.getLogger(__name__)


def compute_index_version() -> str:
    with open(PARSER_RULES_PATH, encoding="utf-8") as handle:
        rules_hash = hashlib.md5(handle.read().encode()).hexdigest()[:8]
    payload = "|".join([
        str(MAX_TOKEN_LEN),
        str(COVER_CONTENT),
        str(HEADING_MAX_LEN),
        rules_hash,
    ])
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _file_hash(file_path: str) -> str:
    digest = hashlib.md5()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(storage_path: str) -> str:
    return os.path.join(storage_path, MANIFEST_FILE)


def _load_manifest(storage_path: str) -> dict:
    path = _manifest_path(storage_path)
    if not os.path.exists(path):
        return {"files": {}, "index_version": ""}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_manifest(storage_path: str, reader: ReadFiles) -> None:
    manifest = {
        "index_version": compute_index_version(),
        "parser_rules": load_parser_rules(),
        "files": {
            os.path.relpath(file_path, reader.data_path): {"hash": _file_hash(file_path)}
            for file_path in reader.file_list
        },
    }
    with open(_manifest_path(storage_path), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def _embed_chunks(embedding: OpenAIEmbedding, chunks: List[DocumentChunk]) -> List[List[float]]:
    if not chunks:
        return []
    return embedding.get_embeddings([chunk.text for chunk in chunks], kind="passage")


def _incremental_build(reader: ReadFiles, storage_path: str) -> IndexStore:
    store = load_index(storage_path)
    manifest = _load_manifest(storage_path)
    old_files = manifest.get("files", {})
    current_version = compute_index_version()
    version_changed = manifest.get("index_version") != current_version

    current_files = {
        os.path.relpath(file_path, reader.data_path): _file_hash(file_path)
        for file_path in reader.file_list
    }

    sources_to_remove = {rel for rel in old_files if rel not in current_files}
    if version_changed:
        sources_to_update = set(current_files.keys())
    else:
        sources_to_update = {
            rel
            for rel, file_hash in current_files.items()
            if rel not in old_files or old_files[rel].get("hash") != file_hash
        }
    sources_to_remove |= sources_to_update

    kept_records = [
        record for record in store.records
        if record.metadata.source not in sources_to_remove
    ]
    store.records = kept_records
    store._matrix = None

    new_chunks: List[DocumentChunk] = []
    for file_path in reader.file_list:
        rel_path = os.path.relpath(file_path, reader.data_path)
        if rel_path in sources_to_update:
            new_chunks.extend(reader.get_chunks_for_file(file_path))

    if new_chunks:
        embedding = OpenAIEmbedding()
        new_vectors = _embed_chunks(embedding, new_chunks)
        store.append_records(new_chunks, new_vectors)

    store.persist(path=storage_path)
    _save_manifest(storage_path, reader)

    reason = "切块/解析规则变更，" if version_changed else ""
    logger.info(
        "%s移除/更新 %s 个文件, 新增 %s 个 chunk, 当前共 %s 个 chunk",
        reason,
        len(sources_to_remove),
        len(new_chunks),
        len(store.records),
    )
    return store


def build_index(
    data_path: str = REPORT_DATA_PATH,
    storage_path: str = STORAGE_PATH,
    force: bool = False,
) -> IndexStore:
    check_env()

    reader = ReadFiles(data_path)
    if not reader.file_list:
        raise NoDataError(f"在 {data_path} 下未找到 docx 文件")

    if os.path.exists(storage_path) and not force:
        return _incremental_build(reader, storage_path)

    if force and os.path.exists(storage_path):
        backup_path = f"{storage_path}.backup"
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.copytree(storage_path, backup_path)
        logger.info("已备份旧索引到 %s", backup_path)
        shutil.rmtree(storage_path)

    logger.info("扫描到 %s 个 docx 文件", len(reader.file_list))
    chunks = reader.get_chunks()
    logger.info("切分为 %s 个项目 chunk", len(chunks))

    store = IndexStore.from_chunks(chunks)
    embedding = OpenAIEmbedding()
    store.set_vectors(_embed_chunks(embedding, chunks))
    store.persist(path=storage_path)
    _save_manifest(storage_path, reader)
    logger.info("向量库已保存到 %s", storage_path)
    return store
