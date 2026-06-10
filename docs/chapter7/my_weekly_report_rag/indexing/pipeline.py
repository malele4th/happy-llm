#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import shutil
from typing import List

from config import REPORT_DATA_PATH, STORAGE_PATH
from exceptions import NoDataError
from indexing.manifest import compute_index_version, file_hash, load_manifest, save_manifest
from indexing.store import IndexStore, load_index
from models import DocumentChunk
from parsing.reader import ReadFiles
from providers.embeddings import OpenAIEmbedding

logger = logging.getLogger(__name__)


def _embed_chunks(embedding: OpenAIEmbedding, chunks: List[DocumentChunk]) -> List[List[float]]:
    if not chunks:
        return []
    return embedding.get_embeddings([chunk.text for chunk in chunks], kind="passage")


def _incremental_build(reader: ReadFiles, storage_path: str) -> IndexStore:
    store = load_index(storage_path)
    manifest = load_manifest(storage_path)
    old_files = manifest.get("files", {})
    version_changed = manifest.get("index_version") != compute_index_version()

    current_files = {
        os.path.relpath(file_path, reader.data_path): file_hash(file_path)
        for file_path in reader.file_list
    }

    sources_to_remove = {rel for rel in old_files if rel not in current_files}
    if version_changed:
        sources_to_update = set(current_files.keys())
    else:
        sources_to_update = {
            rel
            for rel, digest in current_files.items()
            if rel not in old_files or old_files[rel].get("hash") != digest
        }
    sources_to_remove |= sources_to_update

    store.records = [
        record for record in store.records
        if record.metadata.source not in sources_to_remove
    ]
    store._matrix = None

    new_chunks: List[DocumentChunk] = []
    for file_path in reader.file_list:
        rel_path = os.path.relpath(file_path, reader.data_path)
        if rel_path in sources_to_update:
            new_chunks.extend(reader.get_chunks_for_file(file_path))

    if new_chunks:
        embedding = OpenAIEmbedding()
        store.append_records(new_chunks, _embed_chunks(embedding, new_chunks))

    store.persist(path=storage_path)
    save_manifest(storage_path, reader)

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
    save_manifest(storage_path, reader)
    logger.info("向量库已保存到 %s", storage_path)
    return store
