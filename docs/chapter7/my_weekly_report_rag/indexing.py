#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib
import json
import os
import shutil
from typing import List

from config import (
    COVER_CONTENT,
    HEADING_MAX_LEN,
    MANIFEST_FILE,
    MAX_TOKEN_LEN,
    PROJECT_KEYWORDS,
    REPORT_DATA_PATH,
    STORAGE_PATH,
)
from embeddings import OpenAIEmbedding
from exceptions import NoDataError
from parser import ReadFiles
from retrieval import check_env, load_index
from vector_store import VectorStore


def compute_index_version() -> str:
    payload = "|".join([
        str(MAX_TOKEN_LEN),
        str(COVER_CONTENT),
        str(HEADING_MAX_LEN),
        ",".join(sorted(PROJECT_KEYWORDS)),
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
        "files": {
            os.path.relpath(file_path, reader.data_path): {"hash": _file_hash(file_path)}
            for file_path in reader.file_list
        },
    }
    with open(_manifest_path(storage_path), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def _chunks_to_vector(chunks: List) -> VectorStore:
    return VectorStore(
        document=[chunk.text for chunk in chunks],
        metadata=[chunk.metadata.to_dict() for chunk in chunks],
    )


def _incremental_build(reader: ReadFiles, storage_path: str) -> VectorStore:
    vector = load_index(storage_path)
    manifest = _load_manifest(storage_path)
    old_files = manifest.get("files", {})
    current_version = compute_index_version()
    version_changed = manifest.get("index_version") != current_version

    current_files = {
        os.path.relpath(file_path, reader.data_path): _file_hash(file_path)
        for file_path in reader.file_list
    }

    sources_to_remove = {rel for rel in old_files if rel not in current_files}
    sources_to_update = set()
    if version_changed:
        sources_to_update = set(current_files.keys())
    else:
        sources_to_update = {
            rel
            for rel, file_hash in current_files.items()
            if rel not in old_files or old_files[rel].get("hash") != file_hash
        }
    sources_to_remove |= sources_to_update

    kept_docs: List[str] = []
    kept_metadata: List[dict] = []
    kept_vectors: List[List[float]] = []

    for index, doc in enumerate(vector.document):
        meta = vector.metadata[index]
        source = meta.get("source", "")
        if source in sources_to_remove:
            continue
        kept_docs.append(doc)
        kept_metadata.append(meta)
        kept_vectors.append(vector.vectors[index])

    new_chunks = []
    for file_path in reader.file_list:
        rel_path = os.path.relpath(file_path, reader.data_path)
        if rel_path in sources_to_update:
            new_chunks.extend(reader.get_chunks_for_file(file_path))

    if new_chunks:
        embedding = OpenAIEmbedding()
        new_embeddings = embedding.get_embeddings([chunk.text for chunk in new_chunks])
        for chunk, embedding_vector in zip(new_chunks, new_embeddings):
            kept_docs.append(chunk.text)
            kept_metadata.append(chunk.metadata.to_dict())
            kept_vectors.append(embedding_vector)

    vector.document = kept_docs
    vector.metadata = kept_metadata
    vector.vectors = kept_vectors
    vector.persist(path=storage_path)
    _save_manifest(storage_path, reader)

    reason = "切块参数变更，" if version_changed else ""
    print(
        f"增量更新: {reason}移除/更新 {len(sources_to_remove)} 个文件, "
        f"新增 {len(new_chunks)} 个 chunk, 当前共 {len(kept_docs)} 个 chunk"
    )
    return vector


def build_index(
    data_path: str = REPORT_DATA_PATH,
    storage_path: str = STORAGE_PATH,
    force: bool = False,
) -> VectorStore:
    check_env()

    reader = ReadFiles(data_path)
    if not reader.file_list:
        raise NoDataError(f"在 {data_path} 下未找到 docx 文件")

    if os.path.exists(storage_path) and not force:
        return _incremental_build(reader, storage_path)

    if force and os.path.exists(storage_path):
        shutil.rmtree(storage_path)

    print(f"扫描到 {len(reader.file_list)} 个 docx 文件")
    chunks = reader.get_chunks()
    print(f"切分为 {len(chunks)} 个项目 chunk")

    vector = _chunks_to_vector(chunks)
    embedding = OpenAIEmbedding()
    vector.compute_embeddings(embedding_model=embedding)
    vector.persist(path=storage_path)
    _save_manifest(storage_path, reader)
    print(f"向量库已保存到 {storage_path}")
    return vector
