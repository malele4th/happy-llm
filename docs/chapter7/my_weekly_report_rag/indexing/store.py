#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os
import shutil
from typing import List, Optional

import numpy as np

from config import INDEX_PATH, INDEX_TMP_DIR
from exceptions import StorageCorruptError, StorageNotFoundError
from indexing.record import IndexRecord
from models import ChunkMetadata, DocumentChunk

logger = logging.getLogger(__name__)


def load_index(index_path: str = INDEX_PATH) -> "IndexStore":
    if not IndexStore.exists(index_path):
        raise StorageNotFoundError(
            "索引目录不存在或不完整，请先运行: python main.py --build"
        )
    store = IndexStore()
    store.load_from_disk(index_path)
    return store


class IndexStore:
    def __init__(self, records: Optional[List[IndexRecord]] = None) -> None:
        self.records: List[IndexRecord] = records or []
        self._matrix: Optional[np.ndarray] = None

    @property
    def vector_matrix(self) -> np.ndarray:
        if self._matrix is not None:
            return self._matrix
        if not self.records:
            return np.empty((0, 0), dtype=np.float32)
        self._matrix = np.array([record.vector for record in self.records], dtype=np.float32)
        return self._matrix

    @classmethod
    def exists(cls, index_path: str) -> bool:
        return (
            os.path.exists(os.path.join(index_path, "records.json"))
            or os.path.exists(os.path.join(index_path, "document.json"))
        )

    @classmethod
    def from_chunks(cls, chunks: List[DocumentChunk]) -> "IndexStore":
        return cls([IndexRecord(text=chunk.text, metadata=chunk.metadata) for chunk in chunks])

    def set_vectors(self, vectors: List[List[float]]) -> None:
        if len(vectors) != len(self.records):
            raise StorageCorruptError("向量数量与记录数量不一致")
        for record, vector in zip(self.records, vectors):
            record.attach_vector(np.array(vector, dtype=np.float32))
        self._matrix = np.array(vectors, dtype=np.float32)

    def append_records(self, chunks: List[DocumentChunk], vectors: List[List[float]]) -> None:
        if len(chunks) != len(vectors):
            raise StorageCorruptError("新增 chunk 与向量数量不一致")
        for chunk, vector in zip(chunks, vectors):
            record = IndexRecord(text=chunk.text, metadata=chunk.metadata)
            record.attach_vector(np.array(vector, dtype=np.float32))
            self.records.append(record)
        self._matrix = None

    def persist(self, path: str = INDEX_PATH) -> None:
        parent_dir = os.path.dirname(os.path.abspath(path)) or "."
        tmp_path = os.path.join(parent_dir, INDEX_TMP_DIR)
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path, ignore_errors=True)
        os.makedirs(tmp_path, exist_ok=True)

        payload = [record.to_storage_dict() for record in self.records]
        with open(os.path.join(tmp_path, "records.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)

        matrix = self.vector_matrix
        if matrix.size:
            np.save(os.path.join(tmp_path, "vectors.npy"), matrix)

        self._validate_lengths()
        if os.path.exists(path):
            shutil.rmtree(path)
        os.rename(tmp_path, path)
        logger.info("索引已保存到 %s (%s 条记录)", path, len(self.records))

    def load_from_disk(self, path: str = INDEX_PATH) -> None:
        records_path = os.path.join(path, "records.json")
        if os.path.exists(records_path):
            self._load_new_format(path)
        else:
            from indexing.legacy_io import load_legacy_index

            self.records, self._matrix = load_legacy_index(path)
        self._validate_lengths()

    def _load_new_format(self, path: str) -> None:
        with open(os.path.join(path, "records.json"), "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.records = [
            IndexRecord(text=item["text"], metadata=ChunkMetadata.from_dict(item["metadata"]))
            for item in payload
        ]
        vectors_npy = os.path.join(path, "vectors.npy")
        vectors_json = os.path.join(path, "vectors.json")
        if os.path.exists(vectors_npy):
            self._matrix = np.load(vectors_npy).astype(np.float32)
        elif os.path.exists(vectors_json):
            with open(vectors_json, "r", encoding="utf-8") as handle:
                self._matrix = np.array(json.load(handle), dtype=np.float32)
            logger.info("已从 legacy vectors.json 迁移向量格式")
        else:
            raise StorageCorruptError("缺少 vectors.npy 或 vectors.json")
        for index, record in enumerate(self.records):
            record.attach_vector(self._matrix[index])

    def _validate_lengths(self) -> None:
        if not self.records:
            return
        if any(record._vector is None for record in self.records):
            raise StorageCorruptError("存在未加载向量的记录")
        if self._matrix is not None and len(self.records) != len(self._matrix):
            raise StorageCorruptError(
                f"记录数 {len(self.records)} 与向量数 {len(self._matrix)} 不一致"
            )
