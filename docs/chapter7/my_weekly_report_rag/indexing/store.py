#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""向量索引的加载、持久化与内存管理。"""

import json
import logging
import os
import shutil
from typing import List, Optional

import numpy as np

from config import INDEX_PATH, INDEX_TMP_DIR, MANIFEST_FILE
from exceptions import IndexCorruptError, IndexNotFoundError
from indexing.record import IndexRecord
from models import ChunkMetadata, DocumentChunk

logger = logging.getLogger(__name__)


def load_index(index_path: str = INDEX_PATH) -> "IndexStore":
    if not IndexStore.exists(index_path):
        raise IndexNotFoundError(
            "索引目录不存在或不完整，请先运行: python main.py --build"
        )
    store = IndexStore()
    store.load_from_disk(index_path)
    return store


class IndexStore:
    def __init__(self, records: Optional[List[IndexRecord]] = None) -> None:
        self.records: List[IndexRecord] = records or []
        self._matrix: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.records)

    @property
    def record_count(self) -> int:
        return len(self.records)

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
            os.path.isfile(os.path.join(index_path, "records.json"))
            and os.path.isfile(os.path.join(index_path, "vectors.npy"))
        )

    @classmethod
    def from_chunks(cls, chunks: List[DocumentChunk]) -> "IndexStore":
        return cls([IndexRecord(text=chunk.text, metadata=chunk.metadata) for chunk in chunks])

    def invalidate_matrix(self) -> None:
        """记录变更后使向量矩阵缓存失效。"""
        self._matrix = None

    def set_vectors(self, vectors: List[List[float]]) -> None:
        if len(vectors) != len(self.records):
            raise IndexCorruptError("向量数量与记录数量不一致")
        for record, vector in zip(self.records, vectors):
            record.attach_vector(np.array(vector, dtype=np.float32))
        self._matrix = np.array(vectors, dtype=np.float32)

    def append_records(self, chunks: List[DocumentChunk], vectors: List[List[float]]) -> None:
        if len(chunks) != len(vectors):
            raise IndexCorruptError("新增 chunk 与向量数量不一致")
        for chunk, vector in zip(chunks, vectors):
            record = IndexRecord(text=chunk.text, metadata=chunk.metadata)
            record.attach_vector(np.array(vector, dtype=np.float32))
            self.records.append(record)
        self.invalidate_matrix()

    def persist(self, path: str = INDEX_PATH, manifest: Optional[dict] = None) -> None:
        """原子写入：临时目录写完后再 rename 替换，失败时回滚。"""
        parent_dir = os.path.dirname(os.path.abspath(path)) or "."
        tmp_path = os.path.join(parent_dir, INDEX_TMP_DIR)
        backup_path = f"{path}.old"

        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path, ignore_errors=True)
        os.makedirs(tmp_path, exist_ok=True)

        with open(os.path.join(tmp_path, "records.json"), "w", encoding="utf-8") as handle:
            json.dump([record.to_dict() for record in self.records], handle, ensure_ascii=False)

        matrix = self.vector_matrix
        np.save(
            os.path.join(tmp_path, "vectors.npy"),
            matrix if matrix.size else np.empty((0, 0), dtype=np.float32),
        )

        if manifest is not None:
            with open(os.path.join(tmp_path, MANIFEST_FILE), "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)

        self._validate_lengths()
        self._atomic_replace(tmp_path, path, backup_path)
        logger.info("索引已保存到 %s (%s 条记录)", path, len(self.records))

    def _atomic_replace(self, tmp_path: str, target_path: str, backup_path: str) -> None:
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path, ignore_errors=True)
        if os.path.exists(target_path):
            os.rename(target_path, backup_path)
        try:
            os.rename(tmp_path, target_path)
        except OSError:
            if os.path.exists(target_path):
                shutil.rmtree(target_path, ignore_errors=True)
            if os.path.exists(backup_path):
                os.rename(backup_path, target_path)
            raise
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path, ignore_errors=True)

    def load_from_disk(self, path: str = INDEX_PATH) -> None:
        records_path = os.path.join(path, "records.json")
        vectors_path = os.path.join(path, "vectors.npy")
        if not os.path.exists(records_path):
            raise IndexCorruptError("缺少 records.json")
        if not os.path.exists(vectors_path):
            raise IndexCorruptError("缺少 vectors.npy")

        with open(records_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.records = [
            IndexRecord(text=item["text"], metadata=ChunkMetadata.from_dict(item["metadata"]))
            for item in payload
        ]
        self._matrix = np.load(vectors_path).astype(np.float32)
        self._attach_vectors()
        self._validate_lengths()

    def _attach_vectors(self) -> None:
        if self._matrix is None or not self.records:
            return
        for index, record in enumerate(self.records):
            record.attach_vector(self._matrix[index])

    def _validate_lengths(self) -> None:
        if not self.records:
            return
        if any(record._vector is None for record in self.records):
            raise IndexCorruptError("存在未加载向量的记录")
        if self._matrix is not None and len(self.records) != len(self._matrix):
            raise IndexCorruptError(
                f"记录数 {len(self.records)} 与向量数 {len(self._matrix)} 不一致"
            )
