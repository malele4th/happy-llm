#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os

import numpy as np

from exceptions import StorageCorruptError
from indexing.store import IndexRecord
from models import ChunkMetadata

logger = logging.getLogger(__name__)


def load_legacy_index(path: str) -> tuple[list[IndexRecord], np.ndarray]:
    with open(os.path.join(path, "document.json"), "r", encoding="utf-8") as handle:
        documents = json.load(handle)
    metadata_path = os.path.join(path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata_list = json.load(handle)
    else:
        metadata_list = [{} for _ in documents]

    records = [
        IndexRecord(text=text, metadata=ChunkMetadata.from_dict(meta))
        for text, meta in zip(documents, metadata_list)
    ]

    vectors_npy = os.path.join(path, "vectors.npy")
    vectors_json = os.path.join(path, "vectors.json")
    if os.path.exists(vectors_npy):
        matrix = np.load(vectors_npy).astype(np.float32)
    elif os.path.exists(vectors_json):
        with open(vectors_json, "r", encoding="utf-8") as handle:
            matrix = np.array(json.load(handle), dtype=np.float32)
    else:
        raise StorageCorruptError("缺少向量文件")

    if len(records) != len(matrix):
        raise StorageCorruptError("legacy 索引记录与向量数量不一致")

    for index, record in enumerate(records):
        record.attach_vector(matrix[index])

    logger.info("已从 legacy document/metadata 格式加载索引")
    return records, matrix
