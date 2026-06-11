#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""索引单条记录：文本、元数据与对应向量。"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from exceptions import IndexCorruptError
from models import ChunkMetadata


@dataclass
class IndexRecord:
    """索引中的一条记录，向量通过 attach_vector 延迟绑定。"""

    text: str
    metadata: ChunkMetadata
    _vector: Optional[np.ndarray] = None

    def attach_vector(self, vector: np.ndarray) -> None:
        """绑定或更新该记录对应的向量。"""
        self._vector = vector.astype(np.float32)

    @property
    def vector(self) -> np.ndarray:
        if self._vector is None:
            raise IndexCorruptError("记录缺少向量")
        return self._vector

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "metadata": self.metadata.to_dict(),
        }
