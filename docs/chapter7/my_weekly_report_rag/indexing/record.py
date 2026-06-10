#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional

import numpy as np

from exceptions import IndexCorruptError
from models import ChunkMetadata


@dataclass
class IndexRecord:
    text: str
    metadata: ChunkMetadata
    _vector: Optional[np.ndarray] = None

    def attach_vector(self, vector: np.ndarray) -> None:
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
