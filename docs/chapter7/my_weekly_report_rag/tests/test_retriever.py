#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np

from indexing.store import IndexRecord, IndexStore
from models import ChunkMetadata
from retrieval.retriever import Retriever


class _MockEmbedding:
    def get_embedding(self, text: str, kind: str = "passage"):
        if "catchii" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]

    def get_embeddings(self, texts, kind: str = "passage"):
        return [self.get_embedding(text, kind=kind) for text in texts]


class RetrieverTestCase(unittest.TestCase):
    def _record(self, text: str, metadata: ChunkMetadata, vector) -> IndexRecord:
        record = IndexRecord(text=text, metadata=metadata)
        record.attach_vector(np.array(vector, dtype=np.float32))
        return record

    def setUp(self) -> None:
        records = [
            self._record(
                "[meta]\ncatchii 家族房 12/11 评审",
                ChunkMetadata(
                    source="2025/Q4/a.docx",
                    report_date="20251211",
                    project="catchii房间需求",
                ),
                [1.0, 0.0],
            ),
            self._record(
                "[meta]\nrank 模型优化",
                ChunkMetadata(
                    source="2025/Q4/b.docx",
                    report_date="20251218",
                    project="rank模型",
                ),
                [0.0, 1.0],
            ),
        ]
        self.store = IndexStore(records)
        self.retriever = Retriever(self.store)

    def test_query_prefers_matching_project(self) -> None:
        results = self.retriever.query(
            "catchii家族房",
            embedding_model=_MockEmbedding(),
            k=1,
            mode="latest",
        )
        self.assertEqual(len(results), 1)
        self.assertIn("catchii", results[0].metadata.project)


if __name__ == "__main__":
    unittest.main()
