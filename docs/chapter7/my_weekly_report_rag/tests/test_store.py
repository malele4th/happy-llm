#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from indexing.store import IndexStore
from models import ChunkMetadata, DocumentChunk


class IndexStoreTestCase(unittest.TestCase):
    def _sample_store(self) -> IndexStore:
        chunks = [
            DocumentChunk(
                text="meta\nbody one",
                metadata=ChunkMetadata(source="a.docx", project="p1", report_date="20251201"),
            ),
            DocumentChunk(
                text="meta\nbody two",
                metadata=ChunkMetadata(source="b.docx", project="p2", report_date="20251208"),
            ),
        ]
        store = IndexStore.from_chunks(chunks)
        store.set_vectors([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        return store

    def test_persist_and_reload_roundtrip(self) -> None:
        store = self._sample_store()
        manifest = {"index_version": "test", "files": {}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = os.path.join(tmp_dir, "data")
            store.persist(path=index_path, manifest=manifest)

            self.assertTrue(IndexStore.exists(index_path))
            self.assertTrue(os.path.isfile(os.path.join(index_path, "manifest.json")))

            reloaded = IndexStore()
            reloaded.load_from_disk(index_path)
            self.assertEqual(reloaded.record_count, 2)
            self.assertEqual(reloaded.records[0].metadata.project, "p1")

    def test_persist_empty_records(self) -> None:
        store = IndexStore()
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = os.path.join(tmp_dir, "data")
            store.persist(path=index_path)

            reloaded = IndexStore()
            reloaded.load_from_disk(index_path)
            self.assertEqual(reloaded.record_count, 0)
            matrix = np.load(os.path.join(index_path, "vectors.npy"))
            self.assertEqual(matrix.shape, (0, 0))

    def test_atomic_replace_restores_old_index_on_failure(self) -> None:
        import indexing.store as store_module

        real_rename = store_module.os.rename

        def selective_rename(src: str, dst: str) -> None:
            if ".index_tmp" in src:
                raise OSError("disk full")
            return real_rename(src, dst)

        store = self._sample_store()
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = os.path.join(tmp_dir, "data")
            store.persist(path=index_path)

            updated = self._sample_store()
            updated.records[0].metadata.project = "changed"
            with patch.object(store_module.os, "rename", side_effect=selective_rename):
                with self.assertRaises(OSError):
                    updated.persist(path=index_path)

            reloaded = IndexStore()
            reloaded.load_from_disk(index_path)
            self.assertEqual(reloaded.records[0].metadata.project, "p1")


if __name__ == "__main__":
    unittest.main()
