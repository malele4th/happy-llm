#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from models import ChunkMetadata


class ModelsTestCase(unittest.TestCase):
    def test_identity_and_compare_keys(self) -> None:
        meta = ChunkMetadata(
            source="2025/Q1/a.docx",
            report_date="20251211",
            project="catchii",
            chunk_index=1,
        )
        self.assertEqual(meta.identity_key(), ("2025/Q1/a.docx", "catchii", 1))
        self.assertEqual(meta.compare_key(), ("202512", "catchii"))

    def test_from_dict_defaults(self) -> None:
        meta = ChunkMetadata.from_dict({"source": "a.docx"})
        self.assertEqual(meta.section_type, "body")
        self.assertEqual(meta.chunk_index, 0)


if __name__ == "__main__":
    unittest.main()
