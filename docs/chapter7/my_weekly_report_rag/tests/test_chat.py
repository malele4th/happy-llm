#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import MagicMock, patch

from app.chat import ask, ask_detail
from models import ChunkMetadata, SearchResult


class ChatTestCase(unittest.TestCase):
    @patch("app.chat.search")
    @patch("app.chat.RAGSession")
    def test_ask_detail_no_results(self, mock_session_cls, mock_search) -> None:
        mock_search.return_value = []
        mock_session_cls.return_value = MagicMock()

        result = ask_detail("测试问题", session=mock_session_cls.return_value)
        self.assertIn("没有找到", result.answer)
        self.assertEqual(result.citations, [])

    @patch("app.chat.ask_detail")
    def test_ask_formats_answer(self, mock_ask_detail) -> None:
        mock_ask_detail.return_value = MagicMock(
            answer="回答正文",
            citations=[],
            search_results=[
                SearchResult(
                    text="chunk",
                    score=0.9,
                    metadata=ChunkMetadata(source="a.docx", project="catchii"),
                )
            ],
        )
        text = ask("测试")
        self.assertIn("回答正文", text)
        self.assertIn("引用", text)


if __name__ == "__main__":
    unittest.main()
