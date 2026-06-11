#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import MagicMock, patch

from app.chat import ask, ask_detail
from generation.llm import EMPTY_CONTEXT
from models import ChunkMetadata, SearchResult


class ChatTestCase(unittest.TestCase):
    @patch("app.chat.search")
    def test_ask_detail_no_results_still_calls_llm(self, mock_search) -> None:
        mock_search.return_value = []
        mock_session = MagicMock()
        mock_session.chat.chat.return_value = "【非周报内容】我是 malele 周报助手。"

        result = ask_detail("你是谁", session=mock_session)

        mock_session.chat.chat.assert_called_once_with("你是谁", EMPTY_CONTEXT)
        self.assertIn("非周报内容", result.answer)
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
