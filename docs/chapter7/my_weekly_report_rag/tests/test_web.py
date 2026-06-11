#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from models import ChatResponse, Citation
from web.schemas import CitationOut
from web.server import create_app


class WebApiTestCase(unittest.TestCase):
    @contextmanager
    def _client(self, access_token: str = ""):
        mock_session = MagicMock()
        mock_session.store.record_count = 2
        with (
            patch("web.server.setup_logging"),
            patch("web.server.cleanup_tmp_dirs"),
            patch("web.server.check_env"),
            patch("web.server.RAGSession", return_value=mock_session),
            patch("web.server.WEB_ACCESS_TOKEN", access_token),
        ):
            with TestClient(create_app()) as client:
                yield client

    def test_health(self) -> None:
        with self._client() as client:
            response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["chunk_count"], 2)

    @patch("web.server.ask_detail")
    def test_chat(self, mock_ask_detail) -> None:
        mock_ask_detail.return_value = ChatResponse(
            answer="测试回答",
            citations=[
                Citation(
                    index=1,
                    date="2025-12-01",
                    project="catchii",
                    score=0.8,
                    preview="进展说明",
                    source="a.docx",
                )
            ],
            mode="latest",
        )
        with self._client() as client:
            response = client.post(
                "/api/chat",
                json={"message": "catchii 进展"},
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "测试回答")
        self.assertEqual(len(payload["citations"]), 1)
        self.assertEqual(payload["citations"][0]["project"], "catchii")

    def test_chat_requires_token_when_configured(self) -> None:
        with self._client(access_token="secret") as client:
            response = client.post("/api/chat", json={"message": "test"})
        self.assertEqual(response.status_code, 401)

    @patch("web.server.ask_detail")
    def test_chat_with_valid_token(self, mock_ask_detail) -> None:
        mock_ask_detail.return_value = ChatResponse(answer="ok", mode="latest")
        headers = {"Authorization": "Bearer secret"}
        with self._client(access_token="secret") as client:
            response = client.post(
                "/api/chat",
                json={"message": "test"},
                headers=headers,
            )
        self.assertEqual(response.status_code, 200)

    def test_citation_out_from_citation(self) -> None:
        citation = Citation(
            index=1,
            date="2025-12-01",
            project="catchii",
            score=0.5,
            preview="preview",
            source="a.docx",
        )
        out = CitationOut.from_citation(citation)
        self.assertEqual(out.project, "catchii")


if __name__ == "__main__":
    unittest.main()
