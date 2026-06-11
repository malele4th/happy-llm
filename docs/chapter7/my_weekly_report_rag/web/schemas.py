#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Web API 请求/响应模型。"""

from typing import TYPE_CHECKING, List, Literal, Optional

from pydantic import BaseModel, Field

from config import DEFAULT_AUTO_DATE, DEFAULT_K
from models import Citation, SEARCH_MODES

if TYPE_CHECKING:
    from models import ChatResponse


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["latest", "timeline", "compare"] = "latest"
    auto_date: bool = DEFAULT_AUTO_DATE
    year: Optional[int] = None
    month: Optional[int] = Field(default=None, ge=1, le=12)
    k: int = Field(default=DEFAULT_K, ge=1, le=20)


class CitationOut(BaseModel):
    index: int
    date: str
    project: str
    score: float
    preview: str
    source: str = ""

    @classmethod
    def from_citation(cls, citation: Citation) -> "CitationOut":
        return cls(**citation.__dict__)


class ChatResponseOut(BaseModel):
    answer: str
    citations: List[CitationOut]
    mode: str
    filter_year: Optional[int] = None
    filter_month: Optional[int] = None

    @classmethod
    def from_response(cls, response: "ChatResponse") -> "ChatResponseOut":
        return cls(
            answer=response.answer,
            citations=[CitationOut.from_citation(c) for c in response.citations],
            mode=response.mode,
            filter_year=response.filter_year,
            filter_month=response.filter_month,
        )


class HealthOut(BaseModel):
    status: str
    chunk_count: int
    modes: List[str] = list(SEARCH_MODES)
