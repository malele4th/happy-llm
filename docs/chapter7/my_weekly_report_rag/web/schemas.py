#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from models import SEARCH_MODES


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["latest", "timeline", "compare"] = "latest"
    auto_date: bool = True
    year: Optional[int] = None
    month: Optional[int] = None
    k: int = Field(default=5, ge=1, le=20)


class CitationOut(BaseModel):
    index: int
    date: str
    project: str
    score: float
    preview: str
    source: str = ""


class ChatResponseOut(BaseModel):
    answer: str
    citations: List[CitationOut]
    mode: str
    filter_year: Optional[int] = None
    filter_month: Optional[int] = None


class HealthOut(BaseModel):
    status: str
    chunk_count: int
    modes: List[str] = list(SEARCH_MODES)
