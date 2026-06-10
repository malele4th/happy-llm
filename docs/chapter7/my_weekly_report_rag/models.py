#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional, Tuple

from utils import normalize_report_date

SearchMode = Literal["latest", "timeline", "compare"]
SEARCH_MODES: Tuple[SearchMode, ...] = ("latest", "timeline", "compare")
DEFAULT_SEARCH_MODE: SearchMode = "latest"


@dataclass
class ChunkMetadata:
    source: str
    report_date: str = ""
    project: str = "综合"
    chunk_index: int = 0
    author: str = ""
    quarter: str = ""
    section_type: str = "body"

    def __post_init__(self) -> None:
        self.report_date = normalize_report_date(self.report_date)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkMetadata":
        return cls(
            source=data.get("source", ""),
            report_date=data.get("report_date", ""),
            project=data.get("project", "综合"),
            chunk_index=int(data.get("chunk_index", 0)),
            author=data.get("author", ""),
            quarter=data.get("quarter", ""),
            section_type=data.get("section_type", "body"),
        )

    def year_month(self) -> str:
        if len(self.report_date) >= 6:
            return self.report_date[:6]
        return "unknown"

    def identity_key(self) -> Tuple[str, str, int]:
        return (self.source, self.project, self.chunk_index)

    def compare_key(self) -> Tuple[str, str]:
        return (self.year_month(), self.project)

    def sort_key(self) -> Tuple[str, int]:
        return (self.report_date, self.chunk_index)


@dataclass
class DocumentChunk:
    text: str
    metadata: ChunkMetadata


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: ChunkMetadata
    vector_score: float = 0.0
    keyword_score: float = 0.0


@dataclass
class Citation:
    index: int
    date: str
    project: str
    score: float
    preview: str
    source: str = ""


@dataclass
class ChatResponse:
    answer: str
    citations: List[Citation] = field(default_factory=list)
    mode: str = DEFAULT_SEARCH_MODE
    filter_year: Optional[int] = None
    filter_month: Optional[int] = None
