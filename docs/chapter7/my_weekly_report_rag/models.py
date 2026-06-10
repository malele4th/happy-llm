#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import asdict, dataclass
from typing import Literal, Tuple

SearchMode = Literal["latest", "timeline", "compare"]
SEARCH_MODES: Tuple[SearchMode, ...] = ("latest", "timeline", "compare")
DEFAULT_SEARCH_MODE: SearchMode = "latest"


@dataclass
class ChunkMetadata:
    source: str
    report_date: str = ""
    project: str = "综合"
    chunk_index: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkMetadata":
        return cls(
            source=data.get("source", ""),
            report_date=data.get("report_date", ""),
            project=data.get("project", "综合"),
            chunk_index=int(data.get("chunk_index", 0)),
        )

    def year_month(self) -> str:
        if len(self.report_date) >= 6:
            return self.report_date[:6]
        return "unknown"

    def identity_key(self) -> Tuple[str, str, int]:
        return (self.source, self.project, self.chunk_index)


@dataclass
class DocumentChunk:
    text: str
    metadata: ChunkMetadata


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: ChunkMetadata
