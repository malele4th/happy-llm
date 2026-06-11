#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""领域数据模型：chunk、检索结果、问答响应。"""

from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional, Tuple

from config import NO_RESULTS_MESSAGE
from utils import normalize_report_date

SearchMode = Literal["latest", "timeline", "compare"]
SEARCH_MODES: Tuple[SearchMode, ...] = ("latest", "timeline", "compare")
DEFAULT_SEARCH_MODE: SearchMode = "latest"


@dataclass
class ChunkMetadata:
    """单个 chunk 的元数据，用于检索过滤、去重与引用展示。"""

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
        """返回 YYYYMM，用于按月对比去重。"""
        return self.report_date[:6] if len(self.report_date) >= 6 else "unknown"

    def identity_key(self) -> Tuple[str, str, int]:
        """唯一标识一个 chunk（同文件、同项目、同序号）。"""
        return (self.source, self.project, self.chunk_index)

    def compare_key(self) -> Tuple[str, str]:
        """按月对比模式的去重键：(年月, 项目)。"""
        return (self.year_month(), self.project)

    def sort_key(self) -> Tuple[str, int]:
        """时间线模式的排序键：(日期, chunk 序号)。"""
        return (self.report_date, self.chunk_index)


@dataclass
class DocumentChunk:
    """索引构建阶段的文本块：正文 + 元数据。"""

    text: str
    metadata: ChunkMetadata


@dataclass
class SearchResult:
    """检索命中结果，含融合分数与向量/BM25 分项。"""

    text: str
    score: float
    metadata: ChunkMetadata
    vector_score: float = 0.0
    keyword_score: float = 0.0


@dataclass
class Citation:
    """面向用户展示的引用条目。"""

    index: int
    date: str
    project: str
    score: float
    preview: str
    source: str = ""


@dataclass
class ChatResponse:
    """问答完整结果，供 Web API 与 CLI 格式化输出。"""

    answer: str
    citations: List[Citation] = field(default_factory=list)
    mode: str = DEFAULT_SEARCH_MODE
    filter_year: Optional[int] = None
    filter_month: Optional[int] = None
    search_results: List[SearchResult] = field(default_factory=list, repr=False)

    @classmethod
    def not_found(
        cls,
        mode: SearchMode,
        filter_year: Optional[int],
        filter_month: Optional[int],
        message: str = NO_RESULTS_MESSAGE,
    ) -> "ChatResponse":
        """检索无命中时的快捷构造。"""
        return cls(
            answer=message,
            mode=mode,
            filter_year=filter_year,
            filter_month=filter_month,
        )
