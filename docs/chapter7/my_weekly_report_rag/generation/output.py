#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检索结果与问答输出的格式化。"""

from typing import List

from models import Citation, SearchResult
from utils import format_report_date


def extract_chunk_body(text: str) -> str:
    """去掉 chunk 头部 meta 行，返回正文。"""
    return text.split("\n", 1)[-1] if text else ""


def results_to_citations(results: List[SearchResult]) -> List[Citation]:
    """将检索结果转为 Web/CLI 可用的引用列表。"""
    citations: List[Citation] = []
    for index, result in enumerate(results, 1):
        preview = extract_chunk_body(result.text)[:200].replace("\n", " ")
        citations.append(
            Citation(
                index=index,
                date=format_report_date(result.metadata.report_date) or "?",
                project=result.metadata.project or "?",
                score=result.score,
                preview=preview,
                source=result.metadata.source,
            )
        )
    return citations


def format_result_summary(
    index: int,
    result: SearchResult,
    show_scores: bool = False,
) -> str:
    """单条检索结果的一行摘要。"""
    date = format_report_date(result.metadata.report_date) or "?"
    project = result.metadata.project or "?"
    line = f"[{index}] {date} | {project} | score={result.score:.3f}"
    if show_scores:
        line += f" (vec={result.vector_score:.3f}, bm25={result.keyword_score:.3f})"
    return line


def build_numbered_context(results: List[SearchResult]) -> str:
    """将检索片段编号拼接为 LLM 上下文。"""
    return "\n\n---\n\n".join(
        f"[{index}]\n{result.text}" for index, result in enumerate(results, 1)
    )


def format_citations(results: List[SearchResult]) -> str:
    """格式化引用来源块，供 CLI 输出。"""
    lines = ["【引用】"]
    for index, result in enumerate(results, 1):
        lines.append(format_result_summary(index, result))
    return "\n".join(lines)


def format_answer_with_citations(answer: str, results: List[SearchResult]) -> str:
    """拼接回答正文与引用来源。"""
    return f"【回答】\n{answer}\n\n{format_citations(results)}"


def print_search_results(
    results: List[SearchResult],
    verbose: bool = False,
    show_scores: bool = False,
) -> None:
    """CLI 调试：打印检索命中列表。"""
    if not results:
        print("  (未检索到满足相似度阈值的片段)")
        return
    for index, result in enumerate(results, 1):
        print(f"  {format_result_summary(index, result, show_scores=show_scores)}")
        body = extract_chunk_body(result.text)
        if verbose:
            print(f"      {body}")
        else:
            print(f"      {body[:120].replace(chr(10), ' ')}...")
