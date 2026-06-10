#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List

from models import SearchResult
from utils import format_report_date


def format_result_summary(index: int, result: SearchResult) -> str:
    date = format_report_date(result.metadata.report_date) or "?"
    project = result.metadata.project or "?"
    return f"[{index}] {date} | {project} | score={result.score:.3f}"


def build_numbered_context(results: List[SearchResult]) -> str:
    return "\n\n---\n\n".join(
        f"[{index}]\n{result.text}" for index, result in enumerate(results, 1)
    )


def format_citations(results: List[SearchResult]) -> str:
    lines = ["【引用】"]
    for index, result in enumerate(results, 1):
        lines.append(format_result_summary(index, result))
    return "\n".join(lines)


def format_answer_with_citations(answer: str, results: List[SearchResult]) -> str:
    return f"【回答】\n{answer}\n\n{format_citations(results)}"


def print_search_results(results: List[SearchResult], verbose: bool = False) -> None:
    if not results:
        print("  (未检索到满足相似度阈值的片段)")
        return
    for index, result in enumerate(results, 1):
        print(f"  {format_result_summary(index, result)}")
        body = result.text.split("\n", 1)[-1]
        if verbose:
            print(f"      {body}")
        else:
            preview = body[:120].replace("\n", " ")
            print(f"      {preview}...")
