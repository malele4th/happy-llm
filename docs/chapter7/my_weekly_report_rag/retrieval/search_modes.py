#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检索后处理：按模式去重、排序并截取 top-k。"""

from typing import List

from config import SEARCH_CANDIDATE_POOL_FACTOR
from models import SearchMode, SearchResult


def dedupe_latest(results: List[SearchResult]) -> List[SearchResult]:
    """同项目只保留最新周报的一条结果。"""
    by_project: dict[str, SearchResult] = {}
    for result in results:
        project = result.metadata.project
        report_date = result.metadata.report_date
        if project not in by_project:
            by_project[project] = result
            continue
        existing = by_project[project]
        if report_date > existing.metadata.report_date or (
            report_date == existing.metadata.report_date and result.score > existing.score
        ):
            by_project[project] = result
    return sorted(by_project.values(), key=lambda item: item.score, reverse=True)


def dedupe_compare(results: List[SearchResult]) -> List[SearchResult]:
    """按月+项目去重，用于跨月对比。"""
    by_key: dict[tuple[str, str], SearchResult] = {}
    for result in results:
        key = result.metadata.compare_key()
        if key not in by_key or result.score > by_key[key].score:
            by_key[key] = result
    return sorted(by_key.values(), key=lambda item: item.metadata.sort_key())


def apply_search_mode(results: List[SearchResult], mode: SearchMode, k: int) -> List[SearchResult]:
    """根据检索模式对候选结果做后处理并返回 top-k。"""
    pool_size = max(k * SEARCH_CANDIDATE_POOL_FACTOR, k)
    candidates = results[:pool_size]
    if mode == "timeline":
        ordered = sorted(candidates, key=lambda item: item.metadata.sort_key())
        return ordered[:k]
    if mode == "compare":
        return dedupe_compare(candidates)[:k]
    return dedupe_latest(candidates)[:k]
