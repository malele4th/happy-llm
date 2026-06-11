#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""问答编排：检索 → LLM 生成 → 格式化输出。"""

from typing import Optional

from config import DEFAULT_AUTO_DATE, DEFAULT_K, INDEX_PATH
from exceptions import WeeklyReportRagError
from generation.output import (
    build_numbered_context,
    format_answer_with_citations,
    print_search_results,
    results_to_citations,
)
from models import DEFAULT_SEARCH_MODE, ChatResponse, SearchMode
from retrieval.session import RAGSession, search
from utils import format_filter_desc, resolve_date_filter


def _get_session(session: Optional[RAGSession], index_path: str) -> RAGSession:
    return session if session is not None else RAGSession(index_path)


def ask_detail(
    question: str,
    index_path: str = INDEX_PATH,
    k: int = DEFAULT_K,
    year: Optional[int] = None,
    month: Optional[int] = None,
    auto_date: bool = DEFAULT_AUTO_DATE,
    mode: SearchMode = DEFAULT_SEARCH_MODE,
    session: Optional[RAGSession] = None,
) -> ChatResponse:
    """核心问答流程，返回结构化结果（Web / CLI 共用）。"""
    active_session = _get_session(session, index_path)
    filter_year, filter_month = resolve_date_filter(question, year, month, auto_date)

    results = search(
        question,
        k=k,
        year=filter_year,
        month=filter_month,
        mode=mode,
        session=active_session,
    )

    if not results:
        return ChatResponse.not_found(mode, filter_year, filter_month)

    answer = active_session.chat.chat(question, build_numbered_context(results))
    return ChatResponse(
        answer=answer,
        citations=results_to_citations(results),
        mode=mode,
        filter_year=filter_year,
        filter_month=filter_month,
        search_results=results,
    )


def ask(
    question: str,
    index_path: str = INDEX_PATH,
    k: int = DEFAULT_K,
    debug: bool = False,
    year: Optional[int] = None,
    month: Optional[int] = None,
    auto_date: bool = DEFAULT_AUTO_DATE,
    mode: SearchMode = DEFAULT_SEARCH_MODE,
    session: Optional[RAGSession] = None,
) -> str:
    """CLI 问答，返回带引用的纯文本。"""
    detail = ask_detail(
        question,
        index_path=index_path,
        k=k,
        year=year,
        month=month,
        auto_date=auto_date,
        mode=mode,
        session=session,
    )

    if debug:
        print(f"检索模式: {detail.mode} | 过滤: {format_filter_desc(detail.filter_year, detail.filter_month)}")
        if detail.search_results:
            print_search_results(detail.search_results, show_scores=True)

    if not detail.search_results:
        return detail.answer

    return format_answer_with_citations(detail.answer, detail.search_results)


def interactive_chat(
    index_path: str = INDEX_PATH,
    k: int = DEFAULT_K,
    debug: bool = False,
    year: Optional[int] = None,
    month: Optional[int] = None,
    auto_date: bool = DEFAULT_AUTO_DATE,
    mode: SearchMode = DEFAULT_SEARCH_MODE,
) -> None:
    session = RAGSession(index_path)
    print("周报 RAG 交互模式（输入 quit 退出，每轮独立检索）")

    while True:
        try:
            question = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("再见")
            break

        try:
            print(f"\n{ask(question, k=k, debug=debug, year=year, month=month, auto_date=auto_date, mode=mode, session=session)}")
        except WeeklyReportRagError as exc:
            print(f"\n错误: {exc}")
