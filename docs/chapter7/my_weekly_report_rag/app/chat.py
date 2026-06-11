#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""问答编排：检索 → LLM 生成 → 格式化输出。"""

import logging
from typing import Any, Optional

from config import DEFAULT_AUTO_DATE, DEFAULT_K, INDEX_PATH
from exceptions import WeeklyReportRagError
from generation.llm import EMPTY_CONTEXT
from generation.output import (
    build_numbered_context,
    format_answer_with_citations,
    print_search_results,
    results_to_citations,
)
from models import DEFAULT_SEARCH_MODE, ChatResponse, SearchMode
from retrieval.session import RAGSession, search
from utils import format_filter_desc, resolve_date_filter

logger = logging.getLogger(__name__)


def _log_qa_exchange(
    question: str,
    answer: str,
    *,
    source: str,
    mode: SearchMode,
    k: int,
    filter_year: Optional[int],
    filter_month: Optional[int],
    request_info: Optional[dict[str, Any]] = None,
) -> None:
    """将请求元信息、问题与回答写入日志。"""
    meta_parts = [f"来源={source}", f"模式={mode}", f"k={k}"]
    if filter_year is not None:
        meta_parts.append(f"年={filter_year}")
    if filter_month is not None:
        meta_parts.append(f"月={filter_month}")
    if request_info:
        for key, value in request_info.items():
            if value is not None and value != "":
                meta_parts.append(f"{key}={value}")
    logger.info("问答请求 [%s]", ", ".join(meta_parts))
    logger.info("用户问题: %s", question)
    logger.info("回答内容: %s", answer)


def _get_session(session: Optional[RAGSession], index_path: str) -> RAGSession:
    """复用已有会话，避免重复加载索引。"""
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
    source: str = "cli",
    request_info: Optional[dict[str, Any]] = None,
) -> ChatResponse:
    """核心问答流程，返回结构化结果（Web / CLI 共用）。"""
    active_session = _get_session(session, index_path)
    filter_year, filter_month = resolve_date_filter(question, year, month, auto_date)

    # 1. 混合检索
    results = search(
        question,
        k=k,
        year=filter_year,
        month=filter_month,
        mode=mode,
        session=active_session,
    )

    # 2. 组装带编号的上下文，调用 LLM 生成回答
    context = build_numbered_context(results) if results else EMPTY_CONTEXT
    answer = active_session.chat.chat(question, context)
    response = ChatResponse(
        answer=answer,
        citations=results_to_citations(results) if results else [],
        mode=mode,
        filter_year=filter_year,
        filter_month=filter_month,
        search_results=results,
    )
    _log_qa_exchange(
        question,
        answer,
        source=source,
        mode=mode,
        k=k,
        filter_year=filter_year,
        filter_month=filter_month,
        request_info=request_info,
    )
    return response


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
    source: str = "cli",
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
        source=source,
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
    """CLI 交互式问答循环，每轮独立检索。"""
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
            print(
                f"\n{ask(question, k=k, debug=debug, year=year, month=month, auto_date=auto_date, mode=mode, session=session, source='cli-interactive')}"
            )
        except WeeklyReportRagError as exc:
            print(f"\n错误: {exc}")
