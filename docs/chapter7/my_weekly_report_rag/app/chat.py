#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

from config import DEFAULT_K, STORAGE_PATH
from generation.output import (
    build_numbered_context,
    format_answer_with_citations,
    print_search_results,
)
from models import DEFAULT_SEARCH_MODE, SearchMode
from retrieval.session import RAGSession, resolve_date_filter, search


def ask(
    question: str,
    storage_path: str = STORAGE_PATH,
    k: int = DEFAULT_K,
    debug: bool = False,
    year: Optional[int] = None,
    month: Optional[int] = None,
    auto_date: bool = False,
    mode: SearchMode = DEFAULT_SEARCH_MODE,
    session: Optional[RAGSession] = None,
) -> str:
    active_session = session or RAGSession(storage_path)
    filter_year, filter_month = resolve_date_filter(question, year, month, auto_date)

    results = search(
        question,
        k=k,
        year=filter_year,
        month=filter_month,
        mode=mode,
        session=active_session,
    )

    if debug:
        filter_desc = (
            f"year={filter_year}, month={filter_month}"
            if filter_year is not None
            else "无"
        )
        print(f"检索模式: {mode} | 过滤: {filter_desc}")
        print_search_results(results, show_scores=True)

    if not results:
        return "周报中没有找到相关内容，请尝试换个问法或指定 --year/--month。"

    context = build_numbered_context(results)
    answer = active_session.chat.chat(question, context)
    return format_answer_with_citations(answer, results)


def interactive_chat(
    storage_path: str = STORAGE_PATH,
    k: int = DEFAULT_K,
    debug: bool = False,
    year: Optional[int] = None,
    month: Optional[int] = None,
    auto_date: bool = False,
    mode: SearchMode = DEFAULT_SEARCH_MODE,
) -> None:
    session = RAGSession(storage_path)
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

        answer = ask(
            question,
            k=k,
            debug=debug,
            year=year,
            month=month,
            auto_date=auto_date,
            mode=mode,
            session=session,
        )
        print(f"\n{answer}")
