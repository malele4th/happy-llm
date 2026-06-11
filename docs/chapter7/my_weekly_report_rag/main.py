#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""周报 RAG 系统 CLI 入口。"""

import argparse
import logging
import sys
from argparse import Namespace

from app.chat import ask, interactive_chat
from config import (
    DEFAULT_AUTO_DATE,
    DEFAULT_K,
    INDEX_PATH,
    REPORT_DATA_PATH,
    check_env,
    cleanup_tmp_dirs,
    setup_logging,
)
from exceptions import WeeklyReportRagError
from generation.output import print_search_results
from indexing.pipeline import build_index
from models import DEFAULT_SEARCH_MODE, SEARCH_MODES
from retrieval.session import RAGSession, search
from utils import resolve_date_filter

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="周报 RAG 系统")
    parser.add_argument("--build", action="store_true", help="构建或增量更新向量索引")
    parser.add_argument("--force", action="store_true", help="强制全量重建索引（覆盖已有 data 目录）")
    parser.add_argument("--query", type=str, help="单次问答")
    parser.add_argument("--search", type=str, help="仅检索，不调用 LLM（调试用）")
    parser.add_argument("--chat", action="store_true", help="交互式问答")
    parser.add_argument("--web", action="store_true", help="启动 Web 服务（局域网可访问）")
    parser.add_argument("--debug", action="store_true", help="显示检索分数与过滤条件")
    parser.add_argument("--verbose", action="store_true", help="检索时打印完整 chunk 内容")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"检索 top-k 片段（默认 {DEFAULT_K}）")
    parser.add_argument(
        "--mode",
        choices=SEARCH_MODES,
        default=DEFAULT_SEARCH_MODE,
        help="检索模式: latest=同项目取最新, timeline=按时间线, compare=按月对比",
    )
    parser.add_argument("--year", type=int, help="按年份过滤检索结果")
    parser.add_argument("--month", type=int, help="按月份过滤检索结果（需配合 --year）")
    parser.add_argument(
        "--auto-date",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_AUTO_DATE,
        help="从问题文本自动解析年月并过滤",
    )
    parser.add_argument("--data-path", type=str, default=REPORT_DATA_PATH, help="周报原始数据路径")
    parser.add_argument("--index", type=str, default=INDEX_PATH, help="向量索引目录（默认 ./data）")
    return parser


def _run_search(args: Namespace) -> None:
    filter_year, filter_month = resolve_date_filter(
        args.search, args.year, args.month, args.auto_date
    )
    session = RAGSession(args.index)
    results = search(
        args.search,
        k=args.k,
        year=filter_year,
        month=filter_month,
        mode=args.mode,
        session=session,
    )
    if args.debug:
        print(f"检索模式: {args.mode}")
    print_search_results(results, verbose=args.verbose, show_scores=args.debug)


def _run_query(args: Namespace) -> None:
    session = RAGSession(args.index)
    print(
        ask(
            args.query,
            args.index,
            k=args.k,
            debug=args.debug,
            year=args.year,
            month=args.month,
            auto_date=args.auto_date,
            mode=args.mode,
            session=session,
        )
    )


def _dispatch(args: Namespace) -> None:
    if args.build:
        build_index(args.data_path, args.index, force=args.force)
    elif args.search:
        _run_search(args)
    elif args.query:
        _run_query(args)
    elif args.chat:
        interactive_chat(
            args.index,
            k=args.k,
            debug=args.debug,
            year=args.year,
            month=args.month,
            auto_date=args.auto_date,
            mode=args.mode,
        )
    else:
        _build_parser().print_help()


def main() -> None:
    args = _build_parser().parse_args()

    if args.web:
        from web.server import run_server

        run_server()
        return

    setup_logging()
    cleanup_tmp_dirs()
    check_env()

    try:
        _dispatch(args)
    except WeeklyReportRagError as exc:
        logger.error("%s", exc)
        print(f"错误: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
