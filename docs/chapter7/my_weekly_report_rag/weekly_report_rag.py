#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""周报 RAG 系统 CLI 入口。"""

import argparse

from config import DEFAULT_K, REPORT_DATA_PATH, STORAGE_PATH
from service import ask, build_index, interactive_chat, print_search_results, search


def main() -> None:
    parser = argparse.ArgumentParser(description="周报 RAG 系统")
    parser.add_argument("--build", action="store_true", help="构建向量索引")
    parser.add_argument("--force", action="store_true", help="强制重建索引（覆盖已有 storage）")
    parser.add_argument("--query", type=str, help="单次问答")
    parser.add_argument("--search", type=str, help="仅检索，不调用 LLM（调试用）")
    parser.add_argument("--chat", action="store_true", help="交互式问答")
    parser.add_argument("--debug", action="store_true", help="显示检索分数与过滤条件")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"检索 top-k 片段（默认 {DEFAULT_K}）")
    parser.add_argument("--data-path", type=str, default=REPORT_DATA_PATH, help="周报原始数据路径")
    parser.add_argument("--storage", type=str, default=STORAGE_PATH, help="向量库保存路径")
    args = parser.parse_args()

    if args.build:
        build_index(args.data_path, args.storage, force=args.force)
    elif args.search:
        results = search(args.search, args.storage, k=args.k)
        print_search_results(results)
    elif args.query:
        print(ask(args.query, args.storage, k=args.k, debug=args.debug))
    elif args.chat:
        interactive_chat(args.storage, k=args.k, debug=args.debug)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
