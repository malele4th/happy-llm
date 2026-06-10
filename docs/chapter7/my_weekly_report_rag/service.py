#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import sys
from typing import List, Optional

from config import DEFAULT_K, REPORT_DATA_PATH, STORAGE_PATH
from Embeddings import OpenAIEmbedding
from LLM import OpenAIChat
from utils import ReadFiles, parse_date_filter
from VectorBase import SearchResult, VectorStore


def check_env() -> None:
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        print("错误: 请在 .env 中配置 OPENAI_API_KEY 和 OPENAI_BASE_URL")
        sys.exit(1)


def build_index(
    data_path: str = REPORT_DATA_PATH,
    storage_path: str = STORAGE_PATH,
    force: bool = False,
) -> VectorStore:
    check_env()
    if os.path.exists(storage_path) and not force:
        print(f"storage 已存在: {storage_path}，使用 --force 强制重建")
        sys.exit(1)

    if force and os.path.exists(storage_path):
        shutil.rmtree(storage_path)

    reader = ReadFiles(data_path)
    print(f"扫描到 {len(reader.file_list)} 个 docx 文件")
    if not reader.file_list:
        print(f"错误: 在 {data_path} 下未找到 docx 文件")
        sys.exit(1)

    chunks = reader.get_chunks()
    print(f"切分为 {len(chunks)} 个项目 chunk")

    vector = VectorStore(
        document=[chunk.text for chunk in chunks],
        metadata=[chunk.to_metadata() for chunk in chunks],
    )
    embedding = OpenAIEmbedding()
    vector.get_vector(embedding_model=embedding)
    vector.persist(path=storage_path)
    print(f"向量库已保存到 {storage_path}")
    return vector


def load_index(storage_path: str = STORAGE_PATH) -> VectorStore:
    vectors_file = os.path.join(storage_path, "vectors.json")
    document_file = os.path.join(storage_path, "document.json")
    if not os.path.exists(vectors_file) or not os.path.exists(document_file):
        print("错误: storage 不存在或不完整，请先运行: python weekly_report_rag.py --build")
        sys.exit(1)
    vector = VectorStore()
    vector.load_vector(storage_path)
    return vector


def search(
    question: str,
    storage_path: str = STORAGE_PATH,
    k: int = DEFAULT_K,
) -> List[SearchResult]:
    vector = load_index(storage_path)
    embedding = OpenAIEmbedding()
    year, month = parse_date_filter(question)
    return vector.query(
        question,
        embedding_model=embedding,
        k=k,
        year=year,
        month=month,
    )


def print_search_results(results: List[SearchResult]) -> None:
    if not results:
        print("  (未检索到满足相似度阈值的片段)")
        return
    for i, result in enumerate(results, 1):
        meta = result.metadata
        print(
            f"  [{i}] score={result.score:.3f} | "
            f"{meta.get('report_date', '?')} | {meta.get('project', '?')}"
        )
        preview = result.text.split("\n", 1)[-1][:120].replace("\n", " ")
        print(f"      {preview}...")


def ask(
    question: str,
    storage_path: str = STORAGE_PATH,
    k: int = DEFAULT_K,
    debug: bool = False,
) -> str:
    check_env()
    results = search(question, storage_path, k=k)

    if debug:
        year, month = parse_date_filter(question)
        filter_desc = f"year={year}, month={month}" if year else "无"
        print(f"检索过滤: {filter_desc}")
        print_search_results(results)

    if not results:
        return "周报中没有找到相关内容，请尝试换个问法或去掉日期限制。"

    context = "\n\n---\n\n".join(r.text for r in results)
    chat = OpenAIChat()
    return chat.chat(question, [], context)


def interactive_chat(
    storage_path: str = STORAGE_PATH,
    k: int = DEFAULT_K,
    debug: bool = False,
) -> None:
    check_env()
    chat = OpenAIChat()
    history: List[dict] = []

    print("周报 RAG 交互模式（输入 quit 退出）")
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

        results = search(question, storage_path, k=k)
        if debug:
            year, month = parse_date_filter(question)
            filter_desc = f"year={year}, month={month}" if year else "无"
            print(f"检索过滤: {filter_desc}")
            print_search_results(results)

        if not results:
            print("\n周报中没有找到相关内容。")
            continue

        context = "\n\n---\n\n".join(r.text for r in results)
        answer = chat.chat(question, history.copy(), context)
        print(f"\n{answer}")
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
