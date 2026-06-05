#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import sys
from typing import List

from config import DEFAULT_K, REPORT_DATA_PATH, STORAGE_PATH
from Embeddings import OpenAIEmbedding
from LLM import OpenAIChat
from utils import ReadFiles
from VectorBase import VectorStore


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

    docs = reader.get_content()
    print(f"切分为 {len(docs)} 个 chunk")

    vector = VectorStore(docs)
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


def ask(question: str, storage_path: str = STORAGE_PATH, k: int = DEFAULT_K) -> str:
    check_env()
    vector = load_index(storage_path)
    embedding = OpenAIEmbedding()
    contexts = vector.query(question, embedding_model=embedding, k=k)
    context = "\n\n---\n\n".join(contexts)
    chat = OpenAIChat()
    return chat.chat(question, [], context)


def interactive_chat(storage_path: str = STORAGE_PATH, k: int = DEFAULT_K) -> None:
    check_env()
    vector = load_index(storage_path)
    embedding = OpenAIEmbedding()
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

        contexts = vector.query(question, embedding_model=embedding, k=k)
        context = "\n\n---\n\n".join(contexts)
        answer = chat.chat(question, history.copy(), context)
        print(f"\n{answer}")
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
