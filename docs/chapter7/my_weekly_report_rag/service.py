#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib
import json
import os
import shutil
from typing import List, Optional, Tuple

from config import (
    DEFAULT_K,
    DEFAULT_SEARCH_MODE,
    MANIFEST_FILE,
    REPORT_DATA_PATH,
    STORAGE_PATH,
)
from Embeddings import OpenAIEmbedding
from exceptions import EnvConfigError, NoDataError, StorageNotFoundError
from LLM import OpenAIChat
from parser import ReadFiles
from utils import format_report_date, parse_date_filter
from VectorBase import SearchMode, SearchResult, VectorStore


def check_env() -> None:
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        raise EnvConfigError("请在 .env 中配置 OPENAI_API_KEY 和 OPENAI_BASE_URL")


def _file_hash(file_path: str) -> str:
    digest = hashlib.md5()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(storage_path: str) -> str:
    return os.path.join(storage_path, MANIFEST_FILE)


def _load_manifest(storage_path: str) -> dict:
    path = _manifest_path(storage_path)
    if not os.path.exists(path):
        return {"files": {}}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_manifest(storage_path: str, reader: ReadFiles) -> None:
    manifest = {
        "files": {
            os.path.relpath(file_path, reader._path): {"hash": _file_hash(file_path)}
            for file_path in reader.file_list
        }
    }
    with open(_manifest_path(storage_path), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def load_index(storage_path: str = STORAGE_PATH) -> VectorStore:
    vectors_file = os.path.join(storage_path, "vectors.json")
    document_file = os.path.join(storage_path, "document.json")
    if not os.path.exists(vectors_file) or not os.path.exists(document_file):
        raise StorageNotFoundError(
            "storage 不存在或不完整，请先运行: python weekly_report_rag.py --build"
        )
    vector = VectorStore()
    vector.load_vector(storage_path)
    return vector


class RAGSession:
    """复用向量库与 embedding 客户端，避免交互模式下重复加载。"""

    def __init__(self, storage_path: str = STORAGE_PATH) -> None:
        check_env()
        self.storage_path = storage_path
        self.vector = load_index(storage_path)
        self.embedding = OpenAIEmbedding()
        self.chat = OpenAIChat()

    def search(
        self,
        question: str,
        k: int = DEFAULT_K,
        year: Optional[int] = None,
        month: Optional[int] = None,
        mode: SearchMode = DEFAULT_SEARCH_MODE,
    ) -> List[SearchResult]:
        return self.vector.query(
            question,
            embedding_model=self.embedding,
            k=k,
            year=year,
            month=month,
            mode=mode,
        )


def resolve_date_filter(
    question: str,
    year: Optional[int],
    month: Optional[int],
    auto_date: bool,
) -> Tuple[Optional[int], Optional[int]]:
    if year is not None:
        return year, month
    if auto_date:
        return parse_date_filter(question)
    return None, None


def _incremental_build(reader: ReadFiles, storage_path: str) -> VectorStore:
    vector = load_index(storage_path)
    manifest = _load_manifest(storage_path)
    old_files = manifest.get("files", {})

    current_files = {
        os.path.relpath(file_path, reader._path): _file_hash(file_path)
        for file_path in reader.file_list
    }

    sources_to_remove = {
        rel for rel in old_files if rel not in current_files
    }
    sources_to_update = {
        rel
        for rel, file_hash in current_files.items()
        if rel not in old_files or old_files[rel].get("hash") != file_hash
    }
    sources_to_remove |= sources_to_update

    kept_docs: List[str] = []
    kept_metadata: List[dict] = []
    kept_vectors: List[List[float]] = []

    for index, doc in enumerate(vector.document):
        meta = vector.metadata[index]
        source = meta.get("source", "")
        if source in sources_to_remove:
            continue
        kept_docs.append(doc)
        kept_metadata.append(meta)
        kept_vectors.append(vector.vectors[index])

    new_chunks = []
    for file_path in reader.file_list:
        rel_path = os.path.relpath(file_path, reader._path)
        if rel_path in sources_to_update:
            new_chunks.extend(reader.get_chunks_for_file(file_path))

    if new_chunks:
        embedding = OpenAIEmbedding()
        new_embeddings = embedding.get_embeddings([chunk.text for chunk in new_chunks])
        for chunk, embedding_vector in zip(new_chunks, new_embeddings):
            kept_docs.append(chunk.text)
            kept_metadata.append(chunk.to_metadata().to_dict())
            kept_vectors.append(embedding_vector)

    vector.document = kept_docs
    vector.metadata = kept_metadata
    vector.vectors = kept_vectors
    vector.persist(path=storage_path)
    _save_manifest(storage_path, reader)

    print(
        f"增量更新: 移除/更新 {len(sources_to_remove)} 个文件, "
        f"新增 {len(new_chunks)} 个 chunk, 当前共 {len(kept_docs)} 个 chunk"
    )
    return vector


def build_index(
    data_path: str = REPORT_DATA_PATH,
    storage_path: str = STORAGE_PATH,
    force: bool = False,
) -> VectorStore:
    check_env()

    if os.path.exists(storage_path) and not force:
        reader = ReadFiles(data_path)
        if not reader.file_list:
            raise NoDataError(f"在 {data_path} 下未找到 docx 文件")
        return _incremental_build(reader, storage_path)

    if force and os.path.exists(storage_path):
        shutil.rmtree(storage_path)

    reader = ReadFiles(data_path)
    print(f"扫描到 {len(reader.file_list)} 个 docx 文件")
    if not reader.file_list:
        raise NoDataError(f"在 {data_path} 下未找到 docx 文件")

    chunks = reader.get_chunks()
    print(f"切分为 {len(chunks)} 个项目 chunk")

    vector = VectorStore(
        document=[chunk.text for chunk in chunks],
        metadata=[chunk.to_metadata().to_dict() for chunk in chunks],
    )
    embedding = OpenAIEmbedding()
    vector.get_vector(embedding_model=embedding)
    vector.persist(path=storage_path)
    _save_manifest(storage_path, reader)
    print(f"向量库已保存到 {storage_path}")
    return vector


def search(
    question: str,
    storage_path: str = STORAGE_PATH,
    k: int = DEFAULT_K,
    year: Optional[int] = None,
    month: Optional[int] = None,
    auto_date: bool = False,
    mode: SearchMode = DEFAULT_SEARCH_MODE,
    session: Optional[RAGSession] = None,
) -> List[SearchResult]:
    filter_year, filter_month = resolve_date_filter(question, year, month, auto_date)
    if session is not None:
        return session.search(
            question, k=k, year=filter_year, month=filter_month, mode=mode
        )

    vector = load_index(storage_path)
    embedding = OpenAIEmbedding()
    return vector.query(
        question,
        embedding_model=embedding,
        k=k,
        year=filter_year,
        month=filter_month,
        mode=mode,
    )


def build_numbered_context(results: List[SearchResult]) -> str:
    return "\n\n---\n\n".join(
        f"[{index}]\n{result.text}" for index, result in enumerate(results, 1)
    )


def format_citations(results: List[SearchResult]) -> str:
    lines = ["【引用】"]
    for index, result in enumerate(results, 1):
        meta = result.metadata
        date = format_report_date(meta.get("report_date", "")) or "?"
        project = meta.get("project", "?")
        lines.append(f"[{index}] {date} | {project} | score={result.score:.3f}")
    return "\n".join(lines)


def format_answer_with_citations(answer: str, results: List[SearchResult]) -> str:
    return f"【回答】\n{answer}\n\n{format_citations(results)}"


def print_search_results(results: List[SearchResult], verbose: bool = False) -> None:
    if not results:
        print("  (未检索到满足相似度阈值的片段)")
        return
    for index, result in enumerate(results, 1):
        meta = result.metadata
        date = format_report_date(meta.get("report_date", "")) or "?"
        print(
            f"  [{index}] score={result.score:.3f} | "
            f"{date} | {meta.get('project', '?')}"
        )
        body = result.text.split("\n", 1)[-1]
        if verbose:
            print(f"      {body}")
        else:
            preview = body[:120].replace("\n", " ")
            print(f"      {preview}...")


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
    check_env()
    active_session = session or RAGSession(storage_path)

    filter_year, filter_month = resolve_date_filter(
        question, year, month, auto_date
    )
    results = active_session.search(
        question,
        k=k,
        year=filter_year,
        month=filter_month,
        mode=mode,
    )

    if debug:
        filter_desc = (
            f"year={filter_year}, month={filter_month}"
            if filter_year is not None
            else "无"
        )
        print(f"检索模式: {mode} | 过滤: {filter_desc}")
        print_search_results(results)

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
            storage_path=storage_path,
            k=k,
            debug=debug,
            year=year,
            month=month,
            auto_date=auto_date,
            mode=mode,
            session=session,
        )
        print(f"\n{answer}")
