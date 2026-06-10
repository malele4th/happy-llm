#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from typing import List, Optional, Tuple

from docx import Document

from chunker import count_tokens, get_chunk
from config import HEADING_MAX_LEN, MAX_TOKEN_LEN, PROJECT_KEYWORDS
from utils import (
    DocumentChunk,
    format_report_date,
    parse_report_date_from_path,
    parse_report_date_from_text,
)


def is_report_title(line: str) -> bool:
    return "工作周报" in line and bool(re.search(r"\d{4}", line))


def is_heading_style(style_name: Optional[str]) -> bool:
    if not style_name:
        return False
    lower = style_name.lower()
    return (
        "heading" in lower
        or "标题" in style_name
        or style_name.startswith("Heading")
        or style_name in ("标题 1", "标题 2", "标题 3")
    )


def matches_project_keyword(line: str) -> bool:
    lower = line.lower()
    return any(kw.lower() in lower for kw in PROJECT_KEYWORDS)


def is_section_heading(
    line: str,
    next_line: Optional[str],
    style_name: Optional[str] = None,
) -> bool:
    line = line.strip()
    if not line or line.startswith("http"):
        return False
    if is_report_title(line):
        return False
    if is_heading_style(style_name):
        return True
    if re.match(r"^【.+】$", line):
        return True
    if matches_project_keyword(line) and len(line) <= HEADING_MAX_LEN:
        return True
    if len(line) > HEADING_MAX_LEN:
        return False
    if line.endswith(("。", "；", "，", "！", "？", "）", ")", ":", "：")) and len(line) > 15:
        return False
    if re.match(r"^\d{1,2}月\d{1,2}日", line):
        return False
    if re.match(r"^(背景|策略|进展|下周计划|总结|summary)", line, re.I):
        return False
    if len(line) <= HEADING_MAX_LEN:
        if next_line is None:
            return len(line) <= 20
        if len(next_line) <= HEADING_MAX_LEN and not next_line.endswith("。"):
            return True
        if len(next_line) > len(line) + 10 or next_line.endswith("。"):
            return True
    return False


def build_chunk_text(
    source: str,
    report_date: str,
    project: str,
    body_lines: List[str],
) -> str:
    header = (
        f"[来源: {source} | 日期: {format_report_date(report_date)} | 项目: {project}]"
    )
    body = "\n".join([project] + body_lines) if body_lines else project
    return f"{header}\n{body}"


def split_docx_into_sections(file_path: str) -> List[Tuple[str, List[str]]]:
    doc = Document(file_path)
    paragraphs: List[Tuple[str, Optional[str]]] = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style_name = paragraph.style.name if paragraph.style else None
        paragraphs.append((text, style_name))

    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
            if row_text:
                paragraphs.append((row_text, None))

    sections: List[Tuple[str, List[str]]] = []
    current_heading = "综合"
    current_lines: List[str] = []

    for i, (line, style_name) in enumerate(paragraphs):
        if is_report_title(line):
            continue
        next_line = paragraphs[i + 1][0] if i + 1 < len(paragraphs) else None
        if is_section_heading(line, next_line, style_name):
            if current_lines or current_heading != "综合":
                sections.append((current_heading, current_lines))
            current_heading = line
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_lines))

    if not sections and paragraphs:
        sections.append(("综合", [p[0] for p in paragraphs]))

    return sections


def sections_to_chunks(
    rel_path: str,
    report_date: str,
    sections: List[Tuple[str, List[str]]],
) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    for project, body_lines in sections:
        if not body_lines:
            continue
        section_text = build_chunk_text(rel_path, report_date, project, body_lines)
        if count_tokens(section_text) <= MAX_TOKEN_LEN:
            chunks.append(DocumentChunk(section_text, rel_path, report_date, project))
        else:
            for part in get_chunk(section_text):
                chunks.append(DocumentChunk(part, rel_path, report_date, project))
    return chunks


class ReadFiles:
    def __init__(self, path: str) -> None:
        self._path = os.path.abspath(path)
        self.file_list = self.get_files()

    def get_files(self) -> List[str]:
        file_list = []
        for filepath, _, filenames in os.walk(self._path):
            for filename in filenames:
                if filename.startswith("~$") or filename == "tmp.docx":
                    continue
                if filename.endswith(".docx"):
                    file_list.append(os.path.join(filepath, filename))
        return sorted(file_list)

    def _resolve_report_date(
        self,
        file_path: str,
        sections: List[Tuple[str, List[str]]],
    ) -> str:
        report_date = parse_report_date_from_path(file_path)
        if not report_date and sections:
            first_line = sections[0][1][0] if sections[0][1] else sections[0][0]
            report_date = parse_report_date_from_text(first_line)
        return report_date

    def get_chunks_for_file(self, file_path: str) -> List[DocumentChunk]:
        rel_path = os.path.relpath(file_path, self._path)
        sections = split_docx_into_sections(file_path)
        report_date = self._resolve_report_date(file_path, sections)
        return sections_to_chunks(rel_path, report_date, sections)

    def get_chunks(self) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        for file_path in self.file_list:
            chunks.extend(self.get_chunks_for_file(file_path))
        return chunks
