#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import List, Tuple

from models import DocumentChunk
from parsing.docx import sections_to_chunks, split_docx_into_sections
from utils import parse_report_date_from_path, parse_report_date_from_text


class DocxReportReader:
    """扫描周报目录并解析 docx 为 DocumentChunk。"""

    def __init__(self, path: str) -> None:
        self.data_path = os.path.abspath(path)
        self.file_list = self._scan_files()

    def _scan_files(self) -> List[str]:
        file_list = []
        for filepath, _, filenames in os.walk(self.data_path):
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
        rel_path = os.path.relpath(file_path, self.data_path)
        sections = split_docx_into_sections(file_path)
        report_date = self._resolve_report_date(file_path, sections)
        return sections_to_chunks(file_path, rel_path, report_date, sections)

    def get_chunks(self) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        for file_path in self.file_list:
            chunks.extend(self.get_chunks_for_file(file_path))
        return chunks
