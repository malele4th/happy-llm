#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import List

import tiktoken
from docx import Document

from config import COVER_CONTENT, MAX_TOKEN_LEN

enc = tiktoken.get_encoding("cl100k_base")


def get_token_cover(text: str, cover_tokens: int) -> str:
    tokens = enc.encode(text)
    if len(tokens) <= cover_tokens:
        return text
    return enc.decode(tokens[-cover_tokens:])


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

    def get_content(
        self,
        max_token_len: int = MAX_TOKEN_LEN,
        cover_content: int = COVER_CONTENT,
    ) -> List[str]:
        docs = []
        for file in self.file_list:
            content = self.read_file_content(file)
            if not content.strip():
                continue
            rel_path = os.path.relpath(file, self._path)
            chunks = self.get_chunk(content, max_token_len=max_token_len, cover_content=cover_content)
            for chunk in chunks:
                docs.append(f"[来源: {rel_path}]\n{chunk}")
        return docs

    @classmethod
    def get_chunk(
        cls,
        text: str,
        max_token_len: int = MAX_TOKEN_LEN,
        cover_content: int = COVER_CONTENT,
    ) -> List[str]:
        chunk_text = []
        curr_len = 0
        curr_chunk = ""
        token_len = max_token_len - cover_content
        lines = text.splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            line_len = len(enc.encode(line))

            if line_len > max_token_len:
                if curr_chunk:
                    chunk_text.append(curr_chunk)
                    curr_chunk = ""
                    curr_len = 0

                line_tokens = enc.encode(line)
                num_chunks = (len(line_tokens) + token_len - 1) // token_len

                for i in range(num_chunks):
                    start_token = i * token_len
                    end_token = min(start_token + token_len, len(line_tokens))
                    chunk_part = enc.decode(line_tokens[start_token:end_token])

                    if i > 0:
                        overlap_start = max(0, start_token - cover_content)
                        cover_part = enc.decode(line_tokens[overlap_start:start_token])
                        chunk_part = cover_part + chunk_part

                    chunk_text.append(chunk_part)

                curr_chunk = ""
                curr_len = 0

            elif curr_len + line_len + 1 <= token_len:
                if curr_chunk:
                    curr_chunk += "\n"
                    curr_len += 1
                curr_chunk += line
                curr_len += line_len
            else:
                if curr_chunk:
                    chunk_text.append(curr_chunk)
                if chunk_text:
                    cover_part = get_token_cover(chunk_text[-1], cover_content)
                    curr_chunk = cover_part + "\n" + line
                    curr_len = len(enc.encode(cover_part)) + 1 + line_len
                else:
                    curr_chunk = line
                    curr_len = line_len

        if curr_chunk:
            chunk_text.append(curr_chunk)

        return chunk_text

    @classmethod
    def read_file_content(cls, file_path: str) -> str:
        if file_path.endswith(".docx"):
            return cls.read_docx(file_path)
        raise ValueError(f"Unsupported file type: {file_path}")

    @classmethod
    def read_docx(cls, file_path: str) -> str:
        doc = Document(file_path)
        lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                if row_text:
                    lines.append(row_text)
        return "\n".join(lines)
