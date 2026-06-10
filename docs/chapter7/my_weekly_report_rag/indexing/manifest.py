#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib
import json
import os

from config import COVER_CONTENT, MANIFEST_FILE, MAX_TOKEN_LEN, PARSER_RULES_PATH
from parsing.reader import ReadFiles
from parsing.rules import heading_max_len, load_parser_rules


def compute_index_version() -> str:
    with open(PARSER_RULES_PATH, encoding="utf-8") as handle:
        rules_hash = hashlib.md5(handle.read().encode()).hexdigest()[:8]
    payload = "|".join([
        str(MAX_TOKEN_LEN),
        str(COVER_CONTENT),
        str(heading_max_len()),
        rules_hash,
    ])
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def file_hash(file_path: str) -> str:
    digest = hashlib.md5()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(storage_path: str) -> str:
    return os.path.join(storage_path, MANIFEST_FILE)


def load_manifest(storage_path: str) -> dict:
    path = manifest_path(storage_path)
    if not os.path.exists(path):
        return {"files": {}, "index_version": ""}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(storage_path: str, reader: ReadFiles) -> None:
    manifest = {
        "index_version": compute_index_version(),
        "parser_rules": load_parser_rules(),
        "files": {
            os.path.relpath(file_path, reader.data_path): {"hash": file_hash(file_path)}
            for file_path in reader.file_list
        },
    }
    with open(manifest_path(storage_path), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
