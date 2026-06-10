#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from typing import List, Optional, Tuple


def parse_report_date_from_path(file_path: str) -> str:
    match = re.search(r"(\d{8})", os.path.basename(file_path))
    return match.group(1) if match else ""


def parse_report_date_from_text(text: str) -> str:
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}"
    match = re.search(r"(\d{8})", text)
    return match.group(1) if match else ""


def format_report_date(report_date: str) -> str:
    if len(report_date) == 8:
        return f"{report_date[:4]}-{report_date[4:6]}-{report_date[6:8]}"
    return report_date


def parse_date_filter(question: str) -> Tuple[Optional[int], Optional[int]]:
    """从问题文本解析年月，仅用于显式开启自动日期过滤时。"""
    match = re.search(r"(\d{4})年(\d{1,2})月", question)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"(\d{4})年", question)
    if match:
        return int(match.group(1)), None
    match = re.search(r"(\d{8})", question)
    if match:
        d = match.group(1)
        return int(d[:4]), int(d[4:6])
    return None, None


def extract_query_tokens(query: str) -> List[str]:
    tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", query.lower())
    return [token for token in tokens if len(token) >= 2]
