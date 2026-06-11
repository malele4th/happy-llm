#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""日期解析、路径工具与检索过滤辅助函数。"""

import os
import re
from datetime import datetime
from typing import Optional, Tuple


def parse_report_date_from_path(file_path: str) -> str:
    """从文件名中提取 8 位日期，如 工作周报_20251201.docx。"""
    match = re.search(r"(\d{8})", os.path.basename(file_path))
    return normalize_report_date(match.group(1) if match else "")


def parse_report_date_from_text(text: str) -> str:
    """从正文首行等文本中解析日期（支持 YYYY-MM-DD 或连续 8 位）。"""
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        return normalize_report_date(
            f"{match.group(1)}{match.group(2)}{match.group(3)}"
        )
    match = re.search(r"(\d{8})", text)
    return normalize_report_date(match.group(1) if match else "")


def normalize_report_date(report_date: str) -> str:
    """将各种日期格式统一为 YYYYMMDD，无效则返回空串。"""
    if not report_date:
        return ""
    digits = re.sub(r"\D", "", report_date)
    if len(digits) != 8:
        return ""
    try:
        datetime.strptime(digits, "%Y%m%d")
    except ValueError:
        return ""
    return digits


def format_report_date(report_date: str) -> str:
    """将 YYYYMMDD 格式化为 YYYY-MM-DD 供展示。"""
    normalized = normalize_report_date(report_date)
    if len(normalized) == 8:
        return f"{normalized[:4]}-{normalized[4:6]}-{normalized[6:8]}"
    return report_date


def parse_quarter_from_path(file_path: str) -> str:
    """从目录路径解析季度，如 2025/Q3 → 2025Q3。"""
    normalized = file_path.replace("\\", "/")
    for pattern in (r"(\d{4})/Q([1-4])", r"(\d{4})Q([1-4])"):
        match = re.search(pattern, normalized, re.I)
        if match:
            return f"{match.group(1)}Q{match.group(2)}"
    return ""


def parse_author_from_path(file_path: str) -> str:
    """从文件名解析作者，如 工作周报-张三.docx。"""
    basename = os.path.basename(file_path)
    match = re.search(r"工作周报[-_]?(.+?)\.docx$", basename)
    return match.group(1).strip() if match else ""


def parse_date_filter(question: str) -> Tuple[Optional[int], Optional[int]]:
    """从问题文本解析年月，如「2025年12月」。"""
    match = re.search(r"(\d{4})年(\d{1,2})月", question)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"(\d{4})年", question)
    if match:
        return int(match.group(1)), None
    match = re.search(r"(\d{8})", question)
    if match:
        normalized = normalize_report_date(match.group(1))
        if normalized:
            return int(normalized[:4]), int(normalized[4:6])
    return None, None


def resolve_date_filter(
    question: str,
    year: Optional[int],
    month: Optional[int],
    auto_date: bool,
) -> Tuple[Optional[int], Optional[int]]:
    """CLI/Web 统一的日期过滤解析：显式参数优先，其次自动解析。"""
    if year is not None:
        return year, month
    if auto_date:
        return parse_date_filter(question)
    return None, None


def format_filter_desc(year: Optional[int], month: Optional[int]) -> str:
    """调试输出用的过滤条件描述。"""
    if year is None:
        return "无"
    return f"year={year}, month={month}"
