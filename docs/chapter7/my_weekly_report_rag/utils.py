#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from datetime import datetime
from typing import Optional, Tuple


def parse_report_date_from_path(file_path: str) -> str:
    match = re.search(r"(\d{8})", os.path.basename(file_path))
    return normalize_report_date(match.group(1) if match else "")


def parse_report_date_from_text(text: str) -> str:
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        return normalize_report_date(
            f"{match.group(1)}{match.group(2)}{match.group(3)}"
        )
    match = re.search(r"(\d{8})", text)
    return normalize_report_date(match.group(1) if match else "")


def normalize_report_date(report_date: str) -> str:
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
    normalized = normalize_report_date(report_date)
    if len(normalized) == 8:
        return f"{normalized[:4]}-{normalized[4:6]}-{normalized[6:8]}"
    return report_date


def parse_quarter_from_path(file_path: str) -> str:
    normalized = file_path.replace("\\", "/")
    match = re.search(r"(\d{4})/Q([1-4])", normalized, re.I)
    if match:
        return f"{match.group(1)}Q{match.group(2)}"
    match = re.search(r"(\d{4})Q([1-4])", normalized, re.I)
    if match:
        return f"{match.group(1)}Q{match.group(2)}"
    return ""


def parse_author_from_path(file_path: str) -> str:
    basename = os.path.basename(file_path)
    match = re.search(r"工作周报[-_]?(.+?)\.docx$", basename)
    if match:
        return match.group(1).strip()
    return ""


def parse_date_filter(question: str) -> Tuple[Optional[int], Optional[int]]:
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
    if year is not None:
        return year, month
    if auto_date:
        return parse_date_filter(question)
    return None, None
