#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""解析规则加载：从 parser_rules.json 读取标题识别等配置。"""

import json
from functools import lru_cache

from config import PARSER_RULES_PATH


@lru_cache(maxsize=1)
def load_parser_rules() -> dict:
    """加载并缓存 parser_rules.json。"""
    with open(PARSER_RULES_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def heading_max_len() -> int:
    """标题行最大字符数阈值。"""
    return int(load_parser_rules().get("heading_max_len", 40))


def parser_project_hints() -> list[str]:
    """已知项目名关键词列表，辅助识别章节标题。"""
    return list(load_parser_rules().get("parser_project_hints", []))


def section_prefixes() -> dict:
    """章节类型 → 段落前缀映射，用于 infer_section_type。"""
    return dict(load_parser_rules().get("section_prefixes", {}))
