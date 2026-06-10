#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from functools import lru_cache

from config import PARSER_RULES_PATH


@lru_cache(maxsize=1)
def load_parser_rules() -> dict:
    with open(PARSER_RULES_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def heading_max_len() -> int:
    return int(load_parser_rules().get("heading_max_len", 40))


def parser_project_hints() -> list[str]:
    return list(load_parser_rules().get("parser_project_hints", []))


def section_prefixes() -> dict:
    return dict(load_parser_rules().get("section_prefixes", {}))
