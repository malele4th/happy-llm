#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OpenAI 兼容 API 客户端单例。"""

import os
from typing import Optional

from openai import OpenAI

_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    return _client
