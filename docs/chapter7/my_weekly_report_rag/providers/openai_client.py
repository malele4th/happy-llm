#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Optional

from openai import OpenAI

_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        client = OpenAI()
        client.api_key = os.getenv("OPENAI_API_KEY")
        client.base_url = os.getenv("OPENAI_BASE_URL")
        _client = client
    return _client
