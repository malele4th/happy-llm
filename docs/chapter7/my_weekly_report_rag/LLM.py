#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import List

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from config import CHAT_MODEL

_ = load_dotenv(find_dotenv())

WEEKLY_REPORT_PROMPT = """
你是工作周报助手。根据以下周报片段回答用户问题。
要求：用中文回答；引用具体日期/项目/数据，并在句末标注来源编号如 [1][2]；上下文不足时说"周报中没有相关内容"。

问题: {question}
可参考的周报内容：
···
{context}
···
回答:
"""


class OpenAIChat:
    def __init__(self, model: str = CHAT_MODEL) -> None:
        self.model = model
        self.client = OpenAI()
        self.client.api_key = os.getenv("OPENAI_API_KEY")
        self.client.base_url = os.getenv("OPENAI_BASE_URL")

    def chat(self, prompt: str, content: str) -> str:
        messages = [{
            "role": "user",
            "content": WEEKLY_REPORT_PROMPT.format(question=prompt, context=content),
        }]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
            temperature=0.1,
        )
        return response.choices[0].message.content
