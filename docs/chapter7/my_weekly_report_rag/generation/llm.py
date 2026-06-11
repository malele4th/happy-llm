#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM 问答生成（OpenAI 兼容 API）。"""

from config import CHAT_MODEL
from exceptions import ApiRequestError
from providers.openai_client import get_openai_client

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
        self.client = get_openai_client()

    def chat(self, prompt: str, content: str) -> str:
        messages = [{
            "role": "user",
            "content": WEEKLY_REPORT_PROMPT.format(question=prompt, context=content),
        }]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.1,
            )
        except Exception as exc:
            raise ApiRequestError(f"LLM 请求失败: {exc}") from exc

        message = response.choices[0].message.content
        if not message:
            raise ApiRequestError("LLM 返回空内容")
        return message
