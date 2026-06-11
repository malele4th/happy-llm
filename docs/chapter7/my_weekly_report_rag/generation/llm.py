#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM 问答生成（OpenAI 兼容 API）。"""

from config import CHAT_MODEL
from exceptions import ApiRequestError
from providers.openai_client import get_openai_client

EMPTY_CONTEXT = "（未检索到相关周报片段）"

WEEKLY_REPORT_PROMPT = """你是 malele 周报助手，主要根据用户的工作周报回答问题。

请按以下规则用中文回答：

1. **周报相关问题**（询问项目进展、工作内容、数据等），且下方有可用片段：
   基于片段作答，引用具体日期/项目/数据，句末标注来源编号如 [1][2]。

2. **非周报问题**（如你是谁、你能做什么、闲聊、通用知识等与个人周报无关的问题）：
   可以正常简短回答；必须在回答开头单独一行写「【非周报内容】」，不要编造周报信息，不要标注 [1][2]。

3. **看似周报问题但片段不足**（明确在问某项工作/项目/月份进展，但下方无相关片段或信息不够）：
   说明周报中没有找到相关内容，并建议换问法或指定年月；不要标注「【非周报内容】」，不要编造。

问题: {question}
可参考的周报内容：
···
{context}
···
回答:"""


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
