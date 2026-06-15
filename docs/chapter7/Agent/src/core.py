"""Agent 核心：维护对话历史，编排 LLM 与工具调用。"""

import json
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from src.utils import function_to_json

SYSTEM_PROMPT = """
你是一个叫小新的人工智能助手。你的输出应该与用户的语言保持一致。
当用户的问题需要调用工具时，你可以从提供的工具列表中调用适当的工具函数。
"""


class Agent:
    def __init__(
        self,
        client: OpenAI,
        model: str = "Qwen/Qwen2.5-32B-Instruct",
        tools: list[Callable[..., Any]] | None = None,
        verbose: bool = True,
    ):
        self.client = client
        self.tools = list(tools or [])
        # 函数名 -> 可调用对象，用于安全分发工具调用（避免 eval）
        self._tool_registry = {tool.__name__: tool for tool in self.tools}
        self.model = model
        # 发给 LLM 的完整对话历史（含 system / user / assistant / tool）
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.verbose = verbose
        # 工具 JSON Schema，初始化时生成一次，传给 LLM 的 tools 参数
        self._tool_schema = [function_to_json(tool) for tool in self.tools]

    def get_tool_schema(self) -> list[dict[str, Any]]:
        return self._tool_schema

    def handle_tool_call(self, tool_call) -> dict[str, str]:
        """执行单次工具调用，返回符合 OpenAI 格式的 tool 消息。"""
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        function_id = tool_call.id

        if function_name not in self._tool_registry:
            raise ValueError(f"未知工具: {function_name}")

        result = self._tool_registry[function_name](**function_args)

        return {
            "role": "tool",
            "content": str(result),  # API 要求 content 为字符串
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt: str) -> str:
        """处理一轮用户输入，必要时调用工具后返回最终文本。"""
        self.messages.append({"role": "user", "content": prompt})

        # 第一次请求：LLM 决定直接回答或发起 tool_calls
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self._tool_schema,
            stream=False,
        )

        if response.choices[0].message.tool_calls:
            # 须先写入带 tool_calls 的 assistant 消息，再追加各 tool 结果
            assistant_message = {
                "role": "assistant",
                "content": response.choices[0].message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in response.choices[0].message.tool_calls
                ],
            }
            self.messages.append(assistant_message)

            tool_list = []
            for tool_call in response.choices[0].message.tool_calls:
                self.messages.append(self.handle_tool_call(tool_call))
                tool_list.append([tool_call.function.name, tool_call.function.arguments])

            if self.verbose:
                print("调用工具：", response.choices[0].message.content, tool_list)

            # 第二次请求：LLM 根据工具结果生成最终回答
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self._tool_schema,
                stream=False,
            )

        self.messages.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
        return response.choices[0].message.content
