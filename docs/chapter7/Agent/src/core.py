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
        self._tool_registry = {tool.__name__: tool for tool in self.tools}
        self.model = model
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.verbose = verbose
        self._tool_schema = [function_to_json(tool) for tool in self.tools]

    def get_tool_schema(self) -> list[dict[str, Any]]:
        return self._tool_schema

    def handle_tool_call(self, tool_call) -> dict[str, str]:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        function_id = tool_call.id

        if function_name not in self._tool_registry:
            raise ValueError(f"未知工具: {function_name}")

        result = self._tool_registry[function_name](**function_args)

        return {
            "role": "tool",
            "content": str(result),
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt: str) -> str:
        self.messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self._tool_schema,
            stream=False,
        )

        if response.choices[0].message.tool_calls:
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
