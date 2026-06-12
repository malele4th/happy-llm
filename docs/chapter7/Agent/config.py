"""Agent 公共配置与初始化。"""

import os

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from src.core import Agent
from src.tools import get_current_datetime, get_current_temperature, search_wikipedia

load_dotenv(find_dotenv())

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen2.5-32B-Instruct")

DEFAULT_TOOLS = [get_current_datetime, search_wikipedia, get_current_temperature]


def create_client() -> OpenAI:
    if not API_KEY:
        raise ValueError("请设置环境变量 OPENAI_API_KEY（可在 .env 文件中配置）")
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def create_agent(client: OpenAI | None = None, verbose: bool = True) -> Agent:
    return Agent(
        client=client or create_client(),
        model=MODEL,
        tools=DEFAULT_TOOLS,
        verbose=verbose,
    )
