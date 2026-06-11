#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FastAPI Web 服务：健康检查、问答 API 与静态页面。"""

import asyncio
import logging
import socket
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.chat import ask_detail
from config import (
    INDEX_PATH,
    WEB_ACCESS_TOKEN,
    WEB_HOST,
    WEB_PORT,
    check_env,
    cleanup_tmp_dirs,
    setup_logging,
)
from exceptions import WeeklyReportRagError
from retrieval.session import RAGSession
from web.schemas import ChatRequest, ChatResponseOut, HealthOut

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
_query_lock = threading.Lock()
_API_ERROR_MESSAGE = "问答服务暂时不可用，请稍后重试。"


def _local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _verify_token(authorization: Optional[str] = Header(default=None)) -> None:
    if not WEB_ACCESS_TOKEN:
        return
    if authorization != f"Bearer {WEB_ACCESS_TOKEN}":
        raise HTTPException(status_code=401, detail="无效的访问令牌")


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging()
        cleanup_tmp_dirs()
        check_env()
        logger.info("正在加载索引: %s", INDEX_PATH)
        app.state.session = RAGSession(INDEX_PATH)
        logger.info("索引已加载，共 %s 条 chunk", app.state.session.store.record_count)
        yield

    app = FastAPI(title="周报 RAG", lifespan=lifespan)

    @app.get("/api/health", response_model=HealthOut, dependencies=[Depends(_verify_token)])
    def health(request: Request) -> HealthOut:
        session: RAGSession = request.app.state.session
        return HealthOut(status="ok", chunk_count=session.store.record_count)

    @app.post("/api/chat", response_model=ChatResponseOut, dependencies=[Depends(_verify_token)])
    async def chat(request: Request, body: ChatRequest) -> ChatResponseOut:
        session: RAGSession = request.app.state.session

        def _run():
            with _query_lock:
                return ask_detail(
                    body.message,
                    k=body.k,
                    year=body.year,
                    month=body.month,
                    auto_date=body.auto_date,
                    mode=body.mode,
                    session=session,
                )

        try:
            result = await asyncio.to_thread(_run)
        except WeeklyReportRagError:
            logger.exception("问答失败")
            raise HTTPException(status_code=500, detail=_API_ERROR_MESSAGE) from None

        return ChatResponseOut.from_response(result)

    @app.get("/")
    def index_page():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def run_server(host: str = WEB_HOST, port: int = WEB_PORT) -> None:
    import uvicorn

    lan_ip = _local_ip()
    print("\n" + "=" * 52)
    print("  周报 RAG Web 服务已启动")
    print(f"  本机访问:   http://127.0.0.1:{port}")
    if host == "0.0.0.0":
        print(f"  局域网访问: http://{lan_ip}:{port}")
    if WEB_ACCESS_TOKEN:
        print("  已启用 WEB_ACCESS_TOKEN，请在页面侧边栏填写令牌")
    elif host == "0.0.0.0":
        print("  ⚠ 未设置 WEB_ACCESS_TOKEN，局域网用户可直接访问周报内容")
        print("    建议在 .env 中配置 WEB_ACCESS_TOKEN 后再对外暴露")
    print("=" * 52 + "\n")

    uvicorn.run(
        "web.server:create_app",
        factory=True,
        host=host,
        port=port,
        log_level="info",
    )
