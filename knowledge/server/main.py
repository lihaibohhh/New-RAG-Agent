"""`uvicorn knowledge.server.main:app` 启动入口。"""
from __future__ import annotations

from dotenv import load_dotenv

from knowledge.server.app import create_app


load_dotenv()
app = create_app()


__all__ = ["app"]
