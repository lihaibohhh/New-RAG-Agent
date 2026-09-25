"""
health_tools.py:
    注册 check_knowledge_base
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any
from mcp.server.fastmcp import FastMCP
from knowledge.runtime_ports import RagAdminServicePort
from mcp_service.observability import ToolCallTrace
from mcp_service.responses import mcp_err, mcp_ok


def register_health_tools(
    server: FastMCP,
    *,
    service_provider: Callable[[], RagAdminServicePort],
) -> None:
    """
    注册知识库健康检查工具。

    这个工具只负责检查当前 RAG 知识库是否能被访问，
    不执行实际检索。
    """

    @server.tool(description="轻量检查 Chroma 向量库是否可访问，并返回向量库中的文档块数量；不加载嵌入模型或精排器，不执行真实检索")
    async def check_knowledge_base() -> dict[str, Any]:
        trace = ToolCallTrace.start("check_knowledge_base")
        try:
            health = await service_provider().health()
            knowledge_base = dict(health.details.get("knowledge_base") or {})

            return mcp_ok(
                data=knowledge_base,
                meta={
                    **trace.finish(stage="health_check", status="ok"),
                    "mode": "lightweight",
                },
            )

        except Exception as exc:
            meta = trace.finish(
                stage="health_check",
                status="error",
                error_type=type(exc).__name__,
            )
            meta["mode"] = "lightweight"
            return mcp_err(
                f"{type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
                meta=meta,
            )
