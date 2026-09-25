"""MCP Server 与远程 Knowledge Runtime 的进程级生命周期。"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from knowledge.client import create_configured_rag_runtime
from knowledge.runtime_ports import RagRuntimePort
from mcp_service.app import create_mcp_server


logger = logging.getLogger(__name__)


class StdioServerPort(Protocol):
    def run(self, transport: str = "stdio") -> None: ...


@dataclass
class McpServiceApplication:
    """显式持有 Server 与 Runtime，并保证 stdio 退出后释放客户端。"""

    server: StdioServerPort
    rag_runtime: RagRuntimePort

    def run_stdio(self) -> None:
        try:
            self.server.run(transport="stdio")
        finally:
            self.close()

    def close(self) -> None:
        try:
            asyncio.run(self.rag_runtime.close())
        except Exception:
            logger.exception("[RAG-MCP] 关闭 Knowledge Runtime 失败")


def create_configured_mcp_application() -> McpServiceApplication:
    """按环境配置组装 MCP 进程；导入模块本身不会创建 Runtime。"""
    rag_runtime = create_configured_rag_runtime()
    try:
        server = create_mcp_server(rag_runtime=rag_runtime)
    except BaseException:
        asyncio.run(rag_runtime.close())
        raise
    return McpServiceApplication(server=server, rag_runtime=rag_runtime)


__all__ = ["McpServiceApplication", "create_configured_mcp_application"]
