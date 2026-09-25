"""
warmup_tools.py

RAG 预热操作的 MCP 工具适配层。

实际预热运行时放在 RAG runtime 中，查询工具可以复用同一份
就绪状态，而不需要依赖带 MCP 装饰器的函数。
"""
from __future__ import annotations

from typing import Any
from mcp.server.fastmcp import FastMCP

from knowledge.runtime_ports import RagOperationsPort
from mcp_service.observability import ToolCallTrace
from mcp_service.responses import mcp_err, mcp_ok


_START_WARMUP_STATUS_BY_STAGE = {
    "already_running": "already_running",
    "already_done": "already_done",
    "started": "background_singleton_warmup_started",
}

_START_WARMUP_MESSAGE_BY_STAGE = {
    "already_running": "RAG 单例预热正在运行。",
    "already_done": "RAG 单例预热已完成。如需重新执行，请设置 force=true。",
    "started": (
        "RAG 单例预热已在后台启动。"
        "可调用 get_rag_singleton_warmup_status 查看进度。"
    ),
}


def _build_start_warmup_data(result: dict[str, Any]) -> tuple[dict[str, Any], str]:
    stage = str(result.get("stage") or "started")
    data: dict[str, Any] = {
        "status": _START_WARMUP_STATUS_BY_STAGE.get(stage, stage),
        "message": _START_WARMUP_MESSAGE_BY_STAGE.get(stage, "RAG 单例预热状态已更新。"),
    }

    if "task_state" in result:
        data["task_state"] = result["task_state"]
    if "warmup_status" in result:
        data["warmup_status"] = result["warmup_status"]

    return data, stage


def register_warmup_tools(
    server: FastMCP,
    *,
    manager: RagOperationsPort,
) -> None:
    @server.tool(
        description=(
            "后台触发 RAG 单例预热，立即返回。"
            "只加载 retriever 与 reranker 两个进程内单例；"
            "不会写入语义缓存，也不直接返回检索结果。"
            "若已完成，默认不重复执行；传入 force=true 可重跑。"
            "普通业务检索不应依赖手动调用此工具；它主要用于诊断、显式预热和冷启动排障。"
        )
    )
    async def start_rag_singleton_warmup(force: bool = False) -> dict[str, Any]:
        trace = ToolCallTrace.start("start_rag_singleton_warmup")
        try:
            result = manager.start(force=force)
        except Exception as exc:
            return mcp_err(
                f"{type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
                meta=trace.finish(
                    stage="start_exception",
                    status="error",
                    error_type=type(exc).__name__,
                ),
            )
        data, stage = _build_start_warmup_data(result)

        return mcp_ok(
            data=data,
            meta=trace.finish(stage=stage, status="ok"),
        )

    @server.tool(
        description=(
            "查看后台 RAG 单例预热的当前状态；以 warmup_status.state 判断 "
            "not_started/running/done/error/cancelled，而不只看 task_state。"
            "普通业务检索不应依赖手动轮询此工具；它主要用于诊断和排障。"
        )
    )
    async def get_rag_singleton_warmup_status() -> dict[str, Any]:
        trace = ToolCallTrace.start("get_rag_singleton_warmup_status")
        try:
            data = manager.get_status()
        except Exception as exc:
            return mcp_err(
                f"{type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
                meta=trace.finish(
                    stage="status_exception",
                    status="error",
                    error_type=type(exc).__name__,
                ),
            )
        task_state = data.get("task_state", "none")

        return mcp_ok(
            data=data,
            meta=trace.finish(stage=str(task_state), status="ok"),
        )
