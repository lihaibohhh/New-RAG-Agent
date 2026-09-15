"""所有 Agent Tool 共用的稳定结果信封。"""
from __future__ import annotations

from typing import Any, TypedDict


class ToolErrorDetail(TypedDict):
    code: str
    message: str


class ToolResult(TypedDict):
    ok: bool
    tool: str
    query: str
    data: Any
    error: ToolErrorDetail | None
    meta: dict[str, Any]


def tool_success(
    *,
    tool_name: str,
    query: str,
    data: Any,
    meta: dict[str, Any] | None = None,
) -> ToolResult:
    """构造成功工具结果。"""
    return {
        "ok": True,
        "tool": tool_name,
        "query": query,
        "data": data,
        "error": None,
        "meta": meta or {},
    }


def tool_error(
    *,
    tool_name: str,
    query: str,
    message: str,
    code: str = "TOOL_ERROR",
    meta: dict[str, Any] | None = None,
) -> ToolResult:
    """构造失败工具结果。"""
    return {
        "ok": False,
        "tool": tool_name,
        "query": query,
        "data": None,
        "error": {"code": code, "message": message},
        "meta": meta or {},
    }


__all__ = ["ToolErrorDetail", "ToolResult", "tool_error", "tool_success"]
