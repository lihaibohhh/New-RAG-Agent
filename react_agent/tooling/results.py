"""所有 Agent Tool 共用的稳定结果信封。"""
from __future__ import annotations

import json
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


def decode_tool_result(content: Any) -> dict[str, Any]:
    """将工具消息正文解码为 JSON object。

    该函数只统一协议解码与顶层类型校验；解析失败后应该计数、
    记录错误还是忽略，由各业务调用方决定。
    """
    payload = json.loads(content or "{}")
    if not isinstance(payload, dict):
        raise TypeError(
            "工具返回的 JSON 顶层类型应为 object，"
            f"实际为 {type(payload).__name__}"
        )
    return payload


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


__all__ = [
    "ToolErrorDetail",
    "ToolResult",
    "decode_tool_result",
    "tool_error",
    "tool_success",
]
