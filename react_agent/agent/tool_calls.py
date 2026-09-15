"""Agent 对 LangChain 工具调用消息的解析规则。"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage


SYSTEM_SENTINEL_NAMES = frozenset({"system_monitor", "system_terminator"})


def extract_tool_call_ids(message: AnyMessage) -> list[str]:
    """返回模型消息中全部工具调用 ID，去重并保留原顺序。"""
    if not isinstance(message, AIMessage):
        return []

    call_ids: list[str] = []
    for call in getattr(message, "tool_calls", None) or []:
        call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
        if call_id and call_id not in call_ids:
            call_ids.append(call_id)
    for call in getattr(message, "invalid_tool_calls", None) or []:
        call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
        if call_id and call_id not in call_ids:
            call_ids.append(call_id)
    if call_ids:
        return call_ids

    additional = getattr(message, "additional_kwargs", None) or {}
    for call in additional.get("tool_calls", []):
        call_id = call.get("id") if isinstance(call, dict) else None
        if call_id and call_id not in call_ids:
            call_ids.append(call_id)
    return call_ids


def get_tool_call_name(tool_call: Any) -> str:
    """兼容 LangChain 的扁平和 OpenAI function 两种工具调用格式。"""
    if not isinstance(tool_call, dict):
        return str(getattr(tool_call, "name", "") or "")
    return str(
        tool_call.get("name")
        or (tool_call.get("function") or {}).get("name")
        or ""
    )


def extract_recent_tool_messages(messages: list[AnyMessage]) -> list[ToolMessage]:
    """提取消息尾部连续的工具结果，同时跳过 Agent 系统哨兵。"""
    result: list[ToolMessage] = []
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            result.append(message)
            continue
        if (
            isinstance(message, HumanMessage)
            and getattr(message, "name", None) in SYSTEM_SENTINEL_NAMES
        ):
            continue
        break
    result.reverse()
    return result


def find_last_real_human_index(messages: list[AnyMessage]) -> int:
    """返回最后一条非系统哨兵 HumanMessage 的索引，找不到时返回 -1。"""
    return next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], HumanMessage)
            and getattr(messages[index], "name", None) not in SYSTEM_SENTINEL_NAMES
        ),
        -1,
    )


__all__ = [
    "SYSTEM_SENTINEL_NAMES",
    "extract_recent_tool_messages",
    "extract_tool_call_ids",
    "find_last_real_human_index",
    "get_tool_call_name",
]
