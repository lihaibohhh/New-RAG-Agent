"""Agent 对 LangChain 工具调用消息的协议适配。"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from react_agent.tooling.results import tool_error


SYSTEM_SENTINEL_NAMES = frozenset({"system_monitor", "system_terminator"})


def extract_tool_call_ids(message: AnyMessage) -> list[str]:
    """返回模型消息中全部工具调用 ID，去重并保留原顺序。"""
    if not isinstance(message, AIMessage):
        return []

    call_ids: list[str] = []
    for call in getattr(message, "tool_calls", None) or []:
        call_id = (
            call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
        )
        if call_id and call_id not in call_ids:
            call_ids.append(call_id)
    for call in getattr(message, "invalid_tool_calls", None) or []:
        call_id = (
            call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
        )
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
        tool_call.get("name") or (tool_call.get("function") or {}).get("name") or ""
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


def close_tool_calls_for_budget(
    message: AIMessage,
    *,
    termination_reason: str | None,
    error_code: str = "TOOL_BUDGET_EXHAUSTED",
    error_message: str = "本轮工具调用预算已耗尽，该工具未执行。",
) -> list[ToolMessage]:
    """为预算终止时未执行的调用构造一一对应的 ToolMessage。"""
    closed: list[ToolMessage] = []
    seen_ids: set[str] = set()
    calls = list(message.tool_calls or [])
    calls.extend(getattr(message, "invalid_tool_calls", None) or [])

    for call in calls:
        call_id = (
            call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
        )
        if not call_id or call_id in seen_ids:
            continue
        seen_ids.add(call_id)
        tool_name = get_tool_call_name(call) or "unknown"
        args = (
            call.get("args", "")
            if isinstance(call, dict)
            else getattr(call, "args", "")
        )
        closed.append(
            ToolMessage(
                tool_call_id=call_id,
                name=tool_name,
                content=json.dumps(
                    tool_error(
                        tool_name=tool_name,
                        query=str(args)[:200],
                        code=error_code,
                        message=error_message,
                        meta={
                            "executed": False,
                            "termination_reason": termination_reason,
                        },
                    ),
                    ensure_ascii=False,
                ),
            )
        )
    return closed


__all__ = [
    "SYSTEM_SENTINEL_NAMES",
    "close_tool_calls_for_budget",
    "extract_recent_tool_messages",
    "extract_tool_call_ids",
    "find_last_real_human_index",
    "get_tool_call_name",
]
