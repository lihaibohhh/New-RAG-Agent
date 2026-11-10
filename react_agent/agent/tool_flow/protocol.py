"""Agent 对 LangChain 工具调用消息的协议适配。"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from agent_tools.contracts.results import tool_error


SYSTEM_SENTINEL_NAMES = frozenset({"system_monitor", "system_terminator"})


logger = logging.getLogger(__name__)


class ToolProtocolError(ValueError):
    """工具调用与结果无法形成合法的一一对应关系。"""


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


def _raw_tool_call_ids(message: AIMessage) -> list[str]:
    calls = list(message.tool_calls or []) + list(message.invalid_tool_calls or [])
    if not calls:
        calls = list((message.additional_kwargs or {}).get("tool_calls", []))
    return [
        str(
            (
                call.get("id")
                if isinstance(call, dict)
                else getattr(call, "id", "")
            )
            or ""
        )
        for call in calls
    ]


def _repair_missing_tool_results(messages: list[Any]) -> list[Any]:
    """只补齐具有明确调用 ID、但没有对应结果的工具调用。"""
    source = list(messages)
    result: list[Any] = []
    index = 0
    while index < len(source):
        message = source[index]
        result.append(message)
        required_ids = extract_tool_call_ids(message)
        if not required_ids:
            index += 1
            continue

        answered_ids: list[str] = []
        next_index = index + 1
        while next_index < len(source) and isinstance(
            source[next_index], ToolMessage
        ):
            result.append(source[next_index])
            answered_ids.append(source[next_index].tool_call_id)
            next_index += 1
        for tool_call_id in required_ids:
            if tool_call_id in answered_ids:
                continue
            result.append(
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=json.dumps(
                        tool_error(
                            tool_name="unknown",
                            query="",
                            code="TOOL_RESULT_MISSING",
                            message="工具调用未完成，结果未记录。",
                            meta={"executed": False, "synthetic": True},
                        ),
                        ensure_ascii=False,
                    ),
                )
            )
            logger.info("为悬空 tool_call 补 ToolMessage: %s", tool_call_id)
        index = next_index
    return result


def _validate_tool_exchanges(messages: list[Any]) -> None:
    """校验并行调用的每个 ID 恰有一个紧邻的工具结果。"""
    index = 0
    while index < len(messages):
        message = messages[index]
        if isinstance(message, ToolMessage):
            raise ToolProtocolError(f"孤立的工具结果: {message.tool_call_id}")
        if isinstance(message, AIMessage):
            raw_ids = _raw_tool_call_ids(message)
            if any(not call_id for call_id in raw_ids):
                raise ToolProtocolError("模型工具调用缺少 ID")
            if len(raw_ids) != len(set(raw_ids)):
                raise ToolProtocolError("模型消息中存在重复的工具调用 ID")

        call_ids = extract_tool_call_ids(message)
        if not call_ids:
            index += 1
            continue
        results: list[ToolMessage] = []
        next_index = index + 1
        while next_index < len(messages) and isinstance(
            messages[next_index], ToolMessage
        ):
            results.append(messages[next_index])
            next_index += 1
        result_ids = [result.tool_call_id for result in results]
        if len(result_ids) != len(set(result_ids)):
            raise ToolProtocolError("同一工具调用存在重复结果")
        if set(result_ids) != set(call_ids):
            raise ToolProtocolError(
                f"工具调用与结果不匹配: calls={call_ids}, results={result_ids}"
            )
        index = next_index


def repair_and_validate_tool_exchanges(messages: list[Any]) -> list[Any]:
    """补齐明确缺失的工具结果，并严格校验最终消息序列。"""
    repaired = _repair_missing_tool_results(messages)
    _validate_tool_exchanges(repaired)
    return repaired


def repair_and_validate_tool_results(
    assistant_message: AIMessage,
    tool_messages: list[Any],
) -> list[ToolMessage]:
    """在 ToolMessage 进入 State 前补齐并校验一个完整工具批次。"""
    repaired = repair_and_validate_tool_exchanges(
        [assistant_message, *tool_messages]
    )[1:]
    if not all(isinstance(message, ToolMessage) for message in repaired):
        raise ToolProtocolError("工具执行批次包含非 ToolMessage 结果")
    return list(repaired)


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
    "ToolProtocolError",
    "close_tool_calls_for_budget",
    "extract_recent_tool_messages",
    "extract_tool_call_ids",
    "find_last_real_human_index",
    "get_tool_call_name",
    "repair_and_validate_tool_exchanges",
    "repair_and_validate_tool_results",
]
