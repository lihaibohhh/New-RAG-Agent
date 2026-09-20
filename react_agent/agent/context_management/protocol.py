"""发送模型前的工具调用消息协议修复。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from react_agent.agent.context_management.contracts import ToolExchange
from react_agent.agent.tool_flow.calls import extract_tool_call_ids


logger = logging.getLogger(__name__)


class ToolProtocolError(ValueError):
    """待发送消息中的工具调用与结果无法合法配对。"""


def _raw_tool_call_ids(message: AIMessage) -> list[str]:
    calls = list(message.tool_calls or []) + list(message.invalid_tool_calls or [])
    if not calls:
        calls = list((message.additional_kwargs or {}).get("tool_calls", []))
    return [
        str((call.get("id") if isinstance(call, dict) else getattr(call, "id", "")) or "")
        for call in calls
    ]


def validate_tool_exchanges(messages: list[Any]) -> tuple[ToolExchange, ...]:
    """校验并行调用的每个 ID 恰有一个紧邻的工具结果。"""
    exchanges: list[ToolExchange] = []
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
        exchanges.append(
            ToolExchange(
                assistant_message=message,
                tool_messages=tuple(results),
                call_ids=tuple(call_ids),
            )
        )
        index = next_index
    return tuple(exchanges)


def sanitize_dangling_tool_calls(messages: list[Any]) -> list[Any]:
    """为未应答的工具调用补临时 ToolMessage，避免 Provider 协议错误。"""
    source = list(messages)
    result: list[Any] = []
    index = 0
    while index < len(source):
        message = source[index]
        result.append(message)
        required_ids = extract_tool_call_ids(message)
        if required_ids:
            answered_ids: list[str] = []
            next_index = index + 1
            while next_index < len(source) and isinstance(
                source[next_index], ToolMessage
            ):
                result.append(source[next_index])
                answered_ids.append(source[next_index].tool_call_id)
                next_index += 1
            for tool_call_id in required_ids:
                if tool_call_id not in answered_ids:
                    result.append(
                        ToolMessage(
                            tool_call_id=tool_call_id,
                            content=(
                                '{"ok": false, "error": '
                                '"工具调用未完成，结果未记录"}'
                            ),
                        )
                    )
                    logger.info("为悬空 tool_call 补临时 ToolMessage: %s", tool_call_id)
            index = next_index
            continue
        index += 1
    return result


__all__ = [
    "ToolProtocolError",
    "sanitize_dangling_tool_calls",
    "validate_tool_exchanges",
]
