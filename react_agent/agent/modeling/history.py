"""发送给模型前的消息历史裁剪与协议修复。"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, trim_messages

from react_agent.agent.tool_flow.calls import (
    SYSTEM_SENTINEL_NAMES,
    extract_tool_call_ids,
)


logger = logging.getLogger(__name__)


def latest_turn_ai_messages(messages: Sequence[Any]) -> list[AIMessage]:
    """选出最近一轮用户输入后的所有模型回复，包括工具规划消息。"""
    last_human = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], HumanMessage)
        ),
        None,
    )
    if last_human is None:
        return []
    return [
        message
        for message in messages[last_human + 1 :]
        if isinstance(message, AIMessage)
    ]


def _count_tokens_for_trim(messages) -> int:
    """使用 cl100k_base 估算 token；编码资源不可用时降级为字符估算。"""
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        total = 0
        for message in messages:
            content = getattr(message, "content", "") or ""
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            total += len(encoding.encode(str(content)))
            for tool_call in getattr(message, "tool_calls", None) or []:
                total += len(encoding.encode(json.dumps(tool_call, ensure_ascii=False)))
            total += 4
        return total
    except Exception as exc:
        logger.warning(
            "token_counter_fallback | reason=%s: %s",
            type(exc).__name__,
            exc,
        )
        return (
            sum(len(str(getattr(message, "content", ""))) for message in messages) // 3
        )


def collapse_consecutive_humans(messages: list[Any]) -> list[Any]:
    """合并连续 HumanMessage，保持 Provider 可接受的消息顺序。"""
    if not messages:
        return messages
    result: list[Any] = []
    for message in messages:
        if (
            isinstance(message, HumanMessage)
            and result
            and isinstance(result[-1], HumanMessage)
        ):
            previous = result[-1]
            merged = (
                str(previous.content).rstrip() + "\n\n" + str(message.content).lstrip()
            )
            result[-1] = HumanMessage(content=merged)
        else:
            result.append(message)
    return result


def sanitize_dangling_tool_calls(messages: list[Any]) -> list[Any]:
    """为未应答的工具调用补临时 ToolMessage，避免模型协议错误。"""
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
                                '"工具调用未完成，结果在历史处理中丢失"}'
                            ),
                        )
                    )
                    logger.info(
                        "为悬空 tool_call 补临时 ToolMessage: %s",
                        tool_call_id,
                    )
            index = next_index
            continue
        index += 1
    return result


def prepare_model_messages(messages: list[Any], config: Any) -> list[Any]:
    """过滤历史哨兵并按配置裁剪，始终保留最近一个完整轮次。"""
    filtered = [
        message
        for message in messages
        if not (
            isinstance(message, HumanMessage)
            and getattr(message, "name", None) in SYSTEM_SENTINEL_NAMES
        )
    ]
    if not (config.enable_history_truncation and config.max_history_tokens > 0):
        return collapse_consecutive_humans(filtered)

    trimmed = trim_messages(
        filtered,
        max_tokens=config.max_history_tokens,
        strategy="last",
        token_counter=_count_tokens_for_trim,
        include_system=False,
        allow_partial=False,
        start_on="human",
    )
    if not trimmed:
        last_human = next(
            (
                index
                for index in range(len(filtered) - 1, -1, -1)
                if isinstance(filtered[index], HumanMessage)
            ),
            None,
        )
        trimmed = filtered[last_human:] if last_human is not None else filtered[-1:]
        logger.warning("消息裁剪结果为空，已回退到最近一个完整轮次")
    return collapse_consecutive_humans(trimmed)


def build_model_input(
    system_prompt: str,
    directive: str | None,
    messages: list[Any],
) -> list[Any]:
    """组合稳定系统提示、瞬态控制指令和历史消息。"""
    head: list[Any] = [{"role": "system", "content": system_prompt}]
    if directive:
        head.append({"role": "system", "content": directive})
    return head + messages


__all__ = [
    "build_model_input",
    "collapse_consecutive_humans",
    "latest_turn_ai_messages",
    "prepare_model_messages",
    "sanitize_dangling_tool_calls",
]
