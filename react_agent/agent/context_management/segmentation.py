"""按用户轮次划分和规范化 Agent 消息。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from react_agent.agent.context_management.contracts import TurnBlock, TurnSegments
from react_agent.agent.tool_flow import SYSTEM_SENTINEL_NAMES


def is_real_human_message(message: Any) -> bool:
    return (
        isinstance(message, HumanMessage)
        and getattr(message, "name", None) not in SYSTEM_SENTINEL_NAMES
    )


def filter_runtime_sentinels(messages: Sequence[Any]) -> list[Any]:
    """移除仅供图内部控制使用、不应发送给模型的哨兵消息。"""
    return [
        message
        for message in messages
        if not (
            isinstance(message, HumanMessage)
            and getattr(message, "name", None) in SYSTEM_SENTINEL_NAMES
        )
    ]


def segment_turn_blocks(messages: Sequence[Any]) -> tuple[TurnBlock, ...]:
    """按用户轮次分组；紧邻的多条用户消息属于同一轮。"""
    blocks: list[TurnBlock] = []
    current: list[Any] = []
    for message in messages:
        if is_real_human_message(message) and current and not is_real_human_message(
            current[-1]
        ):
            blocks.append(TurnBlock(messages=tuple(current)))
            current = []
        current.append(message)
    if current:
        blocks.append(TurnBlock(messages=tuple(current)))
    return tuple(blocks)


def segment_current_turn(messages: Sequence[Any]) -> TurnSegments:
    """最后一个含真实用户输入的 TurnBlock 为当前轮。"""
    blocks = segment_turn_blocks(messages)
    if not blocks or not any(
        is_real_human_message(message) for message in blocks[-1].messages
    ):
        return TurnSegments(completed=tuple(messages))
    return TurnSegments(
        completed=tuple(
            message for block in blocks[:-1] for message in block.messages
        ),
        current=blocks[-1].messages,
    )


def latest_turn_ai_messages(messages: Sequence[Any]) -> list[AIMessage]:
    """选出最近一轮用户输入后的所有模型回复，包括工具规划消息。"""
    return [
        message
        for message in segment_current_turn(messages).current[1:]
        if isinstance(message, AIMessage)
    ]


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


__all__ = [
    "collapse_consecutive_humans",
    "filter_runtime_sentinels",
    "is_real_human_message",
    "latest_turn_ai_messages",
    "segment_current_turn",
    "segment_turn_blocks",
]
