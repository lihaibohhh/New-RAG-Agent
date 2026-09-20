"""模型历史消息的离线 Token 估算与预算裁剪。"""

from __future__ import annotations

import json
from typing import Any

from react_agent.agent.context_management.segmentation import (
    segment_current_turn,
    segment_turn_blocks,
)


def _serialized_text(value: Any) -> str:
    """把文本块和工具参数计入预算，避免只统计可见正文。"""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _estimate_text_tokens(text: str) -> int:
    """使用 UTF-8 长度作保守近似；不依赖下载的 tokenizer 资源。"""
    return (len(text.encode("utf-8", errors="replace")) + 1) // 2


class MessageTokenCounter:
    """在一次上下文构造中，每条消息只估算一次。"""

    def __init__(self) -> None:
        self._cache: dict[int, tuple[Any, int]] = {}

    def count(self, messages: list[Any]) -> int:
        return sum(self._message_tokens(message) for message in messages)

    def _message_tokens(self, message: Any) -> int:
        key = id(message)
        cached = self._cache.get(key)
        if cached is not None and cached[0] is message:
            return cached[1]

        if isinstance(message, dict):
            content = message.get("content", "")
            tool_calls = message.get("tool_calls") or []
        else:
            content = getattr(message, "content", "")
            tool_calls = getattr(message, "tool_calls", None) or []
        text = _serialized_text(content or "")
        tokens = 4 + _estimate_text_tokens(text)
        for tool_call in tool_calls:
            tokens += _estimate_text_tokens(_serialized_text(tool_call))
        self._cache[key] = (message, tokens)
        return tokens


def estimate_tool_schema_tokens(
    tools: tuple[Any, ...], *, token_counter: MessageTokenCounter
) -> int:
    """估算绑定工具的名称、描述和参数 schema；Provider 包装开销另留余量。"""
    total = 0
    for tool in tools:
        try:
            schema = tool.tool_call_schema.model_json_schema()
        except (AttributeError, TypeError, ValueError):
            schema = getattr(tool, "args", {})
        descriptor = {
            "name": str(getattr(tool, "name", "")),
            "description": str(getattr(tool, "description", "")),
            "parameters": schema,
        }
        total += token_counter.count([{"content": descriptor}])
    return total


def estimate_message_tokens(messages: list[Any]) -> int:
    """离线估算消息 Token；不代表具体模型的精确用量。"""
    return MessageTokenCounter().count(messages)


def trim_history_to_budget(
    messages: list[Any],
    *,
    max_tokens: int,
    token_counter: MessageTokenCounter | None = None,
) -> list[Any]:
    """连续保留最近的完整轮次；当前轮不受历史子预算裁断。"""
    counter = token_counter or MessageTokenCounter()
    segments = segment_current_turn(messages)
    current = list(segments.current)
    remaining = max_tokens - counter.count(current)
    kept: list[Any] = []
    for block in reversed(segment_turn_blocks(segments.completed)):
        block_messages = list(block.messages)
        block_tokens = counter.count(block_messages)
        if block_tokens > remaining:
            break
        kept[:0] = block_messages
        remaining -= block_tokens
    return kept + current


__all__ = [
    "MessageTokenCounter",
    "estimate_message_tokens",
    "estimate_tool_schema_tokens",
    "trim_history_to_budget",
]
