"""基于消息历史计算当前用户轮次的工具调用预算。"""

from __future__ import annotations

import json

from langchain_core.messages import AnyMessage, ToolMessage

from react_agent.agent.tool_flow.calls import find_last_real_human_index


def count_successful_rag_calls_in_current_turn(messages: list[AnyMessage]) -> int:
    """统计当前用户轮次中成功完成的私有知识库检索次数。"""
    last_human_index = find_last_real_human_index(messages)
    if last_human_index < 0:
        return 0

    count = 0
    for message in messages[last_human_index:]:
        if not (
            isinstance(message, ToolMessage)
            and getattr(message, "name", None) == "query_internal_knowledge"
        ):
            continue
        try:
            payload = json.loads(getattr(message, "content", None) or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("ok") is True:
            count += 1
    return count


__all__ = ["count_successful_rag_calls_in_current_turn"]
