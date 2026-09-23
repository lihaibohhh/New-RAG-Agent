"""基于消息历史计算当前用户轮次的工具调用预算。"""

from __future__ import annotations

import json

from langchain_core.messages import AnyMessage, ToolMessage

from react_agent.agent.tool_flow.calls import find_last_real_human_index


_NON_EXECUTED_RAG_ERROR_CODES = frozenset(
    {
        "GRAPH_STEP_BUDGET_EXHAUSTED",
        "INVALID_TOOL_CALL",
        "RAG_CALL_BUDGET_EXHAUSTED",
        "TOOL_BUDGET_EXHAUSTED",
        "TOOL_DISABLED",
    }
)


def _current_turn_rag_messages(messages: list[AnyMessage]) -> list[ToolMessage]:
    last_human_index = find_last_real_human_index(messages)
    if last_human_index < 0:
        return []
    return [
        message
        for message in messages[last_human_index:]
        if isinstance(message, ToolMessage)
        and getattr(message, "name", None) == "query_internal_knowledge"
    ]


def _parse_payload(message: ToolMessage) -> dict | None:
    try:
        payload = json.loads(getattr(message, "content", None) or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def count_attempted_rag_calls_in_current_turn(messages: list[AnyMessage]) -> int:
    """Count RAG calls accepted for execution in the current user turn.

    Successful and failed executions both consume the hard call budget. Calls
    rejected locally before execution do not. Malformed results are counted
    conservatively because the tool was already dispatched.
    """
    count = 0
    for message in _current_turn_rag_messages(messages):
        payload = _parse_payload(message)
        if payload is None:
            count += 1
            continue

        meta = payload.get("meta")
        if isinstance(meta, dict) and meta.get("executed") is False:
            continue
        error = payload.get("error")
        code = error.get("code") if isinstance(error, dict) else None
        if code in _NON_EXECUTED_RAG_ERROR_CODES:
            continue
        count += 1
    return count


def count_successful_rag_calls_in_current_turn(messages: list[AnyMessage]) -> int:
    """统计当前用户轮次中成功完成的私有知识库检索次数。"""
    count = 0
    for message in _current_turn_rag_messages(messages):
        payload = _parse_payload(message)
        if payload is not None and payload.get("ok") is True:
            count += 1
    return count


__all__ = [
    "count_attempted_rag_calls_in_current_turn",
    "count_successful_rag_calls_in_current_turn",
]
