"""将工具消息归一化为供 State 与策略消费的批次摘要。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AnyMessage, ToolMessage

from react_agent.agent.support.time import now_iso_in_timezone
from react_agent.agent.tool_flow.calls import (
    extract_recent_tool_messages,
    find_last_real_human_index,
)


@dataclass(frozen=True)
class ToolBatch:
    runs: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    error_count: int
    last_ok_result: dict[str, Any] | None
    consecutive_rag_misses: int


def parse_tool_batch(
    messages: list[AnyMessage],
    *,
    timezone: str,
) -> ToolBatch | None:
    """解析消息尾部工具结果，并计算当前轮连续 RAG 未命中次数。"""
    tool_messages = extract_recent_tool_messages(messages)
    if not tool_messages:
        return None

    runs: list[dict[str, Any]] = []
    last_ok_result: dict[str, Any] | None = None
    for tool_message in tool_messages:
        try:
            parsed = json.loads(getattr(tool_message, "content", None) or "{}")
            if not isinstance(parsed, dict):
                raise TypeError(
                    "工具返回的 JSON 顶层类型应为 object，"
                    f"实际为 {type(parsed).__name__}"
                )
        except Exception as exc:
            payload: dict[str, Any] = {}
            run_ok: bool | None = False
            run_error: Any = {
                "code": "PARSE_FAILED",
                "message": f"{type(exc).__name__}: {exc}",
            }
        else:
            payload = parsed
            raw_ok = parsed.get("ok")
            run_ok = bool(raw_ok) if raw_ok is not None else None
            run_error = parsed.get("error")

        run = {
            "tool": getattr(tool_message, "name", "") or "unknown_tool",
            "query": payload.get("query"),
            "ok": run_ok,
            "error": run_error,
            "meta": payload.get("meta"),
            "ts": now_iso_in_timezone(timezone),
        }
        if payload.get("ok") is True:
            last_ok_result = payload
        runs.append(run)

    errors = [run for run in runs if run.get("ok") is False]
    return ToolBatch(
        runs=runs,
        errors=errors,
        error_count=len(errors),
        last_ok_result=last_ok_result,
        consecutive_rag_misses=_count_consecutive_rag_misses(messages),
    )


def _count_consecutive_rag_misses(messages: list[AnyMessage]) -> int:
    last_human_index = find_last_real_human_index(messages)
    if last_human_index < 0:
        return 0

    results: list[bool] = []
    for message in messages[last_human_index:]:
        if not (
            isinstance(message, ToolMessage)
            and getattr(message, "name", None) == "query_internal_knowledge"
        ):
            continue
        try:
            payload = json.loads(getattr(message, "content", None) or "{}")
            has_content = payload["meta"]["has_relevant_content"]
        except Exception:
            has_content = None
        if has_content is not None:
            results.append(bool(has_content))

    consecutive_misses = 0
    for has_content in reversed(results):
        if has_content:
            break
        consecutive_misses += 1
    return consecutive_misses


__all__ = ["ToolBatch", "parse_tool_batch"]
