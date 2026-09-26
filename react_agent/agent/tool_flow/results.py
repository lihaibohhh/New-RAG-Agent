"""工具结果的当前轮统计与批次观测投影。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AnyMessage, ToolMessage

from react_agent.agent.time import now_iso_in_timezone
from react_agent.agent.tool_flow.protocol import (
    extract_recent_tool_messages,
    find_last_real_human_index,
)
from react_agent.tooling.results import decode_tool_result


_RAG_TOOL_NAME = "query_internal_knowledge"
_NON_EXECUTED_RAG_ERROR_CODES = frozenset(
    {
        "GRAPH_STEP_BUDGET_EXHAUSTED",
        "INVALID_TOOL_CALL",
        "RAG_CALL_BUDGET_EXHAUSTED",
        "TOOL_BUDGET_EXHAUSTED",
        "TOOL_DISABLED",
    }
)


@dataclass(frozen=True)
class ToolBatch:
    runs: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    error_count: int
    last_ok_result: dict[str, Any] | None
    consecutive_rag_misses: int


def _current_turn_rag_messages(messages: list[AnyMessage]) -> list[ToolMessage]:
    last_human_index = find_last_real_human_index(messages)
    if last_human_index < 0:
        return []
    return [
        message
        for message in messages[last_human_index:]
        if isinstance(message, ToolMessage)
        and getattr(message, "name", None) == _RAG_TOOL_NAME
    ]


def _try_decode(message: ToolMessage) -> dict[str, Any] | None:
    try:
        return decode_tool_result(getattr(message, "content", None))
    except (TypeError, ValueError):
        return None


def count_attempted_rag_calls_in_current_turn(messages: list[AnyMessage]) -> int:
    """Count RAG calls accepted for execution in the current user turn.

    Successful and failed executions both consume the hard call budget. Calls
    rejected locally before execution do not. Malformed results are counted
    conservatively because the tool was already dispatched.
    """
    count = 0
    for message in _current_turn_rag_messages(messages):
        payload = _try_decode(message)
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
        payload = _try_decode(message)
        if payload is not None and payload.get("ok") is True:
            count += 1
    return count


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
            payload = decode_tool_result(getattr(tool_message, "content", None))
        except Exception as exc:
            payload = {}
            run_ok: bool | None = False
            run_error: Any = {
                "code": "PARSE_FAILED",
                "message": f"{type(exc).__name__}: {exc}",
            }
        else:
            raw_ok = payload.get("ok")
            run_ok = bool(raw_ok) if raw_ok is not None else None
            run_error = payload.get("error")

        meta = payload.get("meta")
        executed = not (isinstance(meta, dict) and meta.get("executed") is False)

        run = {
            "tool": getattr(tool_message, "name", "") or "unknown_tool",
            "query": payload.get("query"),
            "executed": executed,
            "ok": run_ok,
            "error": run_error,
            "meta": meta,
            "sources": _source_summaries(payload),
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


def _source_summaries(
    payload: dict[str, Any], *, limit: int = 12
) -> list[dict[str, Any]]:
    """为观测面板提取有界来源位置，不复制工具正文。"""
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        return []
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, Any, str]] = set()
    for item in data["results"]:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or item.get("source_file") or "")
        page = item.get("page", item.get("source_page"))
        chunk_id = str(item.get("chunk_id") or "")
        if not source and page is None and not chunk_id:
            continue
        key = (source, page, chunk_id)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source_file": source or None,
                "source_page": page,
                "chunk_id": chunk_id or None,
            }
        )
        if len(sources) >= limit:
            break
    return sources


def _count_consecutive_rag_misses(messages: list[AnyMessage]) -> int:
    results: list[bool] = []
    for message in _current_turn_rag_messages(messages):
        payload = _try_decode(message)
        if payload is None or payload.get("ok") is not True:
            continue
        try:
            has_content = payload["meta"]["has_relevant_content"]
        except (KeyError, TypeError):
            has_content = None
        if has_content is not None:
            results.append(bool(has_content))

    consecutive_misses = 0
    for has_content in reversed(results):
        if has_content:
            break
        consecutive_misses += 1
    return consecutive_misses


__all__ = [
    "ToolBatch",
    "count_attempted_rag_calls_in_current_turn",
    "count_successful_rag_calls_in_current_turn",
    "parse_tool_batch",
]
