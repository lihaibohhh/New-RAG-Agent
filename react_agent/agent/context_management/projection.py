"""仅对本次模型可见的工具结果执行确定性预算投影。"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import ToolMessage

from react_agent.agent.context_management.segmentation import segment_current_turn


_RAG_TOOL = "query_internal_knowledge"
_MIN_ENVELOPE_CHARS = 256


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _project_rag(payload: dict[str, Any], original_chars: int, limit: int) -> str:
    data = payload.get("data")
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        return _project_envelope(payload, original_chars, limit)

    original_meta = payload.get("meta")
    original_meta = original_meta if isinstance(original_meta, dict) else {}
    total_results = original_meta.get("total_results")
    if not isinstance(total_results, int) or isinstance(total_results, bool):
        total_results = len(results)
    total_results = max(total_results, len(results))
    projected: dict[str, Any] = {
        "ok": payload.get("ok"),
        "tool": payload.get("tool") or _RAG_TOOL,
        "query": str(payload.get("query") or "")[:160],
        "data": {"results": [], "has_visible_content": False},
        "meta": {
            "context_truncated": True,
            "original_chars": original_chars,
            "retrieval_hit": bool(
                original_meta.get("retrieval_hit")
                or original_meta.get("has_relevant_content")
                or total_results
            ),
            "total_results": total_results,
            "visible_results": 0,
            "omitted_results": total_results,
            "has_visible_content": False,
        },
    }
    visible: list[dict[str, Any]] = projected["data"]["results"]
    for raw in results:
        if not isinstance(raw, dict):
            continue
        item = {
            key: raw[key]
            for key in ("chunk_id", "source", "source_file", "page", "source_page")
            if key in raw
        }
        body = str(raw.get("content") or "")
        maximum_excerpt = min(len(body), max(0, limit // max(len(results), 1) // 3))
        low, high = 0, maximum_excerpt
        best = -1
        visible.append(item)
        while low <= high:
            middle = (low + high) // 2
            item["content"] = body[:middle]
            item["content_truncated"] = bool(raw.get("content_truncated")) or (
                middle < len(body)
            )
            projected["meta"]["visible_results"] = len(visible)
            projected["meta"]["omitted_results"] = total_results - len(visible)
            if len(_dump(projected)) <= limit:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best < 0:
            visible.pop()
            break
        item["content"] = body[:best]
        item["content_truncated"] = bool(raw.get("content_truncated")) or (
            best < len(body)
        )

    has_visible_content = any(item.get("content") for item in visible)
    projected["data"]["has_visible_content"] = bool(has_visible_content)
    projected["meta"].update(
        visible_results=len(visible),
        omitted_results=total_results - len(visible),
        has_visible_content=bool(has_visible_content),
    )
    if not has_visible_content:
        projected["meta"]["citation_warning"] = (
            "预算下无可见证据正文，不得据此引用或断言检索无结果"
        )
    encoded = _dump(projected)
    if len(encoded) <= limit:
        return encoded
    return _dump(
        {
            "ok": payload.get("ok"),
            "tool": _RAG_TOOL,
            "data": {"results": []},
            "meta": {
                "context_truncated": True,
                "retrieval_hit": projected["meta"]["retrieval_hit"],
                "omitted_results": total_results,
                "citation_warning": "预算下无可引用正文",
            },
        }
    )


def _project_envelope(payload: dict[str, Any], original_chars: int, limit: int) -> str:
    data = payload.get("data")
    error = payload.get("error")
    projected: dict[str, Any] = {
        "ok": payload.get("ok"),
        "tool": payload.get("tool"),
        "query": str(payload.get("query") or "")[:120],
        "data": {"preview": "", "context_truncated": True},
        "error": (
            {"code": error.get("code"), "message": str(error.get("message") or "")[:160]}
            if isinstance(error, dict)
            else str(error)[:160] if error is not None else None
        ),
        "meta": {"context_truncated": True, "original_chars": original_chars},
    }
    preview = _dump(data)
    low, high = 0, len(preview)
    best = 0
    while low <= high:
        middle = (low + high) // 2
        projected["data"]["preview"] = preview[:middle]
        if len(_dump(projected)) <= limit:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    projected["data"]["preview"] = preview[:best]
    encoded = _dump(projected)
    if len(encoded) <= limit:
        return encoded
    error = payload.get("error")
    return _dump(
        {
            "ok": payload.get("ok"),
            "tool": payload.get("tool"),
            "error_code": error.get("code") if isinstance(error, dict) else None,
            "meta": {"context_truncated": True, "original_chars": original_chars},
        }
    )


def project_tool_content(content: str, max_chars: int) -> str:
    """保留结果状态和 RAG 溯源；任何省略均显式标记。"""
    if len(content) <= max_chars:
        return content
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        marker = "\n[工具结果仅为前缀；其余正文因本次模型输入预算被截断]"
        limit = max(max_chars, len(marker))
        return content[: max(0, limit - len(marker))] + marker
    if not isinstance(payload, dict):
        marker = "\n[工具结果仅为前缀；其余正文因本次模型输入预算被截断]"
        limit = max(max_chars, len(marker))
        return content[: max(0, limit - len(marker))] + marker
    limit = max(max_chars, _MIN_ENVELOPE_CHARS)
    if payload.get("tool") == _RAG_TOOL:
        projected = _project_rag(payload, len(content), limit)
    else:
        projected = _project_envelope(payload, len(content), limit)
    return projected if len(projected) < len(content) else content


def _project_tool_messages_in_range(
    messages: list[Any], *, start: int, stop: int, max_chars: int
) -> tuple[list[Any], int, int]:
    result = list(messages)
    projected_count = 0
    removed_chars = 0
    for index in range(start, stop):
        message = result[index]
        if not isinstance(message, ToolMessage) or not isinstance(
            message.content, str
        ):
            continue
        content = project_tool_content(message.content, max_chars)
        if len(content) >= len(message.content):
            continue
        result[index] = message.model_copy(update={"content": content})
        projected_count += 1
        removed_chars += len(message.content) - len(content)
    return result, projected_count, removed_chars


def project_current_tool_messages(
    messages: list[Any], *, max_chars: int
) -> tuple[list[Any], int, int]:
    """只投影当前轮工具正文，保留调用与结果消息外壳。"""
    current_start = len(segment_current_turn(messages).completed)
    return _project_tool_messages_in_range(
        messages, start=current_start, stop=len(messages), max_chars=max_chars
    )


def project_completed_tool_messages(
    messages: list[Any], *, max_chars: int
) -> tuple[list[Any], int, int]:
    """只投影已完成轮次的工具正文，不触碰当前轮。"""
    current_start = len(segment_current_turn(messages).completed)
    return _project_tool_messages_in_range(
        messages, start=0, stop=current_start, max_chars=max_chars
    )


__all__ = [
    "project_completed_tool_messages",
    "project_current_tool_messages",
    "project_tool_content",
]
