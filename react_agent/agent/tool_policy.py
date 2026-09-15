"""Agent 工具目录、调用配额与上下文大小策略。"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AnyMessage, ToolMessage

from react_agent.agent.tool_calls import find_last_real_human_index


def render_tool_catalog(tools: tuple[Any, ...], *, tools_enabled: bool) -> str:
    """将本轮实际启用的工具集合渲染进系统提示。"""
    if not tools_enabled:
        return "当前运行配置已禁用工具调用（tools_enabled=false），本轮不允许使用任何工具。"
    if not tools:
        return "当前未配置任何工具。"

    lines = ["你当前可使用以下工具："]
    for index, tool in enumerate(tools, start=1):
        name = getattr(tool, "name", None) or getattr(tool, "__name__", "unknown_tool")
        description = " ".join(str(getattr(tool, "description", "") or "").split())
        if len(description) > 240:
            description = description[:239] + "..."
        detail = description or "暂无描述（建议为工具增加 description）"
        lines.append(f"{index}) {name}：{detail}")
    lines.append(
        "当用户询问'你有哪些工具/能做什么'时，只能按上述目录回答，不得编造。"
        "如果工具调用失败，请根据错误原因调整参数后重试，不要盲目重复。"
    )
    return "\n".join(lines)


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


def bound_tool_payload(content: str, max_chars: int) -> str:
    """按工具结果信封结构裁剪正文，同时保留状态、错误和溯源元数据。"""
    floor = max(max_chars, 64)
    try:
        payload = json.loads(content or "{}")
    except Exception:
        return _trim_text(content, floor)
    if not isinstance(payload, dict):
        return _trim_text(content, floor)

    meta = payload.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = payload["meta"] = {}

    if payload.get("ok") is False or payload.get("data") is None:
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            if len(error["message"]) > floor:
                error["message"] = _trim_text(error["message"], floor)
                meta["truncated"] = True
        return json.dumps(payload, ensure_ascii=False)

    data = payload.get("data")
    if not isinstance(data, dict):
        return json.dumps(payload, ensure_ascii=False)
    results = data.get("results")
    if isinstance(results, list) and results:
        kept: list[Any] = []
        used = 0
        for item in results:
            serialized = json.dumps(item, ensure_ascii=False)
            if used + len(serialized) > max_chars:
                break
            kept.append(item)
            used += len(serialized)
        if len(kept) < len(results):
            data["results"] = kept
            meta.update(
                truncated=True,
                kept_results=len(kept),
                total_results=len(results),
            )
        else:
            meta.setdefault("truncated", False)
    return json.dumps(payload, ensure_ascii=False)


def _trim_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


__all__ = [
    "bound_tool_payload",
    "count_successful_rag_calls_in_current_turn",
    "render_tool_catalog",
]
