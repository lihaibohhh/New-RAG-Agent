"""工具结果信封的上下文体积控制。"""

from __future__ import annotations

import json
from typing import Any


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


__all__ = ["bound_tool_payload"]
