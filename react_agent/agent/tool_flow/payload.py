"""工具结果信封的上下文体积控制。"""

from __future__ import annotations

import json
from typing import Any

from react_agent.tooling.results import decode_tool_result


_RAG_TOOL = "query_internal_knowledge"
_MIN_VISIBLE_CONTENT_CHARS = 24


def bound_tool_payload(content: str, max_chars: int) -> str:
    """按工具结果信封结构裁剪正文，同时保留状态、错误和溯源元数据。"""
    floor = max(max_chars, 64)
    try:
        payload = decode_tool_result(content)
    except Exception:
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
    if payload.get("tool") == _RAG_TOOL and isinstance(data.get("results"), list):
        return _bound_rag_payload(payload, max_chars)
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


def _bound_rag_payload(payload: dict[str, Any], max_chars: int) -> str:
    """为多个 RAG 片段预留来源与少量可见正文，再公平分配剩余预算。"""
    budget = max(max_chars, 64)
    data = payload["data"]
    meta = payload["meta"]
    original = data["results"]
    retrieval_hit = bool(
        original or meta.get("retrieval_hit") or meta.get("has_relevant_content")
    )
    meta.update(
        retrieval_hit=retrieval_hit,
        has_relevant_content=False,
        has_visible_content=False,
        total_results=len(original),
        visible_results=0,
        kept_results=0,
        truncated=False,
    )
    data.update(results=[], has_relevant_content=False, has_visible_content=False)

    # 诊断字段不应挤掉来源与正文；工具运行轨迹仍保留核心命中状态。
    optional_meta = (
        "timings",
        "top_score",
        "cache_hit",
        "stage",
        "reranked_count",
        "candidates_count",
        "retrieved_count",
    )
    bodies: list[str] = []
    for raw in original:
        if not isinstance(raw, dict):
            continue
        body = str(raw.get("content") or "")
        if body.startswith("[来源：") and "\n" in body:
            body = body.split("\n", 1)[1]
        body = body.strip()
        if not body:
            continue
        item = {key: value for key, value in raw.items() if key != "content"}
        minimum = min(len(body), _MIN_VISIBLE_CONTENT_CHARS)
        item.update(content=body[:minimum] + ("…" if len(body) > minimum else ""))
        if len(body) > minimum:
            item["content_truncated"] = True
        data["results"].append(item)
        meta["visible_results"] = meta["kept_results"] = len(data["results"])
        while _json_length(payload) > budget and optional_meta:
            meta.pop(optional_meta[0], None)
            optional_meta = optional_meta[1:]
        for key in ("score", "doc_type", "industry"):
            if _json_length(payload) <= budget:
                break
            item.pop(key, None)
        if _json_length(payload) > budget:
            data["results"].pop()
            meta["visible_results"] = meta["kept_results"] = len(data["results"])
            break
        bodies.append(body)

    if not bodies and retrieval_hit:
        return _rag_budget_error(budget, total_results=len(original))

    for index, body in enumerate(bodies):
        item = data["results"][index]
        remaining_items = len(bodies) - index
        baseline = _json_length(payload)
        share = max(0, (budget - baseline) // remaining_items)
        target_length = min(budget, baseline + share)
        current_length = min(len(body), _MIN_VISIBLE_CONTENT_CHARS)
        low, high = current_length, len(body)
        best = current_length
        while low <= high:
            middle = (low + high) // 2
            candidate = body[:middle] + ("…" if middle < len(body) else "")
            item["content"] = candidate
            if middle < len(body):
                item["content_truncated"] = True
            else:
                item.pop("content_truncated", None)
            if _json_length(payload) <= target_length:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        item["content"] = body[:best] + ("…" if best < len(body) else "")
        if best < len(body):
            item["content_truncated"] = True
        else:
            item.pop("content_truncated", None)

    visible = bool(data["results"])
    data["has_relevant_content"] = data["has_visible_content"] = visible
    meta["has_relevant_content"] = meta["has_visible_content"] = visible
    meta["truncated"] = len(bodies) < len(original) or any(
        item.get("content_truncated") for item in data["results"]
    )
    encoded = _dump(payload)
    if len(encoded) > budget:
        return _rag_budget_error(budget, total_results=len(original))
    return encoded


def _json_length(value: dict[str, Any]) -> int:
    return len(_dump(value))


def _dump(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _rag_budget_error(budget: int, *, total_results: int) -> str:
    error = {
        "ok": False,
        "tool": _RAG_TOOL,
        "query": "",
        "data": None,
        "error": {
            "code": "EVIDENCE_PAYLOAD_TOO_LARGE",
            "message": "检索结果超过输出预算，未向模型提供可引用的正文。",
        },
        "meta": {
            "retrieval_hit": bool(total_results),
            "has_relevant_content": False,
            "has_visible_content": False,
            "total_results": total_results,
            "visible_results": 0,
            "truncated": True,
        },
    }
    if _json_length(error) <= budget:
        return _dump(error)
    return _dump({"ok": False, "error": {"code": "EVIDENCE_PAYLOAD_TOO_LARGE"}})


def _trim_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


__all__ = ["bound_tool_payload"]
