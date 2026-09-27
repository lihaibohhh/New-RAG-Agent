"""知识检索工具对外暴露的稳定结果契约。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from agent_tools.contracts.results import decode_tool_result


RETRIEVAL_RESULT_KIND = "knowledge_retrieval"
_LEGACY_RETRIEVAL_TOOL_NAMES = frozenset({"query_internal_knowledge"})
_MIN_VISIBLE_CONTENT_CHARS = 24


@dataclass(frozen=True)
class EvidenceItem:
    """工具协议中的单条标准化证据。"""

    content: str
    source_file: str
    source_page: int | None
    chunk_id: str
    content_truncated: bool = False


@dataclass(frozen=True)
class RetrievalOutcome:
    """Agent 可消费的知识检索结果，不暴露原始 JSON 层级。"""

    ok: bool
    executed: bool
    query: str
    has_relevant_content: bool | None
    retrieval_hit: bool
    evidence: tuple[EvidenceItem, ...]
    total_results: int
    error_code: str | None


def retrieval_meta(
    *,
    has_relevant_content: bool,
    **metadata: Any,
) -> dict[str, Any]:
    """构造带稳定类型标记的知识检索元数据。"""
    return {
        "result_kind": RETRIEVAL_RESULT_KIND,
        "has_relevant_content": has_relevant_content,
        **metadata,
    }


def is_retrieval_payload(payload: dict[str, Any]) -> bool:
    """识别当前协议及迁移前的知识检索结果。"""
    meta = payload.get("meta")
    if isinstance(meta, dict) and meta.get("result_kind") == RETRIEVAL_RESULT_KIND:
        return True
    return str(payload.get("tool") or "") in _LEGACY_RETRIEVAL_TOOL_NAMES


def decode_retrieval_outcome(content: Any) -> RetrievalOutcome | None:
    """将知识检索信封转换为稳定领域结果；其他工具返回 ``None``。"""
    payload = content if isinstance(content, dict) else decode_tool_result(content)
    if not is_retrieval_payload(payload):
        return None

    meta = payload.get("meta")
    meta = meta if isinstance(meta, dict) else {}
    data = payload.get("data")
    data = data if isinstance(data, dict) else {}
    raw_results = data.get("results")
    raw_results = raw_results if isinstance(raw_results, list) else []

    evidence: list[EvidenceItem] = []
    for raw in raw_results:
        if not isinstance(raw, dict):
            continue
        page = raw.get("source_page", raw.get("page"))
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            page = None
        evidence.append(
            EvidenceItem(
                content=str(raw.get("content") or "").strip(),
                source_file=str(
                    raw.get("source_file") or raw.get("source") or ""
                ),
                source_page=page,
                chunk_id=str(raw.get("chunk_id") or ""),
                content_truncated=bool(raw.get("content_truncated")),
            )
        )

    total_results = meta.get("total_results", len(raw_results))
    if not isinstance(total_results, int) or isinstance(total_results, bool):
        total_results = len(raw_results)
    total_results = max(total_results, len(raw_results))
    relevant = meta.get("has_relevant_content")
    if relevant is None:
        relevant = data.get("has_relevant_content")
    if relevant is not None:
        relevant = bool(relevant)
    retrieval_hit = bool(
        meta.get("retrieval_hit") or relevant or total_results
    )
    error = payload.get("error")
    error_code = str(error.get("code")) if isinstance(error, dict) and error.get("code") else None
    return RetrievalOutcome(
        ok=payload.get("ok") is True,
        executed=meta.get("executed") is not False,
        query=str(payload.get("query") or ""),
        has_relevant_content=relevant,
        retrieval_hit=retrieval_hit,
        evidence=tuple(evidence),
        total_results=total_results,
        error_code=error_code,
    )


def bound_retrieval_payload(payload: dict[str, Any], max_chars: int) -> str:
    """在工具执行边界裁剪检索正文，并保留引用位置和命中状态。"""
    budget = max(max_chars, 64)
    data = payload.get("data")
    data = data if isinstance(data, dict) else {}
    meta = payload.get("meta")
    meta = meta if isinstance(meta, dict) else {}
    payload["data"] = data
    payload["meta"] = meta
    original = data.get("results")
    original = original if isinstance(original, list) else []
    retrieval_hit = bool(
        original or meta.get("retrieval_hit") or meta.get("has_relevant_content")
    )
    meta.update(
        result_kind=RETRIEVAL_RESULT_KIND,
        retrieval_hit=retrieval_hit,
        has_relevant_content=False,
        has_visible_content=False,
        total_results=len(original),
        visible_results=0,
        kept_results=0,
        truncated=False,
    )
    data.update(results=[], has_relevant_content=False, has_visible_content=False)

    optional_meta = [
        "timings",
        "top_score",
        "cache_hit",
        "stage",
        "reranked_count",
        "candidates_count",
        "retrieved_count",
    ]
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
            meta.pop(optional_meta.pop(0), None)
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
        return _retrieval_budget_error(payload, budget, len(original))

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
            item["content"] = body[:middle] + ("…" if middle < len(body) else "")
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
        return _retrieval_budget_error(payload, budget, len(original))
    return encoded


def project_retrieval_payload(
    payload: dict[str, Any], original_chars: int, limit: int
) -> str:
    """为模型上下文生成有界的检索结果投影。"""
    outcome = decode_retrieval_outcome(payload)
    if outcome is None:
        raise ValueError("payload 不是知识检索结果")
    projected: dict[str, Any] = {
        "ok": payload.get("ok"),
        "tool": payload.get("tool"),
        "query": outcome.query[:160],
        "data": {"results": [], "has_visible_content": False},
        "meta": {
            "result_kind": RETRIEVAL_RESULT_KIND,
            "context_truncated": True,
            "original_chars": original_chars,
            "retrieval_hit": outcome.retrieval_hit,
            "total_results": outcome.total_results,
            "visible_results": 0,
            "omitted_results": outcome.total_results,
            "has_visible_content": False,
        },
    }
    visible: list[dict[str, Any]] = projected["data"]["results"]
    data = payload.get("data")
    raw_results = data.get("results") if isinstance(data, dict) else []
    raw_results = raw_results if isinstance(raw_results, list) else []
    for index, evidence in enumerate(outcome.evidence):
        raw = raw_results[index] if index < len(raw_results) else {}
        raw = raw if isinstance(raw, dict) else {}
        item: dict[str, Any] = {"chunk_id": evidence.chunk_id}
        if "source" in raw:
            item["source"] = evidence.source_file
        else:
            item["source_file"] = evidence.source_file
        if "page" in raw:
            item["page"] = evidence.source_page
        else:
            item["source_page"] = evidence.source_page
        body = evidence.content
        maximum_excerpt = min(
            len(body), max(0, limit // max(len(outcome.evidence), 1) // 3)
        )
        low, high = 0, maximum_excerpt
        best = -1
        visible.append(item)
        while low <= high:
            middle = (low + high) // 2
            item["content"] = body[:middle]
            item["content_truncated"] = evidence.content_truncated or middle < len(body)
            projected["meta"]["visible_results"] = len(visible)
            projected["meta"]["omitted_results"] = outcome.total_results - len(visible)
            if len(_dump(projected)) <= limit:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best < 0:
            visible.pop()
            break
        item["content"] = body[:best]
        item["content_truncated"] = evidence.content_truncated or best < len(body)

    has_visible_content = any(item.get("content") for item in visible)
    projected["data"]["has_visible_content"] = bool(has_visible_content)
    projected["meta"].update(
        visible_results=len(visible),
        omitted_results=outcome.total_results - len(visible),
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
            "tool": payload.get("tool"),
            "data": {"results": []},
            "meta": {
                "result_kind": RETRIEVAL_RESULT_KIND,
                "context_truncated": True,
                "retrieval_hit": outcome.retrieval_hit,
                "omitted_results": outcome.total_results,
                "citation_warning": "预算下无可引用正文",
            },
        }
    )


def _retrieval_budget_error(
    payload: dict[str, Any], budget: int, total_results: int
) -> str:
    error = {
        "ok": False,
        "tool": payload.get("tool"),
        "query": "",
        "data": None,
        "error": {
            "code": "EVIDENCE_PAYLOAD_TOO_LARGE",
            "message": "检索结果超过输出预算，未向模型提供可引用的正文。",
        },
        "meta": {
            "result_kind": RETRIEVAL_RESULT_KIND,
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


def _json_length(value: dict[str, Any]) -> int:
    return len(_dump(value))


def _dump(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


__all__ = [
    "EvidenceItem",
    "RETRIEVAL_RESULT_KIND",
    "RetrievalOutcome",
    "bound_retrieval_payload",
    "decode_retrieval_outcome",
    "is_retrieval_payload",
    "project_retrieval_payload",
    "retrieval_meta",
]
