"""从原始 Agent 消息派生按用户轮次组织的有界记忆。"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import re
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from agent_tools.contracts.results import decode_tool_result
from agent_tools.contracts.retrieval import decode_retrieval_outcome
from react_agent.agent.context_management.contracts import (
    EvidenceSnapshot,
    TurnMemory,
)
from react_agent.agent.context_management.segmentation import (
    is_real_human_message,
    segment_current_turn,
    segment_turn_blocks,
)
from react_agent.agent.tool_flow import extract_tool_call_ids


MAX_TURN_RAG_EVIDENCE = 3
MAX_TURN_WEB_EVIDENCE = 2
MAX_CONVERSATION_EVIDENCE = 14
MAX_RAG_EXCERPT_CHARS = 600
MAX_WEB_EXCERPT_CHARS = 320
MAX_QUERY_CHARS = 160
MAX_SOURCE_CHARS = 512
MAX_URL_CHARS = 2048
MAX_TITLE_CHARS = 200
MAX_DATE_CHARS = 64
MAX_TURN_SUMMARY_CHARS = 600


_SUMMARY_SIGNAL = re.compile(
    r"\d|%|结论|因此|显示|预计|风险|建议|同比|环比|增长|下降|主要|关键"
)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content)


def _bounded(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _extractive_summary(content: str | None) -> str | None:
    """从最终回答中确定性抽取要点，不生成新事实。"""
    text = str(content or "").strip()
    if not text:
        return None
    if len(text) <= MAX_TURN_SUMMARY_CHARS:
        return text
    units = [
        unit.strip()
        for unit in re.findall(r".*?(?:[\u3002！？!?\uff1b;\n]+|$)", text, flags=re.S)
        if unit.strip()
    ]
    if not units:
        return text[:MAX_TURN_SUMMARY_CHARS]
    ranked = sorted(
        enumerate(units),
        key=lambda item: (
            -(
                3 * bool(re.search(r"\d|%", item[1]))
                + 2 * bool(_SUMMARY_SIGNAL.search(item[1]))
                + int(item[0] in {0, len(units) - 1})
            ),
            item[0],
        ),
    )
    selected_indexes: set[int] = set()
    used = 0
    for index, unit in ranked:
        separator = 1 if selected_indexes else 0
        if used + separator + len(unit) > MAX_TURN_SUMMARY_CHARS:
            continue
        selected_indexes.add(index)
        used += separator + len(unit)
    if not selected_indexes:
        return text[:MAX_TURN_SUMMARY_CHARS]
    return "\n".join(units[index] for index in sorted(selected_indexes))


def turn_id_for_block(block: tuple[Any, ...]) -> str:
    """返回稳定的轮次 ID；无消息 ID 时使用内容指纹。"""
    for message in block:
        if not is_real_human_message(message):
            continue
        message_id = str(getattr(message, "id", None) or "").strip()
        if message_id:
            return message_id
        break
    digest = hashlib.sha256()
    for message in block:
        record = {
            "type": type(message).__name__,
            "content": getattr(message, "content", None),
            "tool_calls": getattr(message, "tool_calls", None),
            "tool_call_id": getattr(message, "tool_call_id", None),
            "name": getattr(message, "name", None),
        }
        digest.update(
            json.dumps(record, ensure_ascii=False, sort_keys=True, default=str).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return f"turn-{digest.hexdigest()[:24]}"


def _final_answer(block: tuple[Any, ...]) -> str | None:
    for message in reversed(block):
        if not isinstance(message, AIMessage) or extract_tool_call_ids(message):
            continue
        content = _message_text(message)
        if content.strip():
            return content
    return None


def _rag_snapshots(
    message: ToolMessage,
    *,
    max_excerpt_chars: int,
) -> tuple[list[EvidenceSnapshot], int]:
    try:
        outcome = decode_retrieval_outcome(message.content)
    except (TypeError, ValueError):
        return [], 0
    if outcome is None or not outcome.ok:
        return [], 0

    snapshots: list[EvidenceSnapshot] = []
    seen: set[tuple[str, str, int | None, str]] = set()
    omitted = max(0, outcome.total_results - len(outcome.evidence))
    tool_message_id = str(
        getattr(message, "id", None)
        or getattr(message, "tool_call_id", None)
        or ""
    )
    for rank, item in enumerate(outcome.evidence, start=1):
        content = item.content.strip()
        if not content:
            continue
        key = (
            item.chunk_id,
            item.source_file,
            item.source_page,
            content,
        )
        if key in seen:
            continue
        seen.add(key)
        excerpt = content[:max_excerpt_chars]
        snapshots.append(
            EvidenceSnapshot(
                kind="rag",
                query=_bounded(outcome.query, MAX_QUERY_CHARS),
                exact_excerpt=excerpt,
                source_file=_bounded(item.source_file, MAX_SOURCE_CHARS),
                source_page=item.source_page,
                chunk_id=_bounded(item.chunk_id, MAX_SOURCE_CHARS),
                rank=rank,
                excerpt_truncated=(
                    item.content_truncated or len(excerpt) < len(content)
                ),
                tool_message_id=tool_message_id,
            )
        )
    return snapshots, omitted


def _web_snapshots(
    message: ToolMessage,
    *,
    max_excerpt_chars: int,
) -> tuple[list[EvidenceSnapshot], int]:
    try:
        payload = decode_tool_result(message.content)
    except (TypeError, ValueError):
        return [], 0
    if payload.get("ok") is not True:
        return [], 0
    data = payload.get("data")
    if not isinstance(data, dict):
        return [], 0
    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        return [], 0

    query = _bounded(payload.get("query"), MAX_QUERY_CHARS)
    meta = payload.get("meta")
    meta = meta if isinstance(meta, dict) else {}
    retrieved_at = _bounded(
        meta.get("retrieved_at") or meta.get("timestamp"), MAX_DATE_CHARS
    )
    tool_message_id = str(
        getattr(message, "id", None)
        or getattr(message, "tool_call_id", None)
        or ""
    )
    snapshots: list[EvidenceSnapshot] = []
    seen: set[tuple[str, str, str]] = set()
    omitted = 0
    for rank, raw in enumerate(raw_results, start=1):
        if not isinstance(raw, dict):
            continue
        content = str(raw.get("content") or "").strip()
        if not content:
            continue
        url = _bounded(raw.get("url"), MAX_URL_CHARS)
        title = _bounded(raw.get("title"), MAX_TITLE_CHARS)
        key = (url, title, content)
        if key in seen:
            continue
        seen.add(key)
        excerpt = content[:max_excerpt_chars]
        snapshots.append(
            EvidenceSnapshot(
                kind="web_search",
                query=query,
                exact_excerpt=excerpt,
                url=url,
                title=title,
                published_at=(
                    _bounded(raw.get("published_date"), MAX_DATE_CHARS) or None
                ),
                retrieved_at=retrieved_at or None,
                rank=rank,
                excerpt_truncated=(
                    len(excerpt) < len(content) or content.endswith("...")
                ),
                tool_message_id=tool_message_id,
            )
        )
    return snapshots, omitted


def _turn_evidence(
    block: tuple[Any, ...],
    *,
    retrieval_tool_names: frozenset[str],
    web_search_tool_names: frozenset[str],
    max_rag_records: int,
    max_web_records: int,
    max_rag_excerpt_chars: int,
    max_web_excerpt_chars: int,
) -> tuple[tuple[EvidenceSnapshot, ...], int]:
    rag: list[EvidenceSnapshot] = []
    web: list[EvidenceSnapshot] = []
    rag_keys: set[tuple[str, str, int | None]] = set()
    web_keys: set[str] = set()
    omitted = 0
    for message in block:
        if not isinstance(message, ToolMessage):
            continue
        if message.name in retrieval_tool_names:
            found, hidden = _rag_snapshots(
                message,
                max_excerpt_chars=max_rag_excerpt_chars,
            )
            omitted += hidden
            for record in found:
                key = (record.chunk_id, record.source_file, record.source_page)
                if key in rag_keys:
                    continue
                if len(rag) >= max_rag_records:
                    omitted += 1
                    continue
                rag_keys.add(key)
                rag.append(record)
            continue
        if message.name in web_search_tool_names:
            found, hidden = _web_snapshots(
                message,
                max_excerpt_chars=max_web_excerpt_chars,
            )
            omitted += hidden
            for record in found:
                key = record.url or f"{record.title}\n{record.exact_excerpt}"
                if key in web_keys:
                    continue
                if len(web) >= max_web_records:
                    omitted += 1
                    continue
                web_keys.add(key)
                web.append(record)
    return tuple([*rag, *web]), omitted


def _apply_global_evidence_limit(
    turns: list[TurnMemory], max_records: int
) -> tuple[tuple[TurnMemory, ...], int]:
    remaining = max(0, max_records)
    bounded_reversed: list[TurnMemory] = []
    for turn in reversed(turns):
        keep = min(len(turn.evidence), remaining)
        evidence = turn.evidence[:keep]
        newly_omitted = len(turn.evidence) - keep
        omitted_count = turn.evidence_omitted_count + newly_omitted
        bounded_reversed.append(
            replace(
                turn,
                evidence=evidence,
                evidence_omitted=omitted_count > 0,
                evidence_omitted_count=omitted_count,
            )
        )
        remaining -= keep
    bounded = tuple(reversed(bounded_reversed))
    return bounded, sum(turn.evidence_omitted_count for turn in bounded)


def materialize_completed_turn_memories(
    messages: list[Any],
    *,
    retrieval_tool_names: frozenset[str],
    web_search_tool_names: frozenset[str],
    max_rag_records_per_turn: int = MAX_TURN_RAG_EVIDENCE,
    max_web_records_per_turn: int = MAX_TURN_WEB_EVIDENCE,
    max_records: int = MAX_CONVERSATION_EVIDENCE,
    max_rag_excerpt_chars: int = MAX_RAG_EXCERPT_CHARS,
    max_web_excerpt_chars: int = MAX_WEB_EXCERPT_CHARS,
) -> tuple[tuple[TurnMemory, ...], int]:
    """重建所有已完成轮次的派生记忆。

    仅从 Checkpoint 中的原始消息读取，不改写任何原消息。
    全局证据容量优先分配给最近轮次。
    """
    if max_rag_records_per_turn < 0 or max_web_records_per_turn < 0:
        raise ValueError("每轮证据上限不能为负数")
    if max_records < 0:
        raise ValueError("全局证据上限不能为负数")
    if max_rag_excerpt_chars < 1 or max_web_excerpt_chars < 1:
        raise ValueError("证据摘录字符上限必须大于 0")

    completed = segment_current_turn(messages).completed
    turns: list[TurnMemory] = []
    for block in segment_turn_blocks(completed):
        user_messages = tuple(
            _message_text(message)
            for message in block
            if is_real_human_message(message) and _message_text(message).strip()
        )
        if not user_messages:
            continue
        evidence, omitted = _turn_evidence(
            block,
            retrieval_tool_names=retrieval_tool_names,
            web_search_tool_names=web_search_tool_names,
            max_rag_records=max_rag_records_per_turn,
            max_web_records=max_web_records_per_turn,
            max_rag_excerpt_chars=max_rag_excerpt_chars,
            max_web_excerpt_chars=max_web_excerpt_chars,
        )
        source_ids = tuple(
            str(message.id)
            for message in block
            if getattr(message, "id", None)
        )
        final_answer = _final_answer(block)
        turns.append(
            TurnMemory(
                turn_id=turn_id_for_block(block),
                user_messages=user_messages,
                final_answer=final_answer,
                summary=_extractive_summary(final_answer),
                evidence=evidence,
                evidence_omitted=omitted > 0,
                evidence_omitted_count=omitted,
                source_message_ids=source_ids,
            )
        )
    return _apply_global_evidence_limit(turns, max_records)


__all__ = [
    "MAX_CONVERSATION_EVIDENCE",
    "MAX_RAG_EXCERPT_CHARS",
    "MAX_TURN_RAG_EVIDENCE",
    "MAX_TURN_SUMMARY_CHARS",
    "MAX_TURN_WEB_EVIDENCE",
    "MAX_WEB_EXCERPT_CHARS",
    "materialize_completed_turn_memories",
    "turn_id_for_block",
]
