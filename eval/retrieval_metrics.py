"""不依赖 LLM 的确定性检索指标。"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


RETRIEVAL_METRIC_KEYS = (
    "hit_at_k",
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "source_page_recall",
    "no_answer_accuracy",
)


def _values(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _normalise_source(value: Any) -> str:
    source = str(value or "").strip().replace("\\", "/").casefold()
    while source.startswith("./"):
        source = source[2:]
    return source


def _same_source(left: Any, right: Any) -> bool:
    first = _normalise_source(left)
    second = _normalise_source(right)
    if not first or not second:
        return False
    return (
        first == second
        or first.endswith("/" + second)
        or second.endswith("/" + first)
    )


def _page(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if page >= 0 else None


def gold_chunk_ids(record: Mapping[str, Any]) -> tuple[str, ...]:
    """读取 v2 标签，并兼容单个 ``chunk_id`` 字段。"""
    raw = _values(record.get("gold_chunk_ids"))
    if not raw and record.get("chunk_id") not in (None, ""):
        raw = [record["chunk_id"]]
    return tuple(dict.fromkeys(str(item).strip() for item in raw if str(item).strip()))


def gold_source_pages(record: Mapping[str, Any]) -> tuple[tuple[str, int | None], ...]:
    """读取来源页标签；旧数据集回退到 source_file/page。"""
    labels: list[tuple[str, int | None]] = []
    raw_sources = _values(record.get("gold_sources"))
    for raw in raw_sources:
        if not isinstance(raw, Mapping):
            continue
        source = str(
            raw.get("source_file") or raw.get("file") or raw.get("source") or ""
        ).strip()
        if not source:
            continue
        pages = _values(raw.get("pages")) or _values(raw.get("page")) or [None]
        labels.extend((source, _page(page)) for page in pages)

    if not labels:
        source = str(record.get("source_file") or "").strip()
        if source:
            labels.append((source, _page(record.get("page"))))

    return tuple(dict.fromkeys(labels))


def _retrieved_chunk_id(item: Mapping[str, Any]) -> str:
    return str(item.get("chunk_id") or "").strip()


def _retrieved_source_page(item: Mapping[str, Any]) -> tuple[str, int | None]:
    return (
        str(item.get("source_file") or item.get("file") or item.get("source") or ""),
        _page(item.get("source_page") if "source_page" in item else item.get("page")),
    )


def _source_page_matches(
    retrieved: tuple[str, int | None],
    gold: tuple[str, int | None],
) -> bool:
    retrieved_source, retrieved_page = retrieved
    gold_source, gold_page = gold
    return _same_source(retrieved_source, gold_source) and (
        gold_page is None or retrieved_page == gold_page
    )


def evaluate_retrieval(
    record: Mapping[str, Any],
    retrieved: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    """计算单条问题的二元相关性排名指标。"""
    k = max(1, int(top_k))
    ranked = list(retrieved)[:k]
    chunk_labels = gold_chunk_ids(record)
    source_labels = gold_source_pages(record)

    if record.get("answerable") is False:
        return {
            "retrieval_evaluable": False,
            "retrieval_label_mode": "no_answer",
            "gold_label_count": 0,
            "retrieved_count": len(ranked),
            "retrieval_top_k": k,
            "hit_at_k": None,
            "precision_at_k": None,
            "recall_at_k": None,
            "mrr": None,
            "ndcg_at_k": None,
            "source_page_recall": None,
            "no_answer_accuracy": 1.0 if not ranked else 0.0,
        }

    if chunk_labels:
        gold_ids = set(chunk_labels)
        relevant = [_retrieved_chunk_id(item) in gold_ids for item in ranked]
        matched_count = len(
            {_retrieved_chunk_id(item) for item in ranked if _retrieved_chunk_id(item) in gold_ids}
        )
        gold_count = len(gold_ids)
        label_mode = "chunk_id"
    elif source_labels:
        relevant = [
            any(
                _source_page_matches(_retrieved_source_page(item), gold)
                for gold in source_labels
            )
            for item in ranked
        ]
        matched_count = sum(
            any(
                _source_page_matches(_retrieved_source_page(item), gold)
                for item in ranked
            )
            for gold in source_labels
        )
        gold_count = len(source_labels)
        label_mode = "source_page_fallback"
    else:
        return {
            "retrieval_evaluable": False,
            "retrieval_label_mode": "missing",
            "gold_label_count": 0,
            "retrieved_count": len(ranked),
            "retrieval_top_k": k,
            **{key: None for key in RETRIEVAL_METRIC_KEYS},
        }

    relevant_count = sum(relevant)
    first_rank = next((rank for rank, hit in enumerate(relevant, 1) if hit), None)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, hit in enumerate(relevant, 1)
        if hit
    )
    ideal_hits = min(gold_count, k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    source_page_recall = None
    if source_labels:
        source_matches = sum(
            any(
                _source_page_matches(_retrieved_source_page(item), gold)
                for item in ranked
            )
            for gold in source_labels
        )
        source_page_recall = source_matches / len(source_labels)

    return {
        "retrieval_evaluable": True,
        "retrieval_label_mode": label_mode,
        "gold_label_count": gold_count,
        "retrieved_count": len(ranked),
        "retrieval_top_k": k,
        "hit_at_k": 1.0 if relevant_count else 0.0,
        "precision_at_k": relevant_count / k,
        "recall_at_k": matched_count / gold_count,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else 0.0,
        "source_page_recall": source_page_recall,
        "no_answer_accuracy": None,
    }


def aggregate_retrieval_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """聚合确定性指标，同时保留实际参与计算的样本数。"""
    summary: dict[str, Any] = {
        "retrieval_evaluable_count": sum(
            bool(record.get("retrieval_evaluable")) for record in records
        ),
        "retrieval_missing_label_count": sum(
            record.get("retrieval_label_mode") == "missing" for record in records
        ),
        "no_answer_count": sum(
            record.get("retrieval_label_mode") == "no_answer" for record in records
        ),
    }
    for key in RETRIEVAL_METRIC_KEYS:
        values = [
            float(record[key])
            for record in records
            if record.get(key) is not None
        ]
        summary[key] = round(sum(values) / len(values), 4) if values else None
    return summary


__all__ = [
    "RETRIEVAL_METRIC_KEYS",
    "aggregate_retrieval_metrics",
    "evaluate_retrieval",
    "gold_chunk_ids",
    "gold_source_pages",
]
