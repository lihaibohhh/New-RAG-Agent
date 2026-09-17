"""评测数据集的 Gold Chunk 与来源页码标签解析。"""

from collections.abc import Mapping, Sequence
from typing import Any


def values(value: Any) -> list[Any]:
    """将单值或序列规范化为列表。"""
    if value in (None, ""):
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def normalize_source(value: Any) -> str:
    """规范化来源路径以便跨平台比较。"""
    source = str(value or "").strip().replace("\\", "/").casefold()
    while source.startswith("./"):
        source = source[2:]
    return source


def same_source(left: Any, right: Any) -> bool:
    """允许完整路径与相对路径的稳定后缀匹配。"""
    first = normalize_source(left)
    second = normalize_source(right)
    if not first or not second:
        return False
    return (
        first == second
        or first.endswith("/" + second)
        or second.endswith("/" + first)
    )


def page_number(value: Any) -> int | None:
    """将合法的非负页码规范化为整数。"""
    if value in (None, ""):
        return None
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if page >= 0 else None


def gold_chunk_ids(record: Mapping[str, Any]) -> tuple[str, ...]:
    """读取 Gold Chunk 标签，并兼容旧的单个 ``chunk_id`` 字段。"""
    raw = values(record.get("gold_chunk_ids"))
    if not raw and record.get("chunk_id") not in (None, ""):
        raw = [record["chunk_id"]]
    return tuple(dict.fromkeys(str(item).strip() for item in raw if str(item).strip()))


def gold_source_pages(record: Mapping[str, Any]) -> tuple[tuple[str, int | None], ...]:
    """读取来源页标签，并兼容顶层 ``source_file/page`` 字段。"""
    labels: list[tuple[str, int | None]] = []
    for raw in values(record.get("gold_sources")):
        if not isinstance(raw, Mapping):
            continue
        source = str(
            raw.get("source_file") or raw.get("file") or raw.get("source") or ""
        ).strip()
        if not source:
            continue
        pages = values(raw.get("pages")) or values(raw.get("page")) or [None]
        labels.extend((source, page_number(page)) for page in pages)

    if not labels:
        source = str(record.get("source_file") or "").strip()
        if source:
            labels.append((source, page_number(record.get("page"))))

    return tuple(dict.fromkeys(labels))


__all__ = [
    "gold_chunk_ids",
    "gold_source_pages",
    "normalize_source",
    "page_number",
    "same_source",
    "values",
]
