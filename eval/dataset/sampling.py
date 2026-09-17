"""Chunk 清洗、分层抽样和证据单元构造。"""

import random
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

from langchain_core.documents import Document

from eval.dataset.models import EvidenceUnit

_NOISE_PATTERNS = re.compile(
    r"免责声明|分析师声明|评级说明|风险提示.*本报告|版权所有.*禁止|执业编号|"
    r"请务必阅读正文之后的免责|本报告仅供.*客户使用|证券投资咨询业务资格",
    re.S,
)


def get_industry(chunk: Document) -> str:
    return str(chunk.metadata.get("industry") or "未知").strip() or "未知"


def get_relative_source(chunk: Document, data_dir: Path) -> str:
    source_file = str(chunk.metadata.get("source_file") or "").strip()
    if source_file:
        return source_file.replace("\\", "/")
    source = chunk.metadata.get("source", "unknown")
    try:
        return str(Path(source).relative_to(data_dir)).replace("\\", "/")
    except ValueError:
        return Path(source).name


def get_document_type(chunk: Document) -> str:
    raw = str(chunk.metadata.get("doc_type") or "").casefold()
    if "table" in raw or re.search(r"(?m)^\s*\|.+\|\s*$", chunk.page_content):
        return "table"
    return "text"


def get_parser(chunk: Document) -> str:
    return str(chunk.metadata.get("parser") or "unknown").strip() or "unknown"


def get_page(chunk: Document) -> int | None:
    value = chunk.metadata.get("page", chunk.metadata.get("source_page"))
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def is_noise_chunk(text: str) -> bool:
    return bool(_NOISE_PATTERNS.search(text[:600]))


def stratified_sample(
    chunks: list[Document], max_chunks: int, seed: int
) -> list[Document]:
    """按行业、内容类型和解析器分层，并在层内轮转来源文档。"""
    rng = random.Random(seed)
    valid = [
        chunk
        for chunk in chunks
        if len(chunk.page_content.strip()) >= 100 and not is_noise_chunk(chunk.page_content)
    ]
    if len(chunks) != len(valid):
        print(f"[generator] 预过滤：跳过 {len(chunks) - len(valid)} 个噪声/过短 chunks")
    if max_chunks <= 0 or not valid:
        return []

    strata: dict[tuple[str, str, str], dict[str, list[Document]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for chunk in valid:
        source = str(
            chunk.metadata.get("source_file")
            or chunk.metadata.get("source")
            or chunk.metadata.get("chunk_id")
            or "unknown"
        ).replace("\\", "/")
        key = (get_industry(chunk), get_document_type(chunk), get_parser(chunk))
        strata[key][source].append(chunk)

    queues: dict[tuple[str, str, str], deque[Document]] = {}
    for key, by_source in strata.items():
        source_names = list(by_source)
        rng.shuffle(source_names)
        for documents in by_source.values():
            rng.shuffle(documents)
        queue: deque[Document] = deque()
        while source_names:
            remaining = []
            for source in source_names:
                documents = by_source[source]
                if documents:
                    queue.append(documents.pop())
                if documents:
                    remaining.append(source)
            source_names = remaining
        queues[key] = queue

    active = list(queues)
    rng.shuffle(active)
    sampled: list[Document] = []
    while active and len(sampled) < min(max_chunks, len(valid)):
        remaining_keys = []
        for key in active:
            queue = queues[key]
            if queue and len(sampled) < max_chunks:
                sampled.append(queue.popleft())
            if queue:
                remaining_keys.append(key)
        active = remaining_keys

    distribution = Counter(
        (get_industry(chunk), get_document_type(chunk), get_parser(chunk))
        for chunk in sampled
    )
    for (industry, document_type, parser), count in sorted(distribution.items()):
        print(f"  分层「{industry}/{document_type}/{parser}」：抽 {count} chunks")
    document_count = len(
        {
            str(chunk.metadata.get("source_file") or chunk.metadata.get("source"))
            for chunk in sampled
        }
    )
    print(f"[generator] 最终采样 {len(sampled)} 个 chunks，覆盖 {document_count} 份文档")
    return sampled


def build_evidence_units(
    chunks: list[Document],
    *,
    multi_chunk_ratio: float,
    seed: int,
    corpus: list[Document] | None = None,
) -> list[EvidenceUnit]:
    """将同一文档相邻页的 Chunk 组成多证据单元。"""
    ratio = max(0.0, min(float(multi_chunk_ratio), 0.5))
    available = [
        chunk
        for chunk in (corpus or chunks)
        if len(chunk.page_content.strip()) >= 100 and not is_noise_chunk(chunk.page_content)
    ]
    by_source: dict[str, list[Document]] = defaultdict(list)
    for chunk in available:
        source = str(
            chunk.metadata.get("source_file") or chunk.metadata.get("source") or "unknown"
        ).replace("\\", "/")
        by_source[source].append(chunk)

    candidates: list[tuple[Document, Document]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for anchor in chunks:
        source = str(
            anchor.metadata.get("source_file") or anchor.metadata.get("source") or "unknown"
        ).replace("\\", "/")
        ordered = sorted(
            by_source.get(source, []),
            key=lambda item: (
                get_page(item) if get_page(item) is not None else 10**9,
                str(item.metadata.get("chunk_id") or ""),
            ),
        )
        anchor_id = str(anchor.metadata.get("chunk_id") or id(anchor))
        anchor_index = next(
            (
                index
                for index, item in enumerate(ordered)
                if str(item.metadata.get("chunk_id") or id(item)) == anchor_id
            ),
            None,
        )
        if anchor_index is None:
            continue
        for neighbor_index in (anchor_index - 1, anchor_index + 1):
            if not 0 <= neighbor_index < len(ordered):
                continue
            neighbor = ordered[neighbor_index]
            neighbor_id = str(neighbor.metadata.get("chunk_id") or id(neighbor))
            pages = sorted(
                page for page in (get_page(anchor), get_page(neighbor)) if page is not None
            )
            pair_key = tuple(sorted((anchor_id, neighbor_id)))
            if len(pages) != 2 or pages[1] - pages[0] > 1 or pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            pair = sorted(
                (anchor, neighbor),
                key=lambda item: (
                    get_page(item) if get_page(item) is not None else 10**9,
                    str(item.metadata.get("chunk_id") or ""),
                ),
            )
            candidates.append((pair[0], pair[1]))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    target = min(round(len(chunks) * ratio), len(candidates))
    used: set[str] = set()
    units: list[EvidenceUnit] = []
    for left, right in candidates:
        if len(units) >= target:
            break
        identifiers = {
            str(left.metadata.get("chunk_id") or id(left)),
            str(right.metadata.get("chunk_id") or id(right)),
        }
        if used.intersection(identifiers):
            continue
        units.append(EvidenceUnit((left, right), "multi_chunk"))
        used.update(identifiers)

    units.extend(
        EvidenceUnit((chunk,), "single_chunk")
        for chunk in chunks
        if str(chunk.metadata.get("chunk_id") or id(chunk)) not in used
    )
    rng.shuffle(units)
    multi_count = sum(unit.scope == "multi_chunk" for unit in units)
    print(
        f"[generator] 证据单元：{len(units)} 组"
        f"（multi={multi_count}, single={len(units) - multi_count}）"
    )
    return units


_get_industry = get_industry
_get_rel_src = get_relative_source
_get_doc_type = get_document_type
_get_parser = get_parser
_get_page = get_page
_is_noise_chunk = is_noise_chunk
