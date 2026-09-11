"""将旧版 source/page 评测集尽可能自动补全为 v2 gold 标签。"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from react_agent.rag.contracts import StoredChunk


logger = logging.getLogger(__name__)


def _source(value: Any) -> str:
    source = str(value or "").strip().replace("\\", "/").casefold()
    while source.startswith("./"):
        source = source[2:]
    return source


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _case_id(record: dict[str, Any]) -> str:
    existing = str(record.get("case_id") or "").strip()
    if existing:
        return existing
    identity = "\0".join(
        (
            str(record.get("source_file") or ""),
            str(record.get("page") or ""),
            str(record.get("question") or ""),
        )
    )
    return "rag_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def enrich_records(
    records: list[dict[str, Any]],
    chunks: list[StoredChunk],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """按来源页和已保存的 chunk 文本前缀进行保守匹配。"""
    by_source_page: dict[tuple[str, int | None], list[StoredChunk]] = defaultdict(list)
    for chunk in chunks:
        key = (_source(chunk.metadata.resolved_source_file()), chunk.metadata.source_page)
        by_source_page[key].append(chunk)

    enriched: list[dict[str, Any]] = []
    stats = {"already_labelled": 0, "matched": 0, "ambiguous": 0, "unresolved": 0}
    for original in records:
        record = dict(original)
        record["schema_version"] = 2
        record["case_id"] = _case_id(record)
        record.setdefault("answerable", True)
        record.setdefault("category", "unclassified")

        source_file = str(record.get("source_file") or "").strip()
        try:
            page = int(record["page"]) if record.get("page") not in (None, "") else None
        except (TypeError, ValueError):
            page = None
        if source_file:
            record["gold_sources"] = [
                {"source_file": source_file, "pages": [page] if page is not None else []}
            ]

        existing_ids = [
            str(value).strip()
            for value in record.get("gold_chunk_ids", [])
            if str(value).strip()
        ]
        if existing_ids:
            record["gold_chunk_ids"] = list(dict.fromkeys(existing_ids))
            stats["already_labelled"] += 1
            enriched.append(record)
            continue

        normalised_source = _source(source_file)
        candidates = by_source_page.get((normalised_source, page), [])
        if not candidates and normalised_source:
            candidates = [
                chunk
                for (candidate_source, candidate_page), source_chunks in by_source_page.items()
                if candidate_page == page
                and (
                    candidate_source.endswith("/" + normalised_source)
                    or normalised_source.endswith("/" + candidate_source)
                )
                for chunk in source_chunks
            ]
        preview = _text(record.get("chunk_text"))
        if preview:
            matches = [
                chunk
                for chunk in candidates
                if _text(chunk.content).startswith(preview)
                or preview.startswith(_text(chunk.content))
            ]
        else:
            matches = candidates if len(candidates) == 1 else []

        if len(matches) == 1:
            record["gold_chunk_ids"] = [matches[0].chunk_id]
            stats["matched"] += 1
        else:
            record["gold_chunk_ids"] = []
            stats["ambiguous" if len(matches) > 1 else "unresolved"] += 1
        enriched.append(record)

    return enriched, stats


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load_chunks_from_sqlite(chroma_dir: Path) -> list[StoredChunk]:
    """Chroma API 不可读 HNSW 时，只读提取文档与 metadata。"""
    database = (chroma_dir / "chroma.sqlite3").resolve()
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT e.id, e.embedding_id, m.key,
                   m.string_value, m.int_value, m.float_value, m.bool_value
            FROM embeddings AS e
            JOIN segments AS s ON s.id = e.segment_id
            JOIN collections AS c ON c.id = s.collection
            JOIN embedding_metadata AS m ON m.id = e.id
            WHERE c.name = 'langchain'
            ORDER BY e.id
            """
        )
        records: dict[int, dict[str, Any]] = {}
        for row_id, embedding_id, key, text, integer, number, boolean in rows:
            record = records.setdefault(
                int(row_id),
                {"chunk_id": str(embedding_id), "content": "", "metadata": {}},
            )
            if text is not None:
                value: Any = text
            elif integer is not None:
                value = integer
            elif number is not None:
                value = number
            else:
                value = bool(boolean) if boolean is not None else None
            if key == "chroma:document":
                record["content"] = str(value or "")
            else:
                record["metadata"][str(key)] = value
    finally:
        connection.close()

    return [
        StoredChunk(
            chunk_id=record["chunk_id"],
            content=record["content"],
            metadata=record["metadata"],
        )
        for record in records.values()
        if record["content"]
    ]


def _load_chunks_from_api() -> list[StoredChunk]:
    """通过 Knowledge Service 分页读取 chunks，不接触本地索引文件。"""
    from react_agent.rag.runtime import create_configured_rag_runtime

    async def read() -> list[StoredChunk]:
        runtime = create_configured_rag_runtime()
        try:
            return list(await runtime.get_admin_service().read_chunks())
        finally:
            await runtime.close()

    return asyncio.run(read())


def load_chunks(
    chroma_dir: Path,
    *,
    sqlite_only: bool = False,
    source: str = "auto",
) -> list[StoredChunk]:
    """优先读取 Knowledge Service；无服务配置时保留本地迁移兼容路径。"""
    if source not in {"auto", "api", "local", "sqlite"}:
        raise ValueError(f"未知 chunk 来源：{source}")
    if sqlite_only:
        source = "sqlite"

    service_configured = bool(os.getenv("KNOWLEDGE_SERVICE_URL", "").strip())
    if source == "api" or (source == "auto" and service_configured):
        if not service_configured:
            raise RuntimeError(
                "--source api 需要配置 KNOWLEDGE_SERVICE_URL 和相应 API Key"
            )
        logger.info("通过 Knowledge Service API 读取评测补标语料")
        return _load_chunks_from_api()

    if source == "sqlite":
        return _load_chunks_from_sqlite(chroma_dir)

    from react_agent.rag.runtime.offline import create_admin_service

    try:
        return list(
            create_admin_service(chroma_dir=str(chroma_dir)).read_chunks_sync()
        )
    except Exception as exc:
        logger.warning(
            "Chroma API 读取失败，评测补标改用只读 SQLite：%s: %s",
            type(exc).__name__,
            exc,
        )
        return _load_chunks_from_sqlite(chroma_dir)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="自动补全旧版 RAG 评测标签")
    parser.add_argument("--dataset", default=str(root / "dataset" / "eval_dataset.jsonl"))
    parser.add_argument("--output", default=str(root / "dataset" / "eval_dataset_v2.jsonl"))
    parser.add_argument("--chroma-dir", default=None)
    parser.add_argument(
        "--source",
        choices=("auto", "api", "local", "sqlite"),
        default="auto",
        help="chunk 来源；auto 在配置服务 URL 时优先 API，否则使用本地兼容路径",
    )
    parser.add_argument(
        "--sqlite-only",
        action="store_true",
        help="不初始化 Chroma/HNSW，仅以只读模式读取 chroma.sqlite3",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"输出已存在，拒绝覆盖：{output}；如确认覆盖请传 --overwrite")

    use_api = args.source == "api" or (
        args.source == "auto"
        and bool(os.getenv("KNOWLEDGE_SERVICE_URL", "").strip())
    )
    if use_api:
        # API 模式不会读取该路径；占位值避免加载本地 RAG 配置和索引。
        chroma_dir = Path(".")
    elif args.chroma_dir:
        chroma_dir = Path(args.chroma_dir)
    else:
        from react_agent.core.config import settings

        chroma_dir = Path(settings.tools.vector_store.CHROMA_DB_PATH)
    chunks = load_chunks(
        chroma_dir,
        sqlite_only=args.sqlite_only,
        source=args.source,
    )
    records = _load_jsonl(Path(args.dataset))
    enriched, stats = enrich_records(records, chunks)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in enriched:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps({"output": str(output), "records": len(enriched), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["enrich_records", "load_chunks"]
