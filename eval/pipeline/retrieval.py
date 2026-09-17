"""评测管道的检索阶段。"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from eval.pipeline.metrics.retrieval import evaluate_retrieval

logger = logging.getLogger(__name__)


async def retrieve_for_one(
    record: dict,
    top_n: int,
    retrieval_service: Any,
    use_query_cache: bool = False,
    retrieval_mode: str = "hybrid",
    debug: bool = False,
) -> dict:
    """通过统一检索服务检索一条样本并保存阶段 Trace。"""
    question = record["question"]
    try:
        result = await retrieval_service.search(
            question,
            top_k=top_n,
            use_query_cache=use_query_cache,
            retrieval_mode=retrieval_mode,
        )
        if debug:
            traced_stages = getattr(result, "stages", {})
            logger.info(
                "[DEBUG] 问题: %s...\n  stage=%s, cache_hit=%s, stages=%s, chunks=%s",
                question[:40],
                getattr(result, "stage", "evaluation_trace"),
                getattr(result, "cache_hit", False),
                {key: len(value) for key, value in traced_stages.items()},
                len(result.chunks),
            )

        contexts: list[str] = []
        sources: list[dict] = []
        retrieved_items: list[dict] = []
        for rank, chunk in enumerate(result.chunks, 1):
            page = chunk.source_page
            source_file = chunk.source_file
            prefix = (
                f"[来源：{Path(source_file).name} 第 {page} 页] "
                if source_file and page is not None
                else ""
            )
            contexts.append(prefix + chunk.content)
            sources.append(
                {"file": source_file, "page": page, "chunk_id": chunk.chunk_id}
            )
            retrieved_items.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.chunk_id,
                    "source_file": source_file,
                    "source_page": page,
                    "score": chunk.score,
                }
            )

        metrics = evaluate_retrieval(record, retrieved_items, top_k=top_n)
        retrieval_trace = {
            name: [
                {
                    "rank": item.rank,
                    "chunk_id": item.chunk_id,
                    "source_file": item.source_file,
                    "source_page": item.source_page,
                    "content_chars": item.content_chars,
                    "doc_type": item.doc_type,
                    "industry": item.industry,
                }
                for item in candidates
            ]
            for name, candidates in getattr(result, "stages", {}).items()
        }
        return {
            **record,
            "contexts": contexts,
            "sources": sources,
            "retrieved_items": retrieved_items,
            "retrieve_ok": True,
            "retrieval_stage": getattr(result, "stage", "evaluation_trace"),
            "retrieval_cache_hit": getattr(result, "cache_hit", False),
            "retrieval_timings": dict(result.timings),
            "retrieval_trace": retrieval_trace,
            "retrieval_configuration": dict(getattr(result, "configuration", {})),
            "retrieval_degraded_sources": list(
                getattr(result, "degraded_sources", ())
            ),
            **metrics,
        }
    except Exception as exc:
        logger.warning("检索失败 [%s...]: %s", question[:30], exc, exc_info=True)
        return {
            **record,
            "contexts": [],
            "sources": [],
            "retrieved_items": [],
            "retrieve_ok": False,
            **evaluate_retrieval(record, [], top_k=top_n),
        }


async def batch_retrieve(
    records: list[dict],
    top_n: int,
    retrieval_service: Any,
    concurrency: int = 8,
    debug_first_n: int = 0,
    use_query_cache: bool = False,
    retrieval_mode: str = "hybrid",
) -> list[dict]:
    """并发检索所有问题，同时保持输入顺序。"""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict | None] = [None] * len(records)
    done_count = 0

    async def bounded_retrieve(index: int, record: dict) -> None:
        nonlocal done_count
        async with semaphore:
            results[index] = await retrieve_for_one(
                record,
                top_n,
                retrieval_service,
                use_query_cache=use_query_cache,
                retrieval_mode=retrieval_mode,
                debug=index < debug_first_n,
            )
        done_count += 1
        if done_count % 10 == 0 or done_count == len(records):
            logger.info("  检索进度：%s/%s", done_count, len(records))

    await asyncio.gather(
        *(bounded_retrieve(index, record) for index, record in enumerate(records))
    )
    return [result for result in results if result is not None]
