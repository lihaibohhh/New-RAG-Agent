"""知识库分阶段评测检索管道。"""
from __future__ import annotations

import time
from typing import Any

from knowledge.contracts import (
    EvaluationCandidate,
    EvaluationRetrievalResult,
    RagDocument,
    RagValidationError,
    RetrievalRequest,
)
from knowledge.ports import EvaluationHybridRetrieverPort, RerankerPort
from knowledge.rag.query.result_builder import build_chunks


class EvaluationRetrievalService:
    """绕过语义查询缓存，返回召回、融合与精排阶段快照。"""

    def __init__(
        self,
        *,
        retriever: EvaluationHybridRetrieverPort,
        reranker: RerankerPort,
        max_content_chars: int,
        rerank_candidates: int,
        production_rerank_top_n: int,
        reranker_batch_size: int,
        device: str,
        source_top_k: int = 10,
        rrf_rank_constant: int = 60,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._max_content_chars = max(200, min(int(max_content_chars), 5000))
        self._rerank_candidates = max(1, min(int(rerank_candidates), 100))
        self._production_rerank_top_n = max(
            1,
            min(int(production_rerank_top_n), 20),
        )
        self._reranker_batch_size = max(1, int(reranker_batch_size))
        self._device = str(device or "unknown")
        self._source_top_k = max(1, int(source_top_k))
        self._rrf_rank_constant = max(1, int(rrf_rank_constant))

    async def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        retrieval_mode: str = "hybrid",
        use_query_cache: bool = False,
    ) -> EvaluationRetrievalResult:
        """执行专用评测检索；不允许查询缓存污染 A/B 结果。"""
        if use_query_cache:
            raise RagValidationError("评测检索管道不允许使用语义查询缓存")
        total_started = time.perf_counter()
        request = RetrievalRequest(
            query=query,
            top_k=top_k,
            filters=filters,
        ).normalized()
        mode = str(retrieval_mode or "hybrid").strip().lower()
        if mode not in {"hybrid", "bm25", "vector"}:
            raise RagValidationError(f"不支持的检索模式: {retrieval_mode}")

        candidate_trace = await self._retriever.retrieve_with_trace(
            request.query,
            filters=request.filters,
            mode=mode,
        )
        candidates = list(candidate_trace.filtered_candidates)
        rerank_input = candidates[: self._rerank_candidates]

        rerank_started = time.perf_counter()
        if rerank_input:
            # 评测管道保留所有通过阈值的排序结果，便于计算
            # Retention/Harm；前 top_k 的顺序与线上精排一致。
            reranked, _top_score = await self._reranker.rerank(
                request.query,
                rerank_input,
                top_n=len(rerank_input),
            )
        else:
            reranked = []
        rerank_elapsed = round(time.perf_counter() - rerank_started, 4)
        final_documents = reranked[: request.top_k]

        stages = {
            "bm25": self._snapshots(candidate_trace.bm25_candidates),
            "vector": self._snapshots(candidate_trace.vector_candidates),
            "fusion": self._snapshots(candidate_trace.fusion_candidates),
            "filtered": self._snapshots(candidate_trace.filtered_candidates),
            "reranker_input": self._snapshots(rerank_input),
            "reranked": self._snapshots(reranked),
            "final": self._snapshots(final_documents),
        }
        timings = {
            **candidate_trace.timings,
            "reranker": rerank_elapsed,
            "evaluation_total": round(time.perf_counter() - total_started, 4),
        }
        return EvaluationRetrievalResult(
            query=request.query,
            retrieval_mode=mode,
            chunks=build_chunks(
                final_documents,
                max_chars=self._max_content_chars,
            ),
            stages=stages,
            timings=timings,
            configuration={
                "query_cache_enabled": False,
                "device": self._device,
                "source_top_k": self._source_top_k,
                "fusion_limit": 2 * self._source_top_k,
                "rrf_rank_constant": self._rrf_rank_constant,
                "rerank_candidates": self._rerank_candidates,
                "production_rerank_top_n": self._production_rerank_top_n,
                "reranker_batch_size": self._reranker_batch_size,
                "final_top_k": request.top_k,
            },
            degraded_sources=candidate_trace.degraded_sources,
        )

    @staticmethod
    def _snapshots(
        documents: list[RagDocument] | tuple[RagDocument, ...],
    ) -> tuple[EvaluationCandidate, ...]:
        return tuple(
            EvaluationCandidate(
                rank=rank,
                chunk_id=document.metadata.chunk_id,
                source_file=document.metadata.resolved_source_file(),
                source_page=document.metadata.source_page,
                content_chars=len(document.content or ""),
                doc_type=document.metadata.doc_type,
                industry=document.metadata.industry,
            )
            for rank, document in enumerate(documents, 1)
        )


__all__ = ["EvaluationRetrievalService"]
