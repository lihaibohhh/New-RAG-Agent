from __future__ import annotations

import asyncio
import threading

from react_agent.rag.contracts import RagDocument
from react_agent.rag.infrastructure.retrieval.reranker import (
    RerankerProviderAdapter,
    _score_with_gate,
)


class _RecordingReranker:
    def __init__(self) -> None:
        self.batches: list[list[tuple[str, str]]] = []

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.batches.append(list(pairs))
        return [float(int(content.removeprefix("doc-"))) for _, content in pairs]


def _documents(count: int) -> list[RagDocument]:
    return [
        RagDocument(content=f"doc-{index}", metadata={"chunk_id": str(index)})
        for index in range(count)
    ]


def test_score_with_gate_splits_pairs_and_preserves_score_order() -> None:
    reranker = _RecordingReranker()
    pairs = [("query", f"doc-{index}") for index in range(5)]

    scores, _queue_wait, _inference_elapsed, pending, batch_count = (
        _score_with_gate(reranker, pairs, threading.BoundedSemaphore(1), 2)
    )

    assert [len(batch) for batch in reranker.batches] == [2, 2, 1]
    assert scores == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert pending == 1
    assert batch_count == 3


def test_adapter_rerank_keeps_global_sorting_across_micro_batches() -> None:
    model = _RecordingReranker()
    adapter = RerankerProviderAdapter(
        device="cpu",
        inference_concurrency=1,
        batch_size=2,
        threshold=-1.0,
    )
    adapter._model = model

    ranked, top_score = asyncio.run(
        adapter.rerank("query", _documents(5), top_n=3)
    )

    assert [len(batch) for batch in model.batches] == [2, 2, 1]
    assert [document.metadata.chunk_id for document in ranked] == ["4", "3", "2"]
    assert top_score == 4.0
