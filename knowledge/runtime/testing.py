"""RAG Runtime testing helpers."""
from __future__ import annotations

from knowledge.runtime.container import KnowledgeRuntime


def reset_runtime_state(runtime: KnowledgeRuntime) -> None:
    """Reset the operational state of one explicit test runtime."""
    runtime.operations.reset_for_testing()


__all__ = ["reset_runtime_state"]
