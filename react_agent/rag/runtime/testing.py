"""RAG Runtime testing helpers."""
from __future__ import annotations

from react_agent.rag.runtime.container import RagRuntime


def reset_runtime_state(runtime: RagRuntime) -> None:
    """Reset the operational state of one explicit test runtime."""
    runtime.operations.reset_for_testing()


__all__ = ["reset_runtime_state"]
