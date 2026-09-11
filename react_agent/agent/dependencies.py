"""Runtime dependencies injected into the Agent graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

from react_agent.agent.context import AgentContext


@dataclass(frozen=True)
class AgentDependencies:
    """Concrete runtime capabilities supplied by the Composition Root."""

    context: AgentContext
    model_provider: Callable[[], Any]
    tools: Tuple[Any, ...]

    def get_model(self) -> Any:
        """Resolve the lazily constructed chat model selected by Runtime."""
        return self.model_provider()

__all__ = ["AgentDependencies"]
