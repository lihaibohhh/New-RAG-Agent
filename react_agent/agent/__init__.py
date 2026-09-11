"""Agent orchestration public API."""
from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "AgentService":
        from react_agent.agent.service import AgentService

        return AgentService
    if name == "AgentDependencies":
        from react_agent.agent.dependencies import AgentDependencies

        return AgentDependencies
    if name == "AgentContext":
        from react_agent.agent.context import AgentContext

        return AgentContext
    if name in {"State", "InputState"}:
        from react_agent.agent.state import InputState, State

        return {"State": State, "InputState": InputState}[name]
    if name in {"build_base_graph", "compile_agent_graph"}:
        from react_agent.agent.graph import build_base_graph, compile_agent_graph

        return {
            "build_base_graph": build_base_graph,
            "compile_agent_graph": compile_agent_graph,
        }[name]
    raise AttributeError(name)


__all__ = [
    "AgentContext",
    "AgentDependencies",
    "AgentService",
    "InputState",
    "State",
    "build_base_graph",
    "compile_agent_graph",
]
