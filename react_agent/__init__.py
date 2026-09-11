"""React Agent package root with lazy subpackage loading."""
from __future__ import annotations

import importlib
from typing import Any


_LAZY_SUBPACKAGES = {
    "agent",
    "conversations",
    "core",
    "mcp_server",
    "rag",
    "runtime",
    "tools",
    "utils",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBPACKAGES:
        module = importlib.import_module(f"react_agent.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)


__all__: list[str] = []
