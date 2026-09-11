"""通用工具包入口；LLM 实现按需加载。"""
from __future__ import annotations

import importlib
from typing import Any


_LAZY_SUBMODULES = {
    "embedder",
    "llm",
    "redis_client",
    "time_utils",
    "timer_logger",
    "token_utils",
    "tool_helpers",
    "tool_utils",
    "usage_logger",
}


def __getattr__(name: str) -> Any:
    if name in {"ChatModelSettings", "load_chat_model"}:
        from react_agent.utils import llm

        return getattr(llm, name)
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"react_agent.utils.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)


__all__ = ["ChatModelSettings", "load_chat_model"]
