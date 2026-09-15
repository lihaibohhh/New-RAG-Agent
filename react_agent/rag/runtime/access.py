"""选择本地 RAG 所有者或远程 Knowledge Service 客户端。"""
from __future__ import annotations

import os

from react_agent.rag.runtime.container import create_rag_runtime
from react_agent.rag.runtime.remote import RemoteRagRuntime
from react_agent.rag.runtime_ports import RagRuntimePort


_VALID_RUNTIME_MODES = frozenset({"local", "remote"})


def create_configured_rag_runtime() -> RagRuntimePort:
    """按显式模式创建 Runtime；远程模式缺少 URL 时立即失败。"""
    mode = os.getenv("RAG_RUNTIME_MODE", "remote").strip().lower()
    if mode not in _VALID_RUNTIME_MODES:
        raise ValueError("RAG_RUNTIME_MODE 仅支持 remote 或 local")
    if mode == "local":
        return create_rag_runtime()

    base_url = os.getenv("KNOWLEDGE_SERVICE_URL", "").strip()
    if not base_url:
        raise RuntimeError(
            "RAG_RUNTIME_MODE=remote 时必须配置 KNOWLEDGE_SERVICE_URL；"
            "只有 Knowledge Service 或显式离线任务可以使用 local 模式"
        )
    try:
        timeout = float(os.getenv("KNOWLEDGE_SERVICE_TIMEOUT", "150"))
    except ValueError:
        timeout = 150.0
    return RemoteRagRuntime(
        base_url=base_url,
        api_key=os.getenv("KNOWLEDGE_SERVICE_API_KEY", "").strip(),
        timeout=max(1.0, timeout),
    )


__all__ = [
    "RagRuntimePort",
    "create_configured_rag_runtime",
]
