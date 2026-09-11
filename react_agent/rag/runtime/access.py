"""选择本地 RAG 所有者或远程 Knowledge Service 客户端。"""
from __future__ import annotations

import os
from typing import Any, Protocol

from react_agent.rag.runtime.container import RagRuntime, create_rag_runtime
from react_agent.rag.runtime.remote import RemoteRagRuntime


class RagRuntimePort(Protocol):
    operations: Any

    def get_retrieval_service(self) -> Any: ...

    def get_admin_service(self) -> Any: ...

    def get_ingestion_service(self) -> Any: ...

    async def close(self) -> None: ...


def create_configured_rag_runtime() -> RagRuntime | RemoteRagRuntime:
    """配置了服务 URL 时禁止当前进程直接打开 Chroma。"""
    base_url = os.getenv("KNOWLEDGE_SERVICE_URL", "").strip()
    if not base_url:
        return create_rag_runtime()
    try:
        timeout = float(os.getenv("KNOWLEDGE_SERVICE_TIMEOUT", "150"))
    except ValueError:
        timeout = 150.0
    return RemoteRagRuntime(
        base_url=base_url,
        api_key=os.getenv("KNOWLEDGE_SERVICE_API_KEY", "").strip(),
        timeout=max(1.0, timeout),
    )


__all__ = ["RagRuntimePort", "create_configured_rag_runtime"]
