"""RAG composition types and factories."""
from typing import TYPE_CHECKING

from react_agent.rag.runtime.container import (
    RagRuntime,
    create_rag_runtime,
)
from react_agent.rag.runtime.access import (
    create_configured_rag_runtime,
)
from react_agent.rag.runtime.remote import RemoteRagRuntime
from react_agent.rag.runtime_ports import (
    AgentRagRuntimePort,
    EvaluationRetrievalServicePort,
    IngestionServicePort,
    RagAdminServicePort,
    RagOperationsPort,
    RagRuntimePort,
    RetrievalServicePort,
)

if TYPE_CHECKING:
    from react_agent.configuration.settings import RagProfile
def get_rag_runtime_profile(requested: str | None = None) -> "RagProfile":
    """延迟解析设备档位，避免导入 Runtime 时立即导入 Torch。"""
    from react_agent.rag.runtime.device import get_rag_runtime_profile as resolve

    return resolve(requested)

__all__ = [
    "RagRuntime",
    "AgentRagRuntimePort",
    "EvaluationRetrievalServicePort",
    "IngestionServicePort",
    "RagAdminServicePort",
    "RagOperationsPort",
    "RagRuntimePort",
    "RetrievalServicePort",
    "RemoteRagRuntime",
    "create_configured_rag_runtime",
    "create_rag_runtime",
    "get_rag_runtime_profile",
]
