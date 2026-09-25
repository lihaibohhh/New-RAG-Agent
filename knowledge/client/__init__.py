"""Public client for the standalone Knowledge Service."""

from knowledge.client.access import (
    create_configured_ingestion_runtime,
    create_configured_rag_runtime,
    create_remote_ingestion_runtime,
    create_remote_rag_runtime,
    load_client_config,
)
from knowledge.client.config import KnowledgeClientConfig
from knowledge.client.remote import (
    KnowledgeServiceClient,
    RemoteEvaluationRetrievalService,
    RemoteIngestionRuntime,
    RemoteIngestionService,
    RemoteRagAdminService,
    RemoteRagOperations,
    RemoteRagRuntime,
)

__all__ = [
    "KnowledgeClientConfig",
    "KnowledgeServiceClient",
    "RemoteEvaluationRetrievalService",
    "RemoteIngestionService",
    "RemoteIngestionRuntime",
    "RemoteRagAdminService",
    "RemoteRagOperations",
    "RemoteRagRuntime",
    "create_configured_ingestion_runtime",
    "create_configured_rag_runtime",
    "create_remote_ingestion_runtime",
    "create_remote_rag_runtime",
    "load_client_config",
]
