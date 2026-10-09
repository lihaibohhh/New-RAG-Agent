"""从统一配置创建远程 Knowledge 业务服务。"""

from __future__ import annotations

from collections.abc import Mapping

from knowledge.client.http import HttpIngestionService, HttpRagService
from knowledge.services import IngestionService, RagService
from knowledge.settings import (
    KnowledgeClientSettings,
    load_client_settings,
)


def create_rag_service(config: KnowledgeClientSettings) -> HttpRagService:
    return HttpRagService(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.timeout,
    )


def create_ingestion_service(
    config: KnowledgeClientSettings,
) -> HttpIngestionService:
    return HttpIngestionService(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.timeout,
    )


def create_configured_rag_service(
    environ: Mapping[str, str] | None = None,
) -> RagService:
    return create_rag_service(load_client_settings(environ))


def create_configured_ingestion_service(
    environ: Mapping[str, str] | None = None,
) -> IngestionService:
    return create_ingestion_service(load_client_settings(environ))


__all__ = [
    "KnowledgeClientSettings",
    "create_configured_ingestion_service",
    "create_configured_rag_service",
    "create_ingestion_service",
    "create_rag_service",
    "load_client_settings",
]
