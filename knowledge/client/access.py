"""从环境配置组装远程 Knowledge Service Runtime。"""
from __future__ import annotations

import os
from collections.abc import Mapping

from knowledge.client.config import KnowledgeClientConfig
from knowledge.client.remote import RemoteIngestionRuntime, RemoteRagRuntime
from knowledge.runtime_ports import IngestionRuntimePort, RagRuntimePort


_VALID_RUNTIME_MODES = frozenset({"local", "remote"})


def load_client_config(
    environ: Mapping[str, str] | None = None,
) -> KnowledgeClientConfig:
    """读取客户端环境配置；非远程模式立即失败。"""
    source = os.environ if environ is None else environ
    mode = source.get("RAG_RUNTIME_MODE", "remote").strip().lower()
    if mode not in _VALID_RUNTIME_MODES:
        raise ValueError("RAG_RUNTIME_MODE 仅支持 remote 或 local")
    if mode == "local":
        raise RuntimeError(
            "RAG_RUNTIME_MODE=local 只允许 Knowledge Service 或显式离线任务使用；"
            "Agent、MCP 与评测入口必须通过 remote 模式访问知识库"
        )

    base_url = source.get("KNOWLEDGE_SERVICE_URL", "").strip()
    if not base_url:
        raise RuntimeError(
            "RAG_RUNTIME_MODE=remote 时必须配置 KNOWLEDGE_SERVICE_URL；"
            "只有 Knowledge Service 或显式离线任务可以使用 local 模式"
        )
    try:
        timeout = float(source.get("KNOWLEDGE_SERVICE_TIMEOUT", "150"))
    except ValueError:
        timeout = 150.0
    return KnowledgeClientConfig(
        base_url=base_url,
        api_key=source.get("KNOWLEDGE_SERVICE_API_KEY", ""),
        timeout=max(1.0, timeout),
    )


def create_remote_rag_runtime(
    config: KnowledgeClientConfig,
) -> RemoteRagRuntime:
    """使用显式客户端配置创建远程 Runtime。"""
    return RemoteRagRuntime(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.timeout,
    )


def create_remote_ingestion_runtime(
    config: KnowledgeClientConfig,
) -> RemoteIngestionRuntime:
    """使用显式客户端配置创建远程建库 Runtime。"""
    return RemoteIngestionRuntime(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.timeout,
    )


def create_configured_rag_runtime() -> RagRuntimePort:
    """从环境配置创建远程 Runtime。"""
    return create_remote_rag_runtime(load_client_config())


def create_configured_ingestion_runtime() -> IngestionRuntimePort:
    """从环境配置创建仅暴露建库能力的远程 Runtime。"""
    return create_remote_ingestion_runtime(load_client_config())


__all__ = [
    "IngestionRuntimePort",
    "RagRuntimePort",
    "create_configured_ingestion_runtime",
    "create_configured_rag_runtime",
    "create_remote_ingestion_runtime",
    "create_remote_rag_runtime",
    "load_client_config",
]
