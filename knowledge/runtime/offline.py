"""RAG 离线任务专用的显式服务工厂。"""
from __future__ import annotations

from knowledge.admin import RagAdminService
from knowledge.runtime.container import create_knowledge_runtime
from knowledge.runtime.config import KnowledgeRuntimeConfig


def create_admin_service(
    *,
    config: KnowledgeRuntimeConfig,
    chroma_dir: str | None = None,
) -> RagAdminService:
    """为指定 Chroma 路径创建非单例管理服务。"""
    runtime = create_knowledge_runtime(config)
    return runtime.create_admin_service(chroma_dir=chroma_dir)


__all__ = ["create_admin_service"]
