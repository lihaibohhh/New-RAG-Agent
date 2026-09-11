"""RAG 离线任务专用的显式服务工厂。"""
from __future__ import annotations

from react_agent.rag.admin import RagAdminService
from react_agent.rag.runtime.container import create_rag_runtime


def create_admin_service(*, chroma_dir: str | None = None) -> RagAdminService:
    """为指定 Chroma 路径创建非单例管理服务。"""
    runtime = create_rag_runtime()
    return runtime.create_admin_service(chroma_dir=chroma_dir)


__all__ = ["create_admin_service"]
