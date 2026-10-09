"""Knowledge Service 的环境配置与本地服务组装入口。"""

from __future__ import annotations

from knowledge.runtime import KnowledgeServices, create_knowledge_services
from knowledge.settings import KnowledgeSettings


def create_knowledge_settings() -> KnowledgeSettings:
    return KnowledgeSettings.from_env()


def create_knowledge_services_from_env() -> KnowledgeServices:
    return create_knowledge_services(create_knowledge_settings())


__all__ = ["create_knowledge_services_from_env", "create_knowledge_settings"]
