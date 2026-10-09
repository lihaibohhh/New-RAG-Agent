"""Knowledge Service 内部组合根；业务调用方不应依赖本包。"""

from knowledge.runtime.container import KnowledgeServices, create_knowledge_services

__all__ = [
    "KnowledgeServices",
    "create_knowledge_services",
]
