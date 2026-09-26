"""RAG 模块独占的知识库存储读取适配器。"""

from knowledge.rag.infrastructure.storage.chroma_knowledge_base import (
    ChromaKnowledgeBaseAdapter,
)

__all__ = ["ChromaKnowledgeBaseAdapter"]
