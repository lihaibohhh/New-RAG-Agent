"""评测数据集生成所需的 Chunk 数据源。"""

import asyncio
import os
from pathlib import Path

from langchain_core.documents import Document


def load_all_chunks(db_path: Path) -> list[Document]:
    """通过 RAG 管理服务读取全量 Chunk，不加载 Embedding 模型。"""
    if os.getenv("KNOWLEDGE_SERVICE_URL", "").strip():
        from knowledge.client import create_configured_rag_runtime

        async def read_remote_chunks():
            runtime = create_configured_rag_runtime()
            try:
                return await runtime.get_admin_service().read_chunks()
            finally:
                await runtime.close()

        stored_chunks = asyncio.run(read_remote_chunks())
        source_label = os.environ["KNOWLEDGE_SERVICE_URL"]
    else:
        from knowledge.runtime import KnowledgeRuntimeConfig
        from knowledge.runtime.offline import create_admin_service

        stored_chunks = create_admin_service(
            config=KnowledgeRuntimeConfig(),
            chroma_dir=str(db_path),
        ).read_chunks_sync()
        source_label = str(db_path)

    print(f"[generator] 通过 RAG 管理服务读取 chunks：{source_label}")
    documents = [
        Document(
            page_content=chunk.content,
            metadata={**dict(chunk.metadata), "chunk_id": chunk.chunk_id},
        )
        for chunk in stored_chunks
    ]
    print(f"[generator] 共读取 {len(documents)} 个 chunks")
    return documents
