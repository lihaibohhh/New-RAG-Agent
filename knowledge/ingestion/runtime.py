"""建库模块拥有的本地对象图与生命周期。"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from knowledge.ingestion.document_service import DocumentParsingService
from knowledge.ingestion.config import IngestionRuntimeConfig
from knowledge.ingestion.infrastructure.parsing.basic_document_parser import (
    BasicDocumentParserAdapter,
)
from knowledge.ingestion.infrastructure.parsing.docling_parser import (
    DoclingServiceAdapter,
)
from knowledge.ingestion.infrastructure.parsing.ingestion_support import (
    DoclingPreflightAdapter,
    PdfPageCounterAdapter,
)
from knowledge.ingestion.infrastructure.parsing.local_pdf_parser import (
    LocalPdfParserAdapter,
)
from knowledge.ingestion.infrastructure.parsing.structured_table_parser import (
    StructuredTableParserAdapter,
)
from knowledge.ingestion.infrastructure.storage.chroma_vector_writer import (
    ChromaVectorWriterAdapter,
)
from knowledge.ingestion.infrastructure.storage.composite_writer import (
    CompositeVectorIndexWriter,
)
from knowledge.ingestion.infrastructure.storage.file_manifest import (
    JsonIngestionManifestAdapter,
)
from knowledge.ingestion.parser_router import PdfParserRouter
from knowledge.ingestion.ports import (
    KnowledgeIndexChangedPort,
    VectorIndexWriterPort,
)
from knowledge.ingestion.service import IngestionService


class LocalIngestionRuntime:
    """只组装并暴露文档解析与增量建库能力。"""

    def __init__(
        self,
        config: IngestionRuntimeConfig,
        *,
        chroma_dir: str,
        embedding_provider: Callable[[], Any],
        chunk_store_provider: Callable[[], VectorIndexWriterPort],
        knowledge_write_lock: threading.RLock,
        index_changed_provider: Callable[[], KnowledgeIndexChangedPort],
    ) -> None:
        self._config = config
        self._chroma_dir = chroma_dir
        self._embedding_provider = embedding_provider
        self._chunk_store_provider = chunk_store_provider
        self._knowledge_write_lock = knowledge_write_lock
        self._index_changed_provider = index_changed_provider
        self._ingestion_service: IngestionService | None = None
        self._document_parsing_service: DocumentParsingService | None = None
        self._lock = threading.RLock()

    def get_ingestion_service(self) -> IngestionService:
        if self._ingestion_service is not None:
            return self._ingestion_service

        document_parser = self.get_document_parsing_service()
        with self._lock:
            if self._ingestion_service is None:
                docling_config = self._config.docling
                preflight_client = DoclingServiceAdapter(
                    base_url=docling_config.base_url,
                    api_key=docling_config.api_key,
                    request_timeout=docling_config.request_timeout,
                    task_timeout=docling_config.task_timeout,
                    poll_interval=docling_config.poll_interval,
                )
                self._ingestion_service = IngestionService(
                    writer=CompositeVectorIndexWriter(
                        ChromaVectorWriterAdapter(
                            chroma_dir=self._chroma_dir,
                            embedding_provider=self._embedding_provider,
                        ),
                        self._chunk_store_provider(),
                        lock=self._knowledge_write_lock,
                    ),
                    manifest=JsonIngestionManifestAdapter(
                        self._config.hash_record_path
                    ),
                    page_counter=PdfPageCounterAdapter(),
                    preflight=DoclingPreflightAdapter(
                        client=preflight_client,
                        enabled=docling_config.enabled,
                        strict_mode=docling_config.strict_mode,
                    ),
                    document_parser=document_parser,
                    index_changed=self._index_changed_provider(),
                    config=self._config.service,
                )
        return self._ingestion_service

    def get_document_parsing_service(self) -> DocumentParsingService:
        if self._document_parsing_service is not None:
            return self._document_parsing_service
        with self._lock:
            if self._document_parsing_service is None:
                self._document_parsing_service = self._build_document_parsing_service()
        return self._document_parsing_service

    def _build_document_parsing_service(self) -> DocumentParsingService:
        config = self._config.docling
        pdf_parser = PdfParserRouter(
            local_parser=LocalPdfParserAdapter(),
            docling_parser=DoclingServiceAdapter(
                base_url=config.base_url,
                api_key=config.api_key,
                request_timeout=config.request_timeout,
                task_timeout=config.task_timeout,
                poll_interval=config.poll_interval,
                max_pages_per_task=config.max_pages_per_task,
                segment_retries=config.segment_retries,
            ),
            docling_enabled=config.enabled,
            strict_mode=config.strict_mode,
            mode=config.parser_mode,
            image_bytes_per_page=config.image_bytes_per_page,
            min_text_chars_per_page=config.min_text_chars_per_page,
            min_image_area_ratio=config.min_image_area_ratio,
            min_table_like_pages_ratio=config.min_table_like_pages_ratio,
            min_multi_column_pages_ratio=config.min_multi_column_pages_ratio,
            max_pdf_pages=config.max_pdf_pages,
            local_min_page_coverage=config.local_min_page_coverage,
            local_min_chars_per_page=config.local_min_chars_per_page,
            local_max_replacement_ratio=config.local_max_replacement_ratio,
            local_min_traceable_ratio=config.local_min_traceable_ratio,
            sample_pages=config.sample_pages,
        )
        return DocumentParsingService(
            pdf_parser=pdf_parser,
            basic_parser=BasicDocumentParserAdapter(),
            table_parser=StructuredTableParserAdapter(),
        )

    async def close(self) -> None:
        """释放建库对象图持有的引用；共享资源由顶层 Runtime 关闭。"""
        self._ingestion_service = None
        self._document_parsing_service = None


__all__ = ["LocalIngestionRuntime"]
