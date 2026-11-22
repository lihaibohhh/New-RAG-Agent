"""建库模块拥有的本地对象图与生命周期。"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from knowledge.ingestion.parsing import DocumentParsingService
from knowledge.settings import IngestionSettings
from knowledge.ingestion.infrastructure.parsing.docling_pdf import (
    DoclingPreflightAdapter,
    DoclingServiceAdapter,
)
from knowledge.ingestion.infrastructure.parsing.local_pdf import LocalPdfParserAdapter
from knowledge.ingestion.infrastructure.parsing.other_documents import (
    OtherDocumentParserAdapter,
)
from knowledge.ingestion.infrastructure.storage import (
    ChromaVectorWriterAdapter,
    CompositeVectorIndexWriter,
    JsonIngestionManifestAdapter,
)
from knowledge.ingestion.pdf_routing import PdfParserRouter
from knowledge.ingestion.service import IngestionPipeline


class LocalIngestionService(IngestionPipeline):
    """本地建库应用服务：组装适配器并直接执行建库流程。"""

    def __init__(
        self,
        config: IngestionSettings,
        *,
        chroma_dir: str,
        embedding_provider: Callable[[], Any],
        chunk_store_provider: Callable[[], Any],
        knowledge_write_lock: threading.RLock,
        index_changed_provider: Callable[[], Any],
    ) -> None:
        docling_config = config.docling
        preflight_client = DoclingServiceAdapter(
            base_url=docling_config.base_url,
            api_key=docling_config.api_key,
            request_timeout=docling_config.request_timeout,
            task_timeout=docling_config.task_timeout,
            poll_interval=docling_config.poll_interval,
        )
        super().__init__(
            writer=CompositeVectorIndexWriter(
                ChromaVectorWriterAdapter(
                    chroma_dir=chroma_dir,
                    embedding_provider=embedding_provider,
                ),
                chunk_store_provider(),
                lock=knowledge_write_lock,
            ),
            manifest=JsonIngestionManifestAdapter(config.hash_record_path),
            preflight=DoclingPreflightAdapter(
                client=preflight_client,
                enabled=docling_config.enabled,
                strict_mode=docling_config.strict_mode,
            ),
            document_parser=self._build_document_parser(config),
            index_changed=index_changed_provider(),
            config=config.batch,
        )

    @staticmethod
    def _build_document_parser(config: IngestionSettings) -> DocumentParsingService:
        docling_config = config.docling
        pdf_parser = PdfParserRouter(
            local_parser=LocalPdfParserAdapter(),
            docling_parser=DoclingServiceAdapter(
                base_url=docling_config.base_url,
                api_key=docling_config.api_key,
                request_timeout=docling_config.request_timeout,
                task_timeout=docling_config.task_timeout,
                poll_interval=docling_config.poll_interval,
                max_pages_per_task=docling_config.max_pages_per_task,
                segment_retries=docling_config.segment_retries,
            ),
            docling_enabled=docling_config.enabled,
            strict_mode=docling_config.strict_mode,
            mode=docling_config.parser_mode,
            image_bytes_per_page=docling_config.image_bytes_per_page,
            min_text_chars_per_page=docling_config.min_text_chars_per_page,
            min_image_area_ratio=docling_config.min_image_area_ratio,
            min_table_like_pages_ratio=docling_config.min_table_like_pages_ratio,
            min_multi_column_pages_ratio=docling_config.min_multi_column_pages_ratio,
            max_pdf_pages=docling_config.max_pdf_pages,
            local_min_page_coverage=docling_config.local_min_page_coverage,
            local_min_chars_per_page=docling_config.local_min_chars_per_page,
            local_max_replacement_ratio=docling_config.local_max_replacement_ratio,
            local_min_traceable_ratio=docling_config.local_min_traceable_ratio,
            sample_pages=docling_config.sample_pages,
        )
        return DocumentParsingService(
            pdf_parser=pdf_parser,
            other_parser=OtherDocumentParserAdapter(),
        )

    async def close(self) -> None:
        """建库资源均由共享资源或单次调用管理，无独立句柄需关闭。"""


__all__ = ["LocalIngestionService"]
