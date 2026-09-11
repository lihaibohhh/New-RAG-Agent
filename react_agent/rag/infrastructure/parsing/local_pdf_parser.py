"""现有金融 PDF 解析器的端口适配器。"""
from __future__ import annotations

import asyncio
import time
import warnings
from typing import Any

from react_agent.rag.contracts import ParseRequest, ParseResult, ParsedChunk
from react_agent.rag.source import source_identity


class LocalPdfParserAdapter:
    """把现有 PyMuPDF/pdfplumber 输出转换成公共解析契约。"""

    async def parse(self, request: ParseRequest) -> ParseResult:
        normalized = request.normalized()
        return await asyncio.to_thread(self._parse_sync, normalized)

    @staticmethod
    def _parse_sync(request: ParseRequest) -> ParseResult:
        from react_agent.rag.infrastructure.parsing.pdf_parser import (
            load_pdf_with_layout,
        )

        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            documents = load_pdf_with_layout(
                request.file_path,
                source_root=request.source_root,
            )

        source_file, industry = source_identity(
            request.file_path,
            request.source_root,
        )
        chunks: list[ParsedChunk] = []
        for index, document in enumerate(documents or []):
            content = str(document.page_content or "").strip()
            if not content:
                continue
            metadata: dict[str, Any] = dict(document.metadata or {})
            page_value = metadata.get("page")
            try:
                page = int(page_value) if page_value is not None else None
            except (TypeError, ValueError):
                page = None
            pages = (page,) if page is not None else ()
            chunk_id = str(
                metadata.get("chunk_id")
                or f"{source_file}::local_{index:06d}"
            )
            doc_type = str(metadata.get("doc_type") or "text")
            chunks.append(
                ParsedChunk(
                    content=content,
                    source_file=source_file,
                    source_page=page,
                    source_pages=pages,
                    chunk_id=chunk_id,
                    doc_type=doc_type,
                    parser_name="local",
                    metadata={
                        **metadata,
                        **({"industry": industry} if industry else {}),
                    },
                )
            )

        return ParseResult(
            source_file=source_file,
            parser_name="local",
            chunks=tuple(chunks),
            processing_time=time.perf_counter() - started,
            warnings=tuple(str(item.message) for item in captured),
        )
