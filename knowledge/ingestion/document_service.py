"""所有知识库文档解析的唯一应用层入口。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from knowledge.contracts import (
    ChunkMetadata,
    KnowledgeDocument,
    SourceReference,
)
from knowledge.ingestion.contracts import (
    ParseRequest,
    ParseResult,
)
from knowledge.ingestion.ports import DocumentParserPort


PDF_EXTENSIONS = {".pdf"}
BASIC_EXTENSIONS = {".docx", ".txt", ".md", ".markdown", ".py"}
TABLE_EXTENSIONS = {".csv", ".xlsx", ".xls"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | BASIC_EXTENSIONS | TABLE_EXTENSIONS


class UnsupportedDocumentError(ValueError):
    """文件格式不属于当前知识库建库范围。"""


class DocumentParsingService:
    """按文件类型委托独立解析器，不实现任何具体解析算法。"""

    def __init__(
        self,
        *,
        pdf_parser: DocumentParserPort,
        basic_parser: DocumentParserPort,
        table_parser: DocumentParserPort,
    ) -> None:
        self._pdf_parser = pdf_parser
        self._basic_parser = basic_parser
        self._table_parser = table_parser

    async def parse(self, request: ParseRequest) -> ParseResult:
        normalized = request.normalized()
        extension = Path(normalized.file_path).suffix.lower()
        if extension in PDF_EXTENSIONS:
            return await self._pdf_parser.parse(normalized)
        if normalized.parser_hint == "docling":
            raise UnsupportedDocumentError("Docling 显式路由当前只支持 PDF 文件")

        local_request = replace(
            normalized,
            parser_hint="local",
            ocr_policy="disabled",
        )
        if extension in BASIC_EXTENSIONS:
            return await self._basic_parser.parse(local_request)
        if extension in TABLE_EXTENSIONS:
            return await self._table_parser.parse(local_request)
        supported = "、".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedDocumentError(
            f"暂不支持的文件类型：{extension or '<无后缀>'}；支持：{supported}"
        )


def _scalar_metadata(metadata: dict[str, Any]) -> dict[str, bool | int | float | str]:
    """过滤为 Chroma 可接受的标量，解析器私有结构不得泄漏到存储层。"""
    return {
        str(key): value
        for key, value in metadata.items()
        if isinstance(value, (bool, int, float, str))
    }


def parse_result_to_documents(
    result: ParseResult,
    *,
    original_path: str,
) -> list[KnowledgeDocument]:
    """把解析结果转换为与具体存储和检索实现无关的公共知识文档。"""
    documents: list[KnowledgeDocument] = []
    for chunk in result.chunks:
        parser_metadata = ChunkMetadata.from_mapping(
            _scalar_metadata(chunk.metadata)
        )
        extra = dict(parser_metadata.extra)
        extra["parser"] = chunk.parser_name
        if chunk.headings:
            extra["headings"] = " > ".join(chunk.headings)
        if result.ocr_policy is not None:
            extra["ocr_policy"] = result.ocr_policy
        if result.route_reasons:
            extra["route_reasons"] = "；".join(result.route_reasons)
        metadata = ChunkMetadata(
            source=SourceReference(
                source_file=chunk.source_file,
                source_path=original_path,
                source_page=chunk.source_page,
                source_pages=chunk.source_pages,
            ),
            chunk_id=chunk.chunk_id,
            doc_type=chunk.doc_type,
            industry=parser_metadata.industry,
            retrieval_score=parser_metadata.retrieval_score,
            extra=extra,
        )
        documents.append(
            KnowledgeDocument(
                content=chunk.content,
                metadata=metadata,
                document_id=chunk.chunk_id,
            )
        )
    return documents


__all__ = [
    "BASIC_EXTENSIONS",
    "DocumentParsingService",
    "PDF_EXTENSIONS",
    "SUPPORTED_EXTENSIONS",
    "TABLE_EXTENSIONS",
    "UnsupportedDocumentError",
    "parse_result_to_documents",
]
