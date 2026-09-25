"""普通文本型文档的独立解析适配器。"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from knowledge.contracts import ParseRequest, ParseResult, ParsedChunk
from knowledge.source import source_identity


_SUPPORTED_EXTENSIONS = {".docx", ".txt", ".md", ".markdown", ".py"}


class BasicDocumentParserAdapter:
    """加载并分块 DOCX、纯文本、Markdown 和 Python 文件。"""

    async def parse(self, request: ParseRequest) -> ParseResult:
        normalized = request.normalized()
        return await asyncio.to_thread(self._parse_sync, normalized)

    @staticmethod
    def _parse_sync(request: ParseRequest) -> ParseResult:
        from langchain_community.document_loaders import Docx2txtLoader, TextLoader
        from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

        started = time.perf_counter()
        path = Path(request.file_path)
        if not path.is_file():
            raise FileNotFoundError(f"待解析文件不存在：{path}")
        extension = path.suffix.lower()
        if extension not in _SUPPORTED_EXTENSIONS:
            raise ValueError(f"普通文档解析器不支持该文件类型：{extension or '<无后缀>'}")

        if extension == ".docx":
            documents = Docx2txtLoader(str(path)).load()
        else:
            documents = TextLoader(str(path), encoding="utf-8").load()

        if extension == ".py":
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=500,
                chunk_overlap=50,
            )
        elif extension in {".md", ".markdown"}:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=500,
                chunk_overlap=50,
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
            )
        split_documents = splitter.split_documents(documents)

        source_file, industry = source_identity(
            request.file_path,
            request.source_root,
        )
        chunks: list[ParsedChunk] = []
        for index, document in enumerate(split_documents):
            content = str(document.page_content or "").strip()
            if not content:
                continue
            metadata: dict[str, Any] = dict(document.metadata or {})
            page_value = metadata.get("page")
            try:
                page = int(page_value) if page_value is not None else None
            except (TypeError, ValueError):
                page = None
            chunks.append(
                ParsedChunk(
                    content=content,
                    source_file=source_file,
                    source_page=page,
                    source_pages=(page,) if page is not None else (),
                    chunk_id=f"{source_file}::basic_{index:06d}",
                    doc_type="text",
                    parser_name="basic",
                    metadata={
                        **metadata,
                        **({"industry": industry} if industry else {}),
                    },
                )
            )

        return ParseResult(
            source_file=source_file,
            parser_name="basic",
            chunks=tuple(chunks),
            processing_time=time.perf_counter() - started,
            ocr_policy="disabled",
        )


__all__ = ["BasicDocumentParserAdapter"]
