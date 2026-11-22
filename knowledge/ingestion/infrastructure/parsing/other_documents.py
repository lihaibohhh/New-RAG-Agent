"""非 PDF 文档的简单解析路径。"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any

from knowledge.ingestion.contracts import ParseRequest, ParseResult, ParsedChunk
from knowledge.ingestion.parsing import source_identity


_TEXT_EXTENSIONS = {".docx", ".txt", ".md", ".markdown", ".py"}
_TABLE_EXTENSIONS = {".csv", ".xlsx", ".xls"}
_SAFE_ID_RE = re.compile(r"[^\w.-]+", re.UNICODE)


class OtherDocumentParserAdapter:
    """统一处理非 PDF 的普通文本和表格文件。"""

    async def parse(self, request: ParseRequest) -> ParseResult:
        normalized = request.normalized()
        extension = Path(normalized.file_path).suffix.lower()
        if extension in _TEXT_EXTENSIONS:
            return await asyncio.to_thread(self._parse_text, normalized)
        if extension in _TABLE_EXTENSIONS:
            return await asyncio.to_thread(self._parse_table, normalized)
        raise ValueError(f"其他文档解析器不支持该文件类型：{extension or '<无后缀>'}")

    @staticmethod
    def _parse_text(request: ParseRequest) -> ParseResult:
        from langchain_community.document_loaders import Docx2txtLoader, TextLoader
        from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

        started = time.perf_counter()
        path = Path(request.file_path)
        if not path.is_file():
            raise FileNotFoundError(f"待解析文件不存在：{path}")
        extension = path.suffix.lower()
        if extension not in _TEXT_EXTENSIONS:
            raise ValueError(
                f"普通文档解析器不支持该文件类型：{extension or '<无后缀>'}"
            )

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


    @staticmethod
    def _parse_table(request: ParseRequest) -> ParseResult:
        import pandas as pd

        started = time.perf_counter()
        path = Path(request.file_path)
        if not path.is_file():
            raise FileNotFoundError(f"待解析文件不存在：{path}")
        extension = path.suffix.lower()
        if extension not in _TABLE_EXTENSIONS:
            raise ValueError(f"表格解析不支持该文件类型：{extension or '<无后缀>'}")

        if extension == ".csv":
            frames: dict[str, Any] = {"csv": pd.read_csv(path, encoding="utf-8-sig")}
        else:
            frames = pd.read_excel(path, sheet_name=None)

        source_file, industry = source_identity(
            request.file_path,
            request.source_root,
        )
        chunks: list[ParsedChunk] = []
        for sheet_name, frame in frames.items():
            headers = [str(header) for header in frame.columns.tolist()]
            safe_sheet = _SAFE_ID_RE.sub("_", str(sheet_name)).strip("_") or "sheet"
            for row_offset, (_, row) in enumerate(frame.iterrows(), start=2):
                parts = [
                    f"{header}：{value}"
                    for header, value in zip(headers, row.values)
                    if _has_value(value)
                ]
                if not parts:
                    continue
                chunks.append(
                    ParsedChunk(
                        content="；".join(parts),
                        source_file=source_file,
                        source_page=None,
                        source_pages=(),
                        chunk_id=(
                            f"{source_file}::sheet_{safe_sheet}::"
                            f"row_{row_offset:06d}"
                        ),
                        doc_type="table_row",
                        parser_name="structured_table",
                        metadata={
                            "sheet": str(sheet_name),
                            "row": row_offset,
                            **({"industry": industry} if industry else {}),
                        },
                    )
                )
        return ParseResult(
            source_file=source_file,
            parser_name="structured_table",
            chunks=tuple(chunks),
            processing_time=time.perf_counter() - started,
            ocr_policy="disabled",
        )


def _has_value(value: Any) -> bool:
    import pandas as pd

    if value is None or not bool(pd.notna(value)):
        return False
    return bool(str(value).strip())


__all__ = ["OtherDocumentParserAdapter"]
