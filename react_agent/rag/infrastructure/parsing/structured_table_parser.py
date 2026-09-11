"""CSV 和 Excel 文件的独立结构化解析适配器。"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any

from react_agent.rag.contracts import ParseRequest, ParseResult, ParsedChunk
from react_agent.rag.source import source_identity


_SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
_SAFE_ID_RE = re.compile(r"[^\w.-]+", re.UNICODE)


def _has_value(value: Any) -> bool:
    import pandas as pd

    if value is None:
        return False
    if not bool(pd.notna(value)):
        return False
    return bool(str(value).strip())


class StructuredTableParserAdapter:
    """把表格的每一行转换为可检索片段，并保留工作表与行号。"""

    async def parse(self, request: ParseRequest) -> ParseResult:
        normalized = request.normalized()
        return await asyncio.to_thread(self._parse_sync, normalized)

    @staticmethod
    def _parse_sync(request: ParseRequest) -> ParseResult:
        import pandas as pd

        started = time.perf_counter()
        path = Path(request.file_path)
        if not path.is_file():
            raise FileNotFoundError(f"待解析文件不存在：{path}")
        extension = path.suffix.lower()
        if extension not in _SUPPORTED_EXTENSIONS:
            raise ValueError(f"表格解析器不支持该文件类型：{extension or '<无后缀>'}")

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
                metadata: dict[str, Any] = {
                    "sheet": str(sheet_name),
                    "row": row_offset,
                    **({"industry": industry} if industry else {}),
                }
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
                        metadata=metadata,
                    )
                )
        return ParseResult(
            source_file=source_file,
            parser_name="structured_table",
            chunks=tuple(chunks),
            processing_time=time.perf_counter() - started,
            ocr_policy="disabled",
        )


__all__ = ["StructuredTableParserAdapter"]
