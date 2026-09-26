"""仅在建库模块内部流转的解析契约。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from knowledge.contracts import KnowledgeValidationError


OcrPolicy = Literal["auto", "force", "disabled"]
ParserHint = Literal["auto", "local", "docling"]


@dataclass(frozen=True)
class ParseRequest:
    """一次文档解析请求，不暴露具体解析器实现。"""

    file_path: str
    source_root: str | None = None
    parser_hint: ParserHint = "auto"
    ocr_policy: OcrPolicy = "auto"

    def normalized(self) -> "ParseRequest":
        file_path = (self.file_path or "").strip()
        if not file_path:
            raise KnowledgeValidationError("待解析文件路径不能为空")
        source_root = (self.source_root or "").strip() or None
        if self.parser_hint not in {"auto", "local", "docling"}:
            raise KnowledgeValidationError(
                "parser_hint 必须是 auto、local 或 docling"
            )
        if self.ocr_policy not in {"auto", "force", "disabled"}:
            raise KnowledgeValidationError(
                "ocr_policy 必须是 auto、force 或 disabled"
            )
        return ParseRequest(
            file_path=file_path,
            source_root=source_root,
            parser_hint=self.parser_hint,
            ocr_policy=self.ocr_policy,
        )


@dataclass(frozen=True)
class ParsedChunk:
    """解析阶段的统一片段，独立于 Docling 和 LangChain。"""

    content: str
    source_file: str
    source_page: int | None
    source_pages: tuple[int, ...]
    chunk_id: str
    doc_type: str = "text"
    headings: tuple[str, ...] = ()
    parser_name: str = "unknown"
    parser_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParseResult:
    """文档解析服务的统一结果。"""

    source_file: str
    parser_name: str
    chunks: tuple[ParsedChunk, ...] = ()
    processing_time: float = 0.0
    warnings: tuple[str, ...] = ()
    ocr_policy: OcrPolicy | None = None
    route_reasons: tuple[str, ...] = ()

    @property
    def has_content(self) -> bool:
        return bool(self.chunks)


__all__ = [
    "OcrPolicy",
    "ParsedChunk",
    "ParserHint",
    "ParseRequest",
    "ParseResult",
]
