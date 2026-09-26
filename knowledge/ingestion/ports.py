"""建库模块拥有、由内部服务和适配器实现的端口。"""
from __future__ import annotations

from typing import Protocol

from knowledge.contracts import KnowledgeDocument
from knowledge.ingestion.contracts import ParseRequest, ParseResult


class DocumentParserPort(Protocol):
    async def parse(self, request: ParseRequest) -> ParseResult: ...


class VectorIndexWriterPort(Protocol):
    def initialize(self) -> None: ...

    def write(
        self,
        documents: list[KnowledgeDocument],
        *,
        batch_label: str,
        global_offset: int,
    ) -> None: ...


class IngestionManifestPort(Protocol):
    def load(self) -> dict[str, str]: ...

    def save(self, record: dict[str, str]) -> None: ...


class PdfPageCounterPort(Protocol):
    def count(self, file_path: str) -> int: ...


class IngestionPreflightPort(Protocol):
    def ensure_ready(self) -> None: ...


class KnowledgeIndexChangedPort(Protocol):
    """接收已提交的知识索引变更事件。"""

    async def notify_index_changed(self) -> None: ...


__all__ = [
    "DocumentParserPort",
    "IngestionManifestPort",
    "IngestionPreflightPort",
    "KnowledgeIndexChangedPort",
    "PdfPageCounterPort",
    "VectorIndexWriterPort",
]
