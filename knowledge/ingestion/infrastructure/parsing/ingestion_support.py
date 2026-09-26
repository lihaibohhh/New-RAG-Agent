"""建库流程使用的解析基础设施辅助适配器。"""
from __future__ import annotations

import asyncio

from knowledge.ingestion.infrastructure.parsing.docling_parser import (
    DoclingServiceAdapter,
)


class PdfPageCounterAdapter:
    def count(self, file_path: str) -> int:
        import fitz

        with fitz.open(file_path) as document:
            return len(document)


class DoclingPreflightAdapter:
    def __init__(
        self,
        *,
        client: DoclingServiceAdapter,
        enabled: bool,
        strict_mode: bool,
    ) -> None:
        self._client = client
        self._enabled = enabled
        self._strict_mode = strict_mode

    def ensure_ready(self) -> None:
        if not self._enabled or not self._strict_mode:
            return
        try:
            asyncio.run(self._client.ensure_ready())
        except Exception as exc:
            raise RuntimeError(
                "Docling 严格建库预检失败；请确认 ingestion profile 已启动且健康"
            ) from exc


__all__ = ["DoclingPreflightAdapter", "PdfPageCounterAdapter"]
