"""Docling Serve 异步解析接口适配器。"""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from react_agent.rag.contracts import ParseRequest, ParseResult, ParsedChunk
from react_agent.rag.source import source_identity


_MARKDOWN_TABLE_RE = re.compile(r"(?m)^\s*\|.*\|\s*$")
_RUNNING_STATES = {"pending", "started", "running", "queued"}
_SUCCESS_STATES = {"success", "completed"}


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PdfSegment:
    """临时 PDF 分段及其在原文档中的页码范围。"""

    path: Path
    index: int
    page_start: int
    page_end: int

    @property
    def page_offset(self) -> int:
        return self.page_start - 1


class DoclingParserError(RuntimeError):
    """Docling 服务拒绝、失败或返回非法结果。"""


class DoclingServiceAdapter:
    """通过异步任务 API 调用独立 Docling Serve 服务。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        request_timeout: float = 120.0,
        task_timeout: float = 3600.0,
        poll_interval: float = 2.0,
        max_pages_per_task: int = 3,
        segment_retries: int = 1,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if max_pages_per_task < 1:
            raise ValueError("max_pages_per_task 必须大于 0")
        if segment_retries < 0:
            raise ValueError("segment_retries 不能小于 0")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._request_timeout = request_timeout
        self._task_timeout = task_timeout
        self._poll_interval = poll_interval
        self._max_pages_per_task = max_pages_per_task
        self._segment_retries = segment_retries
        self._transport = transport

    async def ensure_ready(self) -> None:
        """确认 Docling 已可接收任务；失败时给出统一、可操作的异常。"""
        headers = {"accept": "application/json"}
        if self._api_key:
            headers["X-Api-Key"] = self._api_key
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=httpx.Timeout(min(self._request_timeout, 10.0)),
                transport=self._transport,
            ) as client:
                response = await client.get("/ready")
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise DoclingParserError(
                f"Docling 健康检查失败：{type(exc).__name__}: {exc}"
            ) from exc

    async def parse(self, request: ParseRequest) -> ParseResult:
        normalized = request.normalized()
        path = Path(normalized.file_path)
        if not path.is_file():
            raise FileNotFoundError(f"待解析文件不存在：{path}")
        if path.suffix.lower() != ".pdf":
            raise DoclingParserError("Docling 适配器当前只接收 PDF 文件")

        do_ocr = normalized.ocr_policy != "disabled"
        force_ocr = normalized.ocr_policy == "force"

        headers = {"accept": "application/json"}
        if self._api_key:
            headers["X-Api-Key"] = self._api_key

        timeout = httpx.Timeout(self._request_timeout)
        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout,
            transport=self._transport,
        ) as client:
            page_count = await self._read_page_count(path)
            if page_count is None or page_count <= self._max_pages_per_task:
                return await self._parse_segment_with_retries(
                    client,
                    request=normalized,
                    upload_path=path,
                    do_ocr=do_ocr,
                    force_ocr=force_ocr,
                )

            directory = Path(
                await asyncio.to_thread(tempfile.mkdtemp, prefix="rag_docling_")
            )
            try:
                segments = await asyncio.to_thread(
                    self._split_pdf,
                    path,
                    directory,
                    self._max_pages_per_task,
                )
                logger.info(
                    "Docling 分页解析：source=%s pages=%d segments=%d pages_per_task=%d",
                    path.name,
                    page_count,
                    len(segments),
                    self._max_pages_per_task,
                )
                results: list[ParseResult] = []
                for segment in segments:
                    results.append(
                        await self._parse_segment_with_retries(
                            client,
                            request=normalized,
                            upload_path=segment.path,
                            do_ocr=do_ocr,
                            force_ocr=force_ocr,
                            segment=segment,
                            segment_count=len(segments),
                        )
                    )
            finally:
                await asyncio.to_thread(shutil.rmtree, directory, True)
            return self._merge_segment_results(normalized, results)

    async def _read_page_count(self, path: Path) -> int | None:
        """读取 PDF 页数；损坏文件仍交给 Docling 返回其原始错误。"""
        try:
            return await asyncio.to_thread(self._pdf_page_count, path)
        except Exception as exc:
            logger.warning("读取 PDF 页数失败，将整份交给 Docling：%s", exc)
            return None

    @staticmethod
    def _pdf_page_count(path: Path) -> int:
        import fitz

        with fitz.open(path) as document:
            return len(document)

    @staticmethod
    def _split_pdf(
        source_path: Path,
        target_dir: Path,
        pages_per_task: int,
    ) -> tuple[_PdfSegment, ...]:
        """在系统临时目录生成分页 PDF；调用方负责目录生命周期。"""
        import fitz

        segments: list[_PdfSegment] = []
        with fitz.open(source_path) as source:
            for index, start_index in enumerate(
                range(0, len(source), pages_per_task),
                start=1,
            ):
                end_index = min(start_index + pages_per_task, len(source))
                target_path = target_dir / (
                    f"segment_{index:04d}_pages_"
                    f"{start_index + 1:04d}_{end_index:04d}.pdf"
                )
                with fitz.open() as segment_document:
                    segment_document.insert_pdf(
                        source,
                        from_page=start_index,
                        to_page=end_index - 1,
                    )
                    segment_document.save(target_path)
                segments.append(
                    _PdfSegment(
                        path=target_path,
                        index=index,
                        page_start=start_index + 1,
                        page_end=end_index,
                    )
                )
        return tuple(segments)

    async def _parse_segment_with_retries(
        self,
        client: httpx.AsyncClient,
        *,
        request: ParseRequest,
        upload_path: Path,
        do_ocr: bool,
        force_ocr: bool,
        segment: _PdfSegment | None = None,
        segment_count: int = 1,
    ) -> ParseResult:
        attempts = self._segment_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                payload = await self._submit_and_collect(
                    client,
                    upload_path=upload_path,
                    do_ocr=do_ocr,
                    force_ocr=force_ocr,
                )
                result = self._normalize_result(
                    request,
                    payload,
                    page_offset=segment.page_offset if segment else 0,
                    segment=segment,
                    segment_count=segment_count,
                )
                if not result.has_content:
                    logger.info(
                        "Docling 任务成功但无可索引内容，按正常空结果放行："
                        "source=%s segment=%s/%s pages=%s",
                        request.file_path,
                        segment.index if segment else 1,
                        segment_count,
                        (
                            f"{segment.page_start}-{segment.page_end}"
                            if segment
                            else "all"
                        ),
                    )
                return result
            except (DoclingParserError, TimeoutError, httpx.HTTPError) as exc:
                if attempt >= attempts:
                    page_label = (
                        f"第 {segment.page_start}-{segment.page_end} 页"
                        if segment
                        else "整份文件"
                    )
                    raise DoclingParserError(
                        f"Docling {page_label}解析在 {attempts} 次尝试后失败：{exc}"
                    ) from exc
                logger.warning(
                    "Docling 分段解析失败，将重试：source=%s segment=%s/%s "
                    "attempt=%s/%s error=%s",
                    request.file_path,
                    segment.index if segment else 1,
                    segment_count,
                    attempt,
                    attempts,
                    exc,
                )
                await self._wait_until_ready(client)
        raise AssertionError("Docling 重试循环意外退出")

    async def _submit_and_collect(
        self,
        client: httpx.AsyncClient,
        *,
        upload_path: Path,
        do_ocr: bool,
        force_ocr: bool,
    ) -> dict[str, Any]:
        with upload_path.open("rb") as stream:
            response = await client.post(
                "/v1/chunk/hybrid/file/async",
                files={"files": (upload_path.name, stream, "application/pdf")},
                data={
                    "convert_from_formats": "pdf",
                    "convert_do_ocr": str(do_ocr).lower(),
                    "convert_force_ocr": str(force_ocr).lower(),
                    "convert_table_mode": "fast",
                    "convert_do_pdf_heading_hierarchy": "true",
                    "convert_document_timeout": str(int(self._task_timeout)),
                    "chunking_use_markdown_tables": "true",
                    "chunking_include_raw_text": "true",
                },
            )
        response.raise_for_status()
        task = response.json()
        task_id = str(task.get("task_id") or "")
        if not task_id:
            raise DoclingParserError("Docling 未返回 task_id")

        await self._wait_for_task(client, task_id)
        result_response = await client.get(f"/v1/result/{task_id}")
        result_response.raise_for_status()
        payload = result_response.json()
        result = payload.get("result") if isinstance(payload, dict) else None
        if isinstance(result, dict):
            payload = result
        if not isinstance(payload, dict):
            raise DoclingParserError("Docling 返回的解析结果格式无效")
        return payload

    async def _wait_until_ready(self, client: httpx.AsyncClient) -> None:
        """任务失败后等待服务恢复，覆盖容器短暂重启的场景。"""
        deadline = asyncio.get_running_loop().time() + min(
            self._task_timeout,
            120.0,
        )
        last_error: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            try:
                response = await client.get("/ready")
                response.raise_for_status()
                return
            except httpx.HTTPError as exc:
                last_error = exc
                await asyncio.sleep(self._poll_interval)
        raise DoclingParserError(
            f"Docling 重试前等待就绪超时：{last_error or 'unknown'}"
        )

    async def _wait_for_task(
        self,
        client: httpx.AsyncClient,
        task_id: str,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + self._task_timeout
        while True:
            response = await client.get(f"/v1/status/poll/{task_id}")
            response.raise_for_status()
            status_payload = response.json()
            status = str(status_payload.get("task_status") or "").lower()
            if status in _SUCCESS_STATES:
                return
            if status not in _RUNNING_STATES:
                message = str(
                    status_payload.get("error_message")
                    or status_payload.get("failure")
                    or status
                    or "unknown"
                )
                raise DoclingParserError(f"Docling 任务失败：{message}")
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    f"Docling 任务超过 {self._task_timeout:g} 秒仍未完成"
                )
            await asyncio.sleep(self._poll_interval)

    @staticmethod
    def _normalize_result(
        request: ParseRequest,
        payload: dict[str, Any],
        *,
        page_offset: int = 0,
        segment: _PdfSegment | None = None,
        segment_count: int = 1,
    ) -> ParseResult:
        source_file, industry = source_identity(
            request.file_path,
            request.source_root,
        )
        chunks: list[ParsedChunk] = []
        for fallback_index, item in enumerate(payload.get("chunks") or []):
            if not isinstance(item, dict):
                continue
            content = str(item.get("text") or "").strip()
            if not content:
                continue
            local_pages = tuple(
                sorted(
                    {
                        int(page)
                        for page in (item.get("page_numbers") or [])
                        if str(page).isdigit()
                    }
                )
            )
            pages = tuple(page + page_offset for page in local_pages)
            page = pages[0] if pages else None
            chunk_index = item.get("chunk_index", fallback_index)
            page_label = "-".join(str(value) for value in pages) or "unknown"
            headings = tuple(
                str(value).strip()
                for value in (item.get("headings") or [])
                if str(value).strip()
            )
            doc_items = tuple(str(value) for value in (item.get("doc_items") or ()))
            contains_table = _MARKDOWN_TABLE_RE.search(content) or any(
                "/tables/" in reference for reference in doc_items
            )
            doc_type = "table" if contains_table else "text"
            metadata = dict(item.get("metadata") or {})
            metadata.update(
                {
                    "doc_items": doc_items,
                    "num_tokens": item.get("num_tokens"),
                    **({"industry": industry} if industry else {}),
                }
            )
            if segment is not None:
                metadata.update(
                    {
                        "docling_segment": segment.index,
                        "docling_segment_count": segment_count,
                        "docling_segment_page_start": segment.page_start,
                        "docling_segment_page_end": segment.page_end,
                    }
                )
            chunk_suffix = (
                f"docling_s{segment.index:04d}_{int(chunk_index):06d}"
                if segment is not None
                else f"docling_{int(chunk_index):06d}"
            )
            chunks.append(
                ParsedChunk(
                    content=content,
                    source_file=source_file,
                    source_page=page,
                    source_pages=pages,
                    chunk_id=(
                        f"{source_file}::pages_{page_label}::"
                        f"{chunk_suffix}"
                    ),
                    doc_type=doc_type,
                    headings=headings,
                    parser_name="docling",
                    metadata=metadata,
                )
            )

        if chunks:
            warnings: tuple[str, ...] = ()
        elif segment is not None:
            warnings = (
                f"Docling 第 {segment.page_start}-{segment.page_end} 页"
                "无可索引内容，已正常跳过",
            )
        else:
            warnings = ("Docling 任务成功但无可索引内容，已正常跳过",)
        return ParseResult(
            source_file=source_file,
            parser_name="docling",
            chunks=tuple(chunks),
            processing_time=float(payload.get("processing_time") or 0.0),
            warnings=warnings,
            ocr_policy=request.ocr_policy,
        )

    @staticmethod
    def _merge_segment_results(
        request: ParseRequest,
        results: list[ParseResult],
    ) -> ParseResult:
        source_file, _ = source_identity(request.file_path, request.source_root)
        chunks = tuple(chunk for result in results for chunk in result.chunks)
        return ParseResult(
            source_file=source_file,
            parser_name="docling",
            chunks=chunks,
            processing_time=sum(result.processing_time for result in results),
            warnings=tuple(
                warning
                for result in results
                for warning in result.warnings
            ),
            ocr_policy=request.ocr_policy,
        )
