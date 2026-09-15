"""RAG 增量建库应用服务。"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

from react_agent.core.config import settings
from react_agent.rag.contracts import IngestionReport, RagDocument
from react_agent.rag.ingestion.document_service import (
    SUPPORTED_EXTENSIONS,
    parse_file_to_documents,
)
from react_agent.rag.ports import (
    IngestionManifestPort,
    IngestionPreflightPort,
    DocumentParserPort,
    PdfPageCounterPort,
    RagCacheInvalidatorPort,
    VectorIndexWriterPort,
)
from react_agent.rag.observability import summarize_last_run


logger = logging.getLogger(__name__)


def _parse_one(
    args: tuple[str, str, str, DocumentParserPort],
) -> tuple[str, str, list[RagDocument], str | None, float]:
    """在独立进程中解析一个文件；保持模块顶层以兼容 Windows spawn。"""
    file_path, file_hash, source_root, parser = args
    started = time.perf_counter()
    try:
        chunks = parse_file_to_documents(
            parser,
            file_path,
            source_root=source_root,
        )
        return file_path, file_hash, chunks or [], None, time.perf_counter() - started
    except Exception as exc:
        return (
            file_path,
            file_hash,
            [],
            f"{type(exc).__name__}: {exc}",
            time.perf_counter() - started,
        )


def _file_hash(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class IngestionService:
    """编排扫描、增量判定、解析、批处理、写库和缓存失效。"""

    def __init__(
        self,
        *,
        writer: VectorIndexWriterPort,
        manifest: IngestionManifestPort,
        page_counter: PdfPageCounterPort,
        preflight: IngestionPreflightPort,
        document_parser: DocumentParserPort,
        cache_invalidator: RagCacheInvalidatorPort,
    ) -> None:
        self._writer = writer
        self._manifest = manifest
        self._page_counter = page_counter
        self._preflight = preflight
        self._document_parser = document_parser
        self._cache_invalidator = cache_invalidator
        self._build_lock = threading.Lock()

    async def ingest(self, data_dir: str) -> IngestionReport:
        raw = await asyncio.to_thread(self._build_sync, data_dir)
        status = str(raw.get("status") or "completed")
        cache_invalidated = False
        if status == "completed" and int(raw.get("chunks_written") or 0) > 0:
            await self._cache_invalidator.invalidate()
            cache_invalidated = True

        known_keys = {
            "status",
            "data_dir",
            "files_discovered",
            "files_selected",
            "files_processed",
            "files_skipped",
            "chunks_written",
        }
        return IngestionReport(
            data_dir=str(raw.get("data_dir") or data_dir),
            status=status,
            files_discovered=int(raw.get("files_discovered") or 0),
            files_selected=int(raw.get("files_selected") or 0),
            files_processed=int(raw.get("files_processed") or 0),
            files_skipped=int(raw.get("files_skipped") or 0),
            chunks_written=int(raw.get("chunks_written") or 0),
            cache_invalidated=cache_invalidated,
            details={key: value for key, value in raw.items() if key not in known_keys},
        )

    def ingest_sync(self, data_dir: str) -> IngestionReport:
        """供 CLI/脚本使用；异步调用方应直接 await ingest()。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ingest(data_dir))
        raise RuntimeError("事件循环已运行，请使用 await IngestionService.ingest(...)。")

    def _build_sync(self, data_dir: str) -> dict[str, Any]:
        # Chroma/SQLite 是单写模型，指纹清单也需要与批次提交保持同一临界区。
        with self._build_lock:
            return self._build_locked(data_dir)

    def _build_locked(self, data_dir: str) -> dict[str, Any]:
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error("[RAG] 数据目录不存在 path=%s", data_dir)
            return self._empty_result("missing_data_dir", data_dir)

        all_files = [
            path
            for path in data_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not all_files:
            logger.warning("[RAG] 数据目录没有支持的文件 path=%s", data_dir)
            return self._empty_result("no_supported_files", data_dir)

        hash_record = self._manifest.load()
        new_files: list[tuple[Path, str]] = []
        skipped_files: list[dict[str, object]] = []
        max_pdf_pages = settings.docling.DOCLING_MAX_PDF_PAGES
        for path in all_files:
            if path.suffix.lower() == ".pdf":
                try:
                    page_count = self._page_counter.count(str(path))
                except Exception:
                    logger.warning(
                        "[RAG] PDF 页数检查失败，继续解析 path=%s",
                        path,
                        exc_info=True,
                    )
                else:
                    if page_count > max_pdf_pages:
                        reason = f"共 {page_count} 页，超过建库上限 {max_pdf_pages} 页"
                        skipped_files.append(
                            {
                                "source_file": str(path.relative_to(data_path)),
                                "pages": page_count,
                                "reason": reason,
                            }
                        )
                        continue

            fingerprint = _file_hash(path)
            if hash_record.get(str(path)) != fingerprint:
                new_files.append((path, fingerprint))

        if not new_files:
            all_skipped = len(skipped_files) == len(all_files)
            return {
                "status": "no_eligible_files" if all_skipped else "up_to_date",
                "data_dir": str(data_dir),
                "files_discovered": len(all_files),
                "files_selected": 0,
                "files_processed": 0,
                "files_skipped": len(skipped_files),
                "chunks_written": 0,
                "skipped_files": skipped_files,
            }

        self._preflight.ensure_ready()
        self._writer.initialize()
        result = self._run_pipeline(
            data_path=data_path,
            data_dir=data_dir,
            all_files=all_files,
            new_files=new_files,
            skipped_files=skipped_files,
            hash_record=hash_record,
        )
        if result["status"] == "completed":
            summarize_last_run()
        return result

    def _run_pipeline(
        self,
        *,
        data_path: Path,
        data_dir: str,
        all_files: list[Path],
        new_files: list[tuple[Path, str]],
        skipped_files: list[dict[str, object]],
        hash_record: dict[str, str],
    ) -> dict[str, Any]:
        batch_size = settings.ingestion.RAG_INGESTION_BATCH_SIZE
        workers = min(
            settings.ingestion.RAG_INGESTION_WORKERS,
            cpu_count(),
            len(new_files),
        )
        fail_fast = settings.ingestion.RAG_INGESTION_FAIL_FAST
        source_root = str(data_path.resolve())
        task_args = [
            (str(path), fingerprint, source_root, self._document_parser)
            for path, fingerprint in new_files
        ]

        buffer: list[tuple[str, str, list[RagDocument]]] = []
        total_written = 0
        batch_counter = 0
        files_done = 0
        cpu_parse_total = 0.0

        def flush_batch(
            batch_chunks: list[RagDocument],
            completed_files: dict[str, str],
        ) -> None:
            nonlocal total_written, batch_counter
            batch_counter += 1
            self._writer.write(
                batch_chunks,
                batch_label=f"批次#{batch_counter}",
                global_offset=total_written,
            )
            hash_record.update(completed_files)
            self._manifest.save(hash_record)
            total_written += len(batch_chunks)
            logger.info(
                "[RAG] 建库批次完成 batch=%s total_chunks=%s",
                batch_counter,
                total_written,
            )

        wall_started = time.perf_counter()
        pool = Pool(processes=workers)
        try:
            for file_path, file_hash, chunks, error, elapsed in pool.imap_unordered(
                _parse_one,
                task_args,
            ):
                files_done += 1
                cpu_parse_total += elapsed
                if error:
                    logger.error("[RAG] 文档解析失败 path=%s error=%s", file_path, error)
                    if fail_fast:
                        raise RuntimeError(
                            f"严格建库因解析失败而停止：{Path(file_path).name}（{error}）"
                        )
                    continue
                if not chunks:
                    logger.warning("[RAG] 文档无可写内容 path=%s", file_path)
                    continue

                buffer.append((file_path, file_hash, chunks))
                buffer_chunk_count = sum(len(items) for _, _, items in buffer)
                while buffer_chunk_count >= batch_size:
                    batch_chunks: list[RagDocument] = []
                    remaining_buffer: list[tuple[str, str, list[RagDocument]]] = []
                    completed_files: dict[str, str] = {}
                    taken = 0
                    for path, fingerprint, file_chunks in buffer:
                        if taken >= batch_size:
                            remaining_buffer.append((path, fingerprint, file_chunks))
                            continue
                        remaining = batch_size - taken
                        if len(file_chunks) <= remaining:
                            batch_chunks.extend(file_chunks)
                            taken += len(file_chunks)
                            completed_files[path] = fingerprint
                        else:
                            batch_chunks.extend(file_chunks[:remaining])
                            remaining_buffer.append(
                                (path, fingerprint, file_chunks[remaining:])
                            )
                            taken += remaining
                    buffer = remaining_buffer
                    buffer_chunk_count = sum(len(items) for _, _, items in buffer)
                    flush_batch(batch_chunks, completed_files)

            if buffer:
                tail_chunks = [chunk for _, _, chunks in buffer for chunk in chunks]
                tail_done = {path: fingerprint for path, fingerprint, _ in buffer}
                flush_batch(tail_chunks, tail_done)
        except BaseException:
            pool.terminate()
            pool.join()
            raise
        else:
            pool.close()
            pool.join()

        wall_elapsed = time.perf_counter() - wall_started
        parallel_ratio = round(cpu_parse_total / wall_elapsed, 2) if wall_elapsed else 0
        logger.info(
            "[RAG] 解析完成 wall_seconds=%.2f cpu_seconds=%.2f workers=%s ratio=%s",
            wall_elapsed,
            cpu_parse_total,
            workers,
            parallel_ratio,
        )

        if total_written == 0:
            return {
                "status": "no_content",
                "data_dir": str(data_dir),
                "files_discovered": len(all_files),
                "files_selected": len(new_files),
                "files_processed": files_done,
                "files_skipped": len(skipped_files),
                "chunks_written": 0,
                "skipped_files": skipped_files,
            }
        return {
            "status": "completed",
            "data_dir": str(data_dir),
            "files_discovered": len(all_files),
            "files_selected": len(new_files),
            "files_processed": files_done,
            "files_skipped": len(skipped_files),
            "chunks_written": total_written,
            "skipped_files": skipped_files,
        }

    @staticmethod
    def _empty_result(status: str, data_dir: str) -> dict[str, Any]:
        return {
            "status": status,
            "data_dir": str(data_dir),
            "files_discovered": 0,
            "files_selected": 0,
            "files_processed": 0,
            "files_skipped": 0,
            "chunks_written": 0,
        }


__all__ = ["IngestionService"]
