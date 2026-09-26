"""建库流水线的阶段计时和汇总。"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


logger = logging.getLogger(__name__)
DEFAULT_LOG_PATH = Path("ingestion_metrics.jsonl")
_current_run_records: list[dict[str, Any]] = []


@contextmanager
def timer(
    stage: str,
    meta: dict[str, Any] | None = None,
    log_path: Path = DEFAULT_LOG_PATH,
    print_result: bool = True,
) -> Iterator[None]:
    """记录 RAG 建库阶段耗时、状态和可选吞吐量。"""
    started_at = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception as exc:
        status = f"failed: {type(exc).__name__}"
        raise
    finally:
        elapsed = time.perf_counter() - started_at
        metadata = meta or {}
        record: dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "stage": stage,
            "status": status,
            "elapsed_s": round(elapsed, 3),
            **metadata,
        }
        if elapsed > 0 and "token_count" in metadata:
            record["tokens_per_sec"] = round(metadata["token_count"] / elapsed, 1)
        if elapsed > 0 and "doc_count" in metadata:
            record["docs_per_sec"] = round(metadata["doc_count"] / elapsed, 3)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("[RAG metrics] 写入失败: %s", exc)
        _current_run_records.append(record)
        if print_result:
            logger.info(
                "[RAG metrics] stage=%s status=%s elapsed=%.3fs",
                stage,
                status,
                elapsed,
            )


def summarize_last_run(clear: bool = True) -> None:
    """通过日志输出最近一次建库流水线的阶段汇总。"""
    global _current_run_records
    if not _current_run_records:
        logger.info("[RAG metrics] 本次运行无计时记录")
        return
    totals: dict[str, float] = {}
    failures: dict[str, int] = {}
    for record in _current_run_records:
        stage = str(record["stage"])
        totals[stage] = totals.get(stage, 0.0) + float(record["elapsed_s"])
        if record.get("status") != "success":
            failures[stage] = failures.get(stage, 0) + 1
    logger.info(
        "[RAG metrics] total=%.3fs stages=%s failures=%s",
        sum(totals.values()),
        totals,
        failures,
    )
    if clear:
        _current_run_records = []


__all__ = ["DEFAULT_LOG_PATH", "summarize_last_run", "timer"]
