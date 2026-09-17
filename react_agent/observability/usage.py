"""将已计算的本轮用量写入结构化日志。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
DEFAULT_USAGE_LOG = Path("logs/usage_metrics.jsonl")


def log_usage(
    usage: dict[str, Any],
    *,
    username: str,
    thread_id: str,
    question: str,
    latency_ms: float,
    log_path: Path = DEFAULT_USAGE_LOG,
) -> None:
    """记录已计算的一轮会话用量；不重算 token 或费用。"""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "username": username,
        "thread_id": thread_id,
        "question": question[:200],
        "latency_ms": round(latency_ms, 1),
        **usage,
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("[usage] 写入失败: %s", exc)
__all__ = ["DEFAULT_USAGE_LOG", "log_usage"]
