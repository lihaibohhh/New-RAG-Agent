"""应用会话用量的记录、增量计算和展示格式化。"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)
DEFAULT_USAGE_LOG = Path("logs/usage_metrics.jsonl")
_CUMULATIVE_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cache_hit_tokens",
    "cache_miss_tokens",
    "total_tokens",
    "estimated_cost_usd",
    "llm_call_count",
)


def log_usage(
    result: dict[str, Any],
    *,
    username: str,
    thread_id: str,
    question: str,
    latency_ms: float,
    prev_snapshot: dict[str, Any] | None = None,
    log_path: Path = DEFAULT_USAGE_LOG,
) -> dict[str, Any]:
    """记录一轮会话用量；仅在实际写入时创建日志目录。"""
    usage = extract_usage(result, prev_snapshot=prev_snapshot)
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
    return usage


def extract_cumulative_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    """提取用于下一轮差值计算的累计状态。"""
    snapshot = {key: result.get(key, 0) for key in _CUMULATIVE_KEYS}
    snapshot["tool_runs_count"] = len(result.get("tool_runs", []))
    return snapshot


def extract_usage(
    result: dict[str, Any],
    prev_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """从 Agent 累计状态中计算当前轮次增量。"""
    previous = prev_snapshot or {}
    usage = {
        key: result.get(key, 0) - previous.get(key, 0)
        for key in _CUMULATIVE_KEYS
    }
    usage["tool_runs_count"] = len(result.get("tool_runs", [])) - previous.get(
        "tool_runs_count",
        0,
    )
    return usage


def format_usage_for_user(
    usage: dict[str, Any],
    latency_ms: float,
) -> dict[str, str]:
    """把内部计量字段转换为 Streamlit 展示文本。"""
    cost = float(usage.get("estimated_cost_usd", 0.0))
    cost_text = "< ¥0.01" if cost < 0.001 else f"≈ ¥{cost * 7.2:.2f}"
    latency_text = (
        f"{latency_ms:.0f} ms"
        if latency_ms < 1000
        else f"{latency_ms / 1000:.1f} 秒"
    )
    tool_count = int(usage.get("tool_runs_count", 0))
    return {
        "耗时": latency_text,
        "模型调用": f"{int(usage.get('llm_call_count', 0))} 次",
        "工具使用": f"{tool_count} 次" if tool_count > 0 else "未使用",
        "本轮成本": cost_text,
        "Token 消耗": f"{int(usage.get('total_tokens', 0)):,}",
    }


class SessionUsageTracker:
    """维护 Streamlit 会话内的累计用量。"""

    def __init__(self) -> None:
        self.total_cost_usd = 0.0
        self.total_tokens = 0
        self.total_llm_calls = 0
        self.total_tool_runs = 0
        self.turn_count = 0
        self.turn_usages: list[dict[str, Any]] = []

    def record_turn(self, usage: dict[str, Any], latency_ms: float) -> None:
        self.total_cost_usd += float(usage.get("estimated_cost_usd", 0.0))
        self.total_tokens += int(usage.get("total_tokens", 0))
        self.total_llm_calls += int(usage.get("llm_call_count", 0))
        self.total_tool_runs += int(usage.get("tool_runs_count", 0))
        self.turn_count += 1
        self.turn_usages.append(
            {
                "turn": self.turn_count,
                "latency_ms": round(latency_ms, 1),
                **usage,
            }
        )

    def check_budget(self, limit_usd: float = 1.0) -> str | None:
        if self.total_cost_usd < limit_usd:
            return None
        return (
            f"⚠️ 本次会话累计成本已达 ${self.total_cost_usd:.4f} "
            f"(≈ ¥{self.total_cost_usd * 7.2:.2f})，"
            f"超过预算阈值 ${limit_usd:.2f}。"
        )


__all__ = [
    "DEFAULT_USAGE_LOG",
    "SessionUsageTracker",
    "extract_cumulative_snapshot",
    "extract_usage",
    "format_usage_for_user",
    "log_usage",
]
