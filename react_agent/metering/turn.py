"""从 Agent Checkpoint 累计状态投影当前轮次用量。"""

from __future__ import annotations

from typing import Any


_CUMULATIVE_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cache_hit_tokens",
    "cache_miss_tokens",
    "total_tokens",
    "estimated_cost_cny",
    "unpriced_model_count",
    "llm_call_count",
)
_CURRENT_VALUE_KEYS = (
    "pricing_requested_model",
    "pricing_response_model",
    "pricing_model",
    "pricing_tariff",
    "pricing_rate_card_version",
)


def extract_cumulative_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    """提取供下一轮差值计算使用的累计状态。"""
    snapshot = {key: result.get(key, 0) for key in _CUMULATIVE_KEYS}
    snapshot.update({key: result.get(key, "") for key in _CURRENT_VALUE_KEYS})
    snapshot["tool_runs_count"] = len(result.get("tool_runs", []))
    return snapshot


def extract_usage(
    result: dict[str, Any],
    prev_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """从 Agent 累计状态计算本轮增量。"""
    previous = prev_snapshot or {}
    usage = {
        key: result.get(key, 0) - previous.get(key, 0)
        for key in _CUMULATIVE_KEYS
    }
    usage.update({key: result.get(key, "") for key in _CURRENT_VALUE_KEYS})
    # tool_runs 在 Agent State 中最多保留 50 条，避免窗口滚动产生负数。
    usage["tool_runs_count"] = max(
        0,
        len(result.get("tool_runs", [])) - int(previous.get("tool_runs_count", 0)),
    )
    return usage


__all__ = ["extract_cumulative_snapshot", "extract_usage"]
