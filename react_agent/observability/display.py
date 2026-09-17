"""面向交互界面的已计量结果展示与会话内汇总。"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any


def format_cost_cny(value: Any, *, unpriced_count: int = 0) -> str:
    """展示人民币估算费用，明确区分未知价格和真实零费用。"""
    cost = float(value or 0.0)
    if unpriced_count > 0:
        if cost > 0:
            return f"≥ ¥{cost:.4f}（另有 {unpriced_count} 次未计价）"
        return f"费用不可用（{unpriced_count} 次未计价）"
    if cost == 0:
        return "¥0.0000"
    if cost < 0.01:
        rounded = Decimal(str(cost)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        return f"¥{rounded:.4f}"
    rounded = Decimal(str(cost)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"¥{rounded:.2f}"


def format_usage_for_user(
    usage: dict[str, Any],
    latency_ms: float,
) -> dict[str, str]:
    """把当前轮用量转换为 Streamlit 展示文本。"""
    cost_text = format_cost_cny(
        usage.get("estimated_cost_cny", 0.0),
        unpriced_count=int(usage.get("unpriced_model_count", 0)),
    )
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
    """维护交互界面会话内的累计用量。"""

    def __init__(self) -> None:
        self.total_cost_cny = 0.0
        self.total_unpriced_model_calls = 0
        self.total_tokens = 0
        self.total_llm_calls = 0
        self.total_tool_runs = 0
        self.turn_count = 0
        self.turn_usages: list[dict[str, Any]] = []

    def record_turn(self, usage: dict[str, Any], latency_ms: float) -> None:
        self.total_cost_cny += float(usage.get("estimated_cost_cny", 0.0))
        self.total_unpriced_model_calls += int(usage.get("unpriced_model_count", 0))
        self.total_tokens += int(usage.get("total_tokens", 0))
        self.total_llm_calls += int(usage.get("llm_call_count", 0))
        self.total_tool_runs += int(usage.get("tool_runs_count", 0))
        self.turn_count += 1
        self.turn_usages.append(
            {"turn": self.turn_count, "latency_ms": round(latency_ms, 1), **usage}
        )

    def check_budget(self, limit_cny: float = 7.2) -> str | None:
        """仅返回 UI 提示，不参与 API 强制预算。"""
        if self.total_cost_cny < limit_cny:
            return None
        return (
            f"⚠️ 本次会话累计估算成本已达 ¥{self.total_cost_cny:.2f}，"
            f"超过预算阈值 ¥{limit_cny:.2f}。"
        )


__all__ = ["SessionUsageTracker", "format_cost_cny", "format_usage_for_user"]
