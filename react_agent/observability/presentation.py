"""将 Agent 运行状态与用量转换为安全、可读的界面信息。"""

from __future__ import annotations

from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal, Mapping

from agent_tools.presentation import TOOL_PROGRESS_LABELS


@dataclass(frozen=True)
class AgentProgress:
    """可由入站适配器安全展示的小型进度更新。"""

    stage: str
    label: str
    state: Literal["running", "complete"] = "running"


def _safe_tool_name(value: Any) -> str:
    raw = str(value or "").strip()
    safe = "".join(
        character
        for character in raw
        if character.isalnum() or character in "_-"
    )
    return safe[:48] or "unknown"


def _is_compaction_event(event: Mapping[str, Any]) -> bool:
    metadata = event.get("metadata")
    node = metadata.get("langgraph_node") if isinstance(metadata, Mapping) else None
    return node == "compact_history" or "context_compaction" in tuple(
        event.get("tags") or ()
    )


def progress_from_event(event: Mapping[str, Any]) -> AgentProgress | None:
    """将 LangChain 事件转换为有界且不泄露载荷的进度信息。"""
    event_name = str(event.get("event") or "")
    metadata = event.get("metadata")
    node = metadata.get("langgraph_node") if isinstance(metadata, Mapping) else None

    if event_name == "on_chat_model_start":
        if _is_compaction_event(event):
            return AgentProgress(
                stage="history_compaction",
                label="正在整理较早的对话上下文……",
            )
        if node == "finalize_model":
            return AgentProgress(
                stage="finalize_model",
                label="正在整理最终回答……",
            )
        if node == "call_model":
            return AgentProgress(
                stage="call_model",
                label="正在分析问题并规划下一步……",
            )
        return None

    if event_name == "on_chat_model_stream":
        if _is_compaction_event(event):
            return None
        data = event.get("data")
        chunk = data.get("chunk") if isinstance(data, Mapping) else None
        if not getattr(chunk, "content", None):
            return None
        return AgentProgress(stage="model_stream", label="正在生成回答……")

    if event_name in {"on_tool_start", "on_tool_end"}:
        tool_name = _safe_tool_name(event.get("name"))
        labels = TOOL_PROGRESS_LABELS.get(tool_name)
        if labels is None:
            action = "正在调用" if event_name == "on_tool_start" else "已完成"
            label = f"工具 {tool_name} {action}……"
        else:
            label = labels[0 if event_name == "on_tool_start" else 1]
        suffix = "start" if event_name == "on_tool_start" else "end"
        return AgentProgress(stage=f"tool:{tool_name}:{suffix}", label=label)

    if event_name == "on_chain_end" and not event.get("parent_ids"):
        return AgentProgress(
            stage="complete",
            label="Agent 已完成回答",
            state="complete",
        )

    return None


def root_graph_output(event: Mapping[str, Any]) -> dict[str, Any] | None:
    """从根图的结束事件中提取最终状态。"""
    if event.get("event") != "on_chain_end" or event.get("parent_ids"):
        return None
    data = event.get("data")
    output = data.get("output") if isinstance(data, Mapping) else None
    return dict(output) if isinstance(output, Mapping) else None


async def consume_progress_events(
    events: AsyncIterable[Mapping[str, Any]],
    emit: Callable[[AgentProgress], None],
) -> dict[str, Any]:
    """消费一次图运行事件，去重发送进度并返回最终状态。"""
    final_result: dict[str, Any] | None = None
    last_progress: AgentProgress | None = None
    async for event in events:
        progress = progress_from_event(event)
        if progress is not None and progress != last_progress:
            emit(progress)
            last_progress = progress
        root_output = root_graph_output(event)
        if root_output is not None:
            final_result = root_output
    if final_result is None:
        raise RuntimeError("Agent 事件流结束时未返回最终图状态")
    return final_result


def format_cost_cny(value: Any, *, unpriced_count: int = 0) -> str:
    """展示人民币估算费用，明确区分未知价格和真实零费用。"""
    cost = float(value or 0.0)
    if unpriced_count > 0:
        if cost > 0:
            return f"≥ ¥{cost:.4f}（另有 {unpriced_count} 次未计价）"
        return f"费用不可用（{unpriced_count} 次未计价）"
    if cost == 0:
        return "¥0.0000"
    precision = Decimal("0.0001") if cost < 0.01 else Decimal("0.01")
    rounded = Decimal(str(cost)).quantize(precision, rounding=ROUND_HALF_UP)
    return f"¥{rounded:.4f}" if cost < 0.01 else f"¥{rounded:.2f}"


def format_usage_for_user(
    usage: dict[str, Any],
    latency_ms: float,
) -> dict[str, str]:
    """把当前轮用量转换为界面展示文本。"""
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


__all__ = [
    "AgentProgress",
    "SessionUsageTracker",
    "consume_progress_events",
    "format_cost_cny",
    "format_usage_for_user",
    "progress_from_event",
    "root_graph_output",
]
