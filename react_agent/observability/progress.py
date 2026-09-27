"""Map LangChain events to safe, user-facing Agent progress updates."""

from __future__ import annotations

from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from agent_tools.presentation import TOOL_PROGRESS_LABELS


@dataclass(frozen=True)
class AgentProgress:
    """A small progress update that is safe to render in an inbound adapter."""

    stage: str
    label: str
    state: Literal["running", "complete"] = "running"


def _safe_tool_name(value: Any) -> str:
    raw = str(value or "").strip()
    safe = "".join(character for character in raw if character.isalnum() or character in "_-")
    return safe[:48] or "unknown"


def _is_compaction_event(event: Mapping[str, Any]) -> bool:
    metadata = event.get("metadata")
    node = metadata.get("langgraph_node") if isinstance(metadata, Mapping) else None
    return node == "compact_history" or "context_compaction" in tuple(
        event.get("tags") or ()
    )


def progress_from_event(event: Mapping[str, Any]) -> AgentProgress | None:
    """Return a bounded progress label without exposing prompts or tool payloads."""
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
        return AgentProgress(
            stage="model_stream",
            label="正在生成回答……",
        )

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
    """Extract the final graph state from the root ``on_chain_end`` event."""
    if event.get("event") != "on_chain_end" or event.get("parent_ids"):
        return None
    data = event.get("data")
    output = data.get("output") if isinstance(data, Mapping) else None
    return dict(output) if isinstance(output, Mapping) else None


async def consume_progress_events(
    events: AsyncIterable[Mapping[str, Any]],
    emit: Callable[[AgentProgress], None],
) -> dict[str, Any]:
    """Consume one graph run, emit deduplicated progress, and return final state."""
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


__all__ = [
    "AgentProgress",
    "consume_progress_events",
    "progress_from_event",
    "root_graph_output",
]
