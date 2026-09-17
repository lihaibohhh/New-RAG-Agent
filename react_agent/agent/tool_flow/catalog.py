"""将运行时可用工具渲染为模型可读目录。"""

from __future__ import annotations

from typing import Any


def render_tool_catalog(tools: tuple[Any, ...], *, tools_enabled: bool) -> str:
    if not tools_enabled:
        return "当前运行配置已禁用工具调用（tools_enabled=false），本轮不允许使用任何工具。"
    if not tools:
        return "当前未配置任何工具。"

    lines = ["你当前可使用以下工具："]
    for index, tool in enumerate(tools, start=1):
        name = getattr(tool, "name", None) or getattr(tool, "__name__", "unknown_tool")
        description = " ".join(str(getattr(tool, "description", "") or "").split())
        if len(description) > 240:
            description = description[:239] + "..."
        detail = description or "暂无描述（建议为工具增加 description）"
        lines.append(f"{index}) {name}：{detail}")
    lines.append(
        "当用户询问'你有哪些工具/能做什么'时，只能按上述目录回答，不得编造。"
        "如果工具调用失败，请根据错误原因调整参数后重试，不要盲目重复。"
    )
    return "\n".join(lines)


__all__ = ["render_tool_catalog"]
