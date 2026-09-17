"""失败恢复与主动收口使用的瞬态控制指令。"""

from __future__ import annotations

from typing import Any


def render_tool_recovery_directive(errors: list[dict[str, Any]]) -> str:
    """把当前工具批次错误渲染为一次瞬态恢复指令。"""
    summaries: list[str] = []
    for run in errors:
        error_info = run.get("error") or "未知错误"
        if isinstance(error_info, dict):
            error_info = (
                f"{error_info.get('code', 'ERR')}: "
                f"{error_info.get('message', str(error_info))}"
            )
        summaries.append(
            f"- {run.get('tool', 'unknown_tool')}: {str(error_info)[:300]}"
        )
    return (
        "【工具恢复指令】上一批工具调用存在失败：\n"
        + "\n".join(summaries)
        + "\n请分析错误原因并修正参数；不要以相同参数盲目重复调用。"
        "若错误不可恢复，请基于已有结果直接回答并明确说明限制。"
    )


def render_finalization_directive(reason: str | None) -> str:
    """按主动终止原因生成禁止继续调用工具的最终回答指令。"""
    descriptions = {
        "MODEL_ROUND_BUDGET_EXHAUSTED": "常规模型推理轮次已达到上限",
        "TOOL_BATCH_BUDGET_EXHAUSTED": "工具执行批次已达到上限",
        "TOOL_RETRY_BUDGET_EXHAUSTED": "工具失败恢复次数已达到上限",
        "RAG_CALL_BUDGET_EXHAUSTED": "内部知识库调用次数已达到上限",
        "RAG_CONSECUTIVE_MISS": "内部知识库连续未检索到相关内容",
    }
    detail = descriptions.get(reason or "", "本轮 Agent 已进入主动收口阶段")
    return (
        f"【最终回答指令】{detail}。禁止继续调用任何工具。"
        "请仅根据当前对话和已经取得的工具结果生成最终用户答复。"
        "优先总结已确认的信息；证据不足或工具失败的部分必须明确说明，"
        "不得猜测、补造数据，也不要向用户暴露 recursion_limit 等内部实现参数。"
    )


__all__ = [
    "render_finalization_directive",
    "render_tool_recovery_directive",
]
