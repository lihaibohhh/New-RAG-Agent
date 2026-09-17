"""用户轮次初始化与失败恢复节点。"""

from __future__ import annotations

import logging
from typing import Any

from react_agent.agent.contracts.state import State
from react_agent.agent.prompting.directives import render_tool_recovery_directive


logger = logging.getLogger(__name__)


async def prepare_turn(state: State) -> dict[str, Any]:
    """在每次新用户输入进入图时重置当前轮业务状态。"""
    return {
        "turn_model_rounds": 0,
        "turn_tool_batches": 0,
        "turn_tool_retries": 0,
        "termination_reason": None,
        "last_tool_batch_errors": [],
        "consecutive_failures": 0,
        "pending_directive": None,
    }


async def reflection_node(state: State) -> dict[str, Any]:
    """将最近工具批次错误转换为一次受预算约束的恢复指令。"""
    if not state.last_tool_batch_errors:
        return {}
    logger.warning(
        "tool_batch_recovery | errors=%s retry=%s",
        len(state.last_tool_batch_errors),
        state.turn_tool_retries + 1,
    )
    return {
        "pending_directive": render_tool_recovery_directive(
            state.last_tool_batch_errors
        ),
        "turn_tool_retries": state.turn_tool_retries + 1,
    }


__all__ = ["prepare_turn", "reflection_node"]
