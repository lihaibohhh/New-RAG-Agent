"""Agent 内部的工具调用协议与执行编排。"""

from react_agent.agent.tool_flow.execution import (
    bound_tool_messages,
    execute_dynamic_tools,
)
from react_agent.agent.tool_flow.payload import bound_tool_payload
from react_agent.agent.tool_flow.protocol import (
    SYSTEM_SENTINEL_NAMES,
    close_tool_calls_for_budget,
    extract_recent_tool_messages,
    extract_tool_call_ids,
    find_last_real_human_index,
    get_tool_call_name,
)
from react_agent.agent.tool_flow.results import (
    ToolBatch,
    count_attempted_rag_calls_in_current_turn,
    count_successful_rag_calls_in_current_turn,
    parse_tool_batch,
)


__all__ = [
    "SYSTEM_SENTINEL_NAMES",
    "ToolBatch",
    "bound_tool_messages",
    "bound_tool_payload",
    "close_tool_calls_for_budget",
    "count_attempted_rag_calls_in_current_turn",
    "count_successful_rag_calls_in_current_turn",
    "execute_dynamic_tools",
    "extract_recent_tool_messages",
    "extract_tool_call_ids",
    "find_last_real_human_index",
    "get_tool_call_name",
    "parse_tool_batch",
]
