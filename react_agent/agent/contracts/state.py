"""Agent LangGraph 状态契约。

定义 Agent 的“状态结构”（工程可用版，中文情景默认）。

为什么要把 State 设计清楚？
- graph.py 的执行流以 State 为唯一“共享载体”：消息会持续追加到 state.messages 里
- tools.py 已统一返回结构（ok/data/error/meta），State 里最好预留“工具执行记录/错误统计/可复用事实”等字段
- prompting/ 强调中文回答与“需要核实时必须用工具”的规则，State 预留位置以便做：工具失败回退、事实缓存、引用整理等

注意：
- 当前图依赖 messages 与 LangGraph 注入的剩余步骤
- 下面新增字段全部“向后兼容”：不改 graph.py 也能正常跑；以后你想增强再用它们。
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from typing_extensions import Annotated


# -----------------------------
# 输入态（外部接口）
# -----------------------------
@dataclass
class InputState:
    """外部输入到图里的“最小状态”。

    目前仅暴露 messages。messages 会在整个 ReAct 循环中不断累积，典型序列为：
      1) HumanMessage：用户输入
      2) AIMessage：模型规划（可能带 tool_calls）
      3) ToolMessage：工具返回（或错误）
      4) AIMessage：最终回答（不再带 tool_calls）
      5) HumanMessage：下一轮用户输入
    其中 2-4 可重复多次。

    `add_messages` 的意义：
    - 让 messages 以“追加”为主（append-only），并按 message.id 合并更新；
    - 避免重复消息导致上下文膨胀/错乱。
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )


# -----------------------------
# 扩展态（图内部完整状态）
# -----------------------------
@dataclass
class State(InputState):
    """图内部的完整状态（在 InputState 基础上扩展）。

    设计目标（中文场景、工程可用）：
    1) 可观测：记录工具调用轨迹、错误次数、调试信息，方便定位“为什么模型没用工具/为什么没输出”
    2) 可扩展：预留 memory/citations 等字段，后续可做事实缓存、引用整理、摘要等能力
    3) 不污染：不把过大的原始数据塞进 State（工具层已做 shrink），这里只保存必要摘要/索引
    """

    # LangGraph managed：接近 recursion_limit 时置 True，仅用于异常边界观测/保险。
    is_last_step: IsLastStep = field(default=False)
    # LangGraph managed：本次图调用的剩余步骤，不写入 Checkpoint。
    remaining_steps: RemainingSteps = field(default=50)

    # -----------------------------
    # 工具调用轨迹（建议：只存“结构化摘要”，别存巨量原文）
    # -----------------------------
    # FIX-⑨: 自定义 reducer，追加后只保留最近 50 条，防止 tool_runs 无限累积占用内存
    tool_runs: Annotated[List[Dict[str, Any]], lambda old, new: (old + new)[-50:]] = (
        field(default_factory=list)
    )
    """工具调用记录（按时间追加）。

    建议每条记录形如：
    {
      "tool": "search",
      "query": "...",
      "ok": True/False,
      "error": {"code": "...", "message": "..."} | None,
      "meta": {...},
      "ts": "2026-01-12T..."
    }

    与 tools.py 的统一返回结构天然匹配：ok/data/error/meta
    """

    tool_error_count: int = 0
    """工具失败次数（后续可用于：连续失败 -> 回退回答/停止调用）。"""

    last_tool_result: Optional[Dict[str, Any]] = None
    """最近一次成功工具的信封摘要；不重复保存正文。"""

    turn_evidence: List[Dict[str, Any]] = field(default_factory=list)
    """本轮模型可见 RAG 片段的有界索引，按 chunk_id 合并。"""

    turn_evidence_omitted_count: int = 0
    """因账本容量或异常元数据而未纳入索引的候选片段数。"""

    conversation_evidence: List[Dict[str, Any]] = field(default_factory=list)
    """跨轮持久化的 RAG 来源索引；只保存查询与来源位置，不复制正文。"""

    conversation_evidence_omitted_count: int = 0
    """因容量、限幅或异常元数据未进入跨轮来源索引的累计数量。"""

    conversation_evidence_scanned_message_count: int = 0
    """跨轮来源索引已扫描到的完整消息数量，用于旧 Checkpoint 增量迁移。"""

    conversation_summary: Optional[Dict[str, Any]] = None
    """已完成旧轮次的派生摘要、来源消息 ID 与游标；不替换 messages 原文。"""

    turn_compaction_usage: Optional[Dict[str, Any]] = None
    """本轮额外摘要模型调用的用量；供非流式 API 精确计入配额。"""

    compaction_control: Optional[Dict[str, Any]] = None
    """历史压缩的持久化控制记录。

    记录最近候选批次、拒绝原因和下次重试所需的最小来源 Token，
    避免同一批不可压缩历史在每个用户轮次反复付费。
    """

    # -----------------------------
    # 事实缓存与工作记忆（中文问答很实用）
    # -----------------------------
    memory: Dict[str, Any] = field(default_factory=dict)
    """跨步/跨轮可复用的“工作记忆”。

    推荐用途：
    - 缓存已确认的事实（例如搜索到的权威结论）
    - 缓存用户偏好（语言、格式要求）
    - 缓存中间产物（对搜索结果的摘要、结构化结论）
    """

    citations: Annotated[List[Dict[str, str]], operator.add] = field(
        default_factory=list
    )
    """引用/来源清单（可选，中文场景常需要“来源提示”）。

    每条建议形如：
    {"title": "...", "url": "..."} 或 {"source": "...", "note": "..."}
    """

    # -----------------------------
    # 调试与可观测性
    # -----------------------------
    debug_log: Annotated[List[str], operator.add] = field(default_factory=list)
    """调试日志（按步追加）。"""

    step_counter: int = 0
    """你自己的步数计数（可选）。"""

    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    prompt_tokens: float = 0.0
    total_tokens: int = 0
    # 旧 Checkpoint 反序列化兼容占位；不再参与新调用的计价或展示。
    estimated_cost_usd: float = 0.0
    estimated_cost_cny: float = 0.0
    unpriced_model_count: int = 0
    pricing_requested_model: str = ""
    pricing_response_model: str = ""
    pricing_model: str = ""
    pricing_tariff: str = ""
    pricing_rate_card_version: str = ""
    llm_call_count: int = 0

    pending_directive: Optional[str] = None
    """瞬态系统控制指令（如反思提示）。reflection_node 写入，call_model 读取后作为 SystemMessage
    拼入当次模型输入并随即清空；不进入 messages 历史，避免过期提示堆积。"""

    # 在现有字段后面添加

    consecutive_failures: int = 0
    """连续工具失败次数（用于检测重复错误）"""

    # -----------------------------
    # 当前用户轮次的业务预算
    # -----------------------------
    turn_model_rounds: int = 0
    """本轮常规 Agent 模型调用次数；不包含预留的最终总结调用。"""

    turn_tool_batches: int = 0
    """本轮已经进入工具执行阶段的批次数。"""

    turn_tool_retries: int = 0
    """本轮工具失败后已经发给模型的恢复指令次数。"""

    termination_reason: Optional[str] = None
    """业务预算或证据策略要求主动收口时的结构化原因。"""

    last_tool_batch_errors: List[Dict[str, Any]] = field(default_factory=list)
    """最近一个工具批次的全部错误；每批覆盖，不跨批累积。"""

    # FIX-⑪: 删除三个从未实现的字段（全局 grep 确认无其他文件引用）：
    #   tool_failure_counts — 无实现，无读取方
    #   tool_blacklist      — 无实现，无读取方
    #   reflection_count    — 无实现，无读取方
