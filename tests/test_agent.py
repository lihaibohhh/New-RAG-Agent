# ruff: noqa: E402
from dotenv import load_dotenv

load_dotenv()

import asyncio
import atexit
import logging
import threading
import time
import streamlit as st
from langchain_core.messages import HumanMessage
import os

# TAVILY_API_KEY 通过 .env / 系统环境变量提供（load_dotenv() 已加载），不在源码中硬编码
if not os.environ.get("TAVILY_API_KEY"):
    logging.getLogger(__name__).warning(
        "[启动] 未检测到 TAVILY_API_KEY 环境变量，search 工具可能不可用"
    )

from react_agent.agent import AgentContext
from react_agent.conversations import (
    load_conversation_persistence_config,
    project_display_messages,
)
from react_agent.runtime import (
    close_application_services,
    create_application_services,
    warmup_application_services,
)
from react_agent.observability.display import (
    SessionUsageTracker,
    format_cost_cny,
    format_usage_for_user,
)
from react_agent.metering.turn import extract_cumulative_snapshot, extract_usage
from react_agent.observability import log_usage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── 必须是第一个 Streamlit 调用 ──────────────────────────────
st.set_page_config(page_title="Enterprise ReAct Agent", page_icon="🤖")

# ── 登录拦截：未登录不渲染任何后续内容 ──────────────────────
if "username" not in st.session_state:
    st.session_state.username = None

if st.session_state.username is None:
    st.title("请先登录")
    username = st.text_input("输入你的用户名（内部工号或姓名拼音）")
    if st.button("进入") and username.strip():
        st.session_state.username = username.strip()
        st.session_state.thread_id = f"user:{username.strip()}"
        st.rerun()
    st.stop()

# ── 以下内容仅登录后可见 ──────────────────────────────────────
st.title(f"Enterprise ReAct Agent — {st.session_state.username}")


# ── 应用服务初始化（单例，热重载不重建）─────────────────────
@st.cache_resource
def get_services_and_loop():
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    def run(coro):
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    services = run(
        create_application_services(
            agent_context=AgentContext(),
            conversation_config=load_conversation_persistence_config(),
        )
    )

    try:
        run(warmup_application_services(services))
        logger.info("[启动] 预热完成")
    except Exception as e:
        logger.info(f"[启动] 预热失败（不影响启动）：{e}")

    def close_runtime():
        if loop.is_closed():
            return
        try:
            run(close_application_services(services))
        except Exception as exc:
            logger.warning("[关闭] 应用资源释放失败：%s", exc)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=5)

    atexit.register(close_runtime)
    return services, loop


application_services, _loop = get_services_and_loop()
agent = application_services.agent


def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


# ── 会话状态初始化 ────────────────────────────────────────────
if "messages" not in st.session_state:
    try:
        persisted_messages = run_async(
            application_services.conversations.get_history(
                st.session_state.thread_id
            )
        )
        st.session_state.messages = project_display_messages(persisted_messages)
    except Exception as exc:
        logger.warning("持久化会话恢复失败，本次 UI 从空记录开始: %s", exc)
        st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ▸ 新增：会话级用量追踪器
if "usage_tracker" not in st.session_state:
    st.session_state.usage_tracker = SessionUsageTracker()
elif not hasattr(st.session_state.usage_tracker, "total_cost_cny"):
    # 开发热重载时丢弃旧版美元计价 Tracker，避免旧会话对象缺少新字段。
    st.session_state.usage_tracker = SessionUsageTracker()

# ▸ 新增：最近一轮的用量展示数据（供侧边栏渲染）
if "last_turn_display" not in st.session_state:
    st.session_state.last_turn_display = None

# ▸ 新增：上一轮结束时的累计快照（用于做差值算本轮增量）
if "prev_usage_snapshot" not in st.session_state:
    try:
        st.session_state.prev_usage_snapshot = run_async(
            agent.get_usage_snapshot(st.session_state.thread_id)
        )
    except Exception as exc:
        logger.warning("持久化用量基线读取失败，本次 UI 从零计量: %s", exc)
        st.session_state.prev_usage_snapshot = None


# ── 统计面板渲染函数 ─────────────────────────────────────────
def render_stats(placeholder):
    tracker = st.session_state.usage_tracker
    if tracker.turn_count == 0:
        return
    with placeholder.container():
        st.divider()
        st.caption("📊 本次会话统计")

        col1, col2 = st.columns(2)
        col1.metric("对话轮数", f"{tracker.turn_count}")
        col2.metric(
            "累计成本",
            format_cost_cny(
                tracker.total_cost_cny,
                unpriced_count=tracker.total_unpriced_model_calls,
            ),
        )

        col3, col4 = st.columns(2)
        col3.metric("模型调用", f"{tracker.total_llm_calls} 次")
        col4.metric("工具调用", f"{tracker.total_tool_runs} 次")

        with st.expander("查看每轮明细"):
            for turn in tracker.turn_usages:
                latency = turn["latency_ms"]
                cost_text = format_cost_cny(
                    turn.get("estimated_cost_cny", 0),
                    unpriced_count=int(turn.get("unpriced_model_count", 0)),
                )
                tokens = turn.get("total_tokens", 0)
                st.text(
                    f"第 {turn['turn']} 轮 | "
                    f"{latency / 1000:.1f}s | "
                    f"{tokens:,} tokens | "
                    f"{cost_text}"
                )

        warning = tracker.check_budget(limit_cny=7.2)
        if warning:
            st.warning(warning)


# ── 侧边栏 ────────────────────────────────────────────────────
with st.sidebar:
    st.write(f"当前用户：**{st.session_state.username}**")
    st.write(f"会话 ID：`{st.session_state.thread_id}`")

    # ▸ 预留占位符，并渲染已有历史数据
    stats_placeholder = st.empty()
    render_stats(stats_placeholder)

    st.divider()
    if st.button("退出登录"):
        st.session_state.clear()
        st.rerun()

# ── 渲染历史对话 ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # ▸ 新增：历史消息如果有 usage 信息，在底部灰色小字展示
        if msg["role"] == "assistant" and msg.get("usage_display"):
            display = msg["usage_display"]
            parts = [f"{k}: {v}" for k, v in display.items()]
            st.caption(" · ".join(parts))
        if msg["role"] == "assistant" and msg.get("tool_runs"):
            with st.expander("🛠️ 查看工具调用轨迹"):
                st.json(msg["tool_runs"])

# ── 处理新消息 ────────────────────────────────────────────────
def queue_prompt() -> None:
    """提交回调先锁定输入，避免回答渲染期间再次触发 rerun。"""
    value = str(st.session_state.get("chat_prompt") or "").strip()
    if value and not st.session_state.pending_prompt:
        st.session_state.pending_prompt = value


st.chat_input(
    "Agent 正在回答，请稍候..."
    if st.session_state.pending_prompt
    else "请输入您的问题...",
    key="chat_prompt",
    disabled=bool(st.session_state.pending_prompt),
    on_submit=queue_prompt,
)

if prompt := st.session_state.pending_prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent 正在思考并调度工具..."):
            # ▸ 新增：计时
            t0 = time.perf_counter()

            try:
                result = run_async(
                    agent.invoke(
                        [HumanMessage(content=prompt)],
                        thread_id=st.session_state.thread_id,
                    )
                )
            except Exception:
                st.session_state.pending_prompt = None
                raise

            latency_ms = (time.perf_counter() - t0) * 1000

            final_ai_msg = result["messages"][-1].content

            # ▸ 新增：三层用量处理
            # 第一层 (计量)：根据上一轮快照计算增量
            usage = extract_usage(result, st.session_state.prev_usage_snapshot)

            # 第二层 (观测)：只记录已计算的用量
            log_usage(
                usage,
                username=st.session_state.username,
                thread_id=st.session_state.thread_id,
                question=prompt,
                latency_ms=latency_ms,
            )

            # ▸ 保存本轮结束时的累计快照，供下一轮做差值
            snapshot = extract_cumulative_snapshot(result)
            st.session_state.prev_usage_snapshot = snapshot

            # 第三层 (界面)：累计到会话级追踪器
            st.session_state.usage_tracker.record_turn(usage, latency_ms)
            render_stats(stats_placeholder)  # ← 本轮数据写入后立即刷新侧边栏

            # 第四层 (用户)：生成展示文本
            usage_display = format_usage_for_user(usage, latency_ms)
            st.session_state.last_turn_display = usage_display

            current_tool_count = int(usage.get("tool_runs_count", 0))
            all_tool_runs = list(result.get("tool_runs") or [])
            current_tool_runs = (
                all_tool_runs[-current_tool_count:] if current_tool_count else []
            )

            # Agent 结果一旦返回，先原子化写入页面状态，再做展示动画。
            # 若动画期间发生 rerun，下一次渲染仍能恢复完整回答。
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_ai_msg,
                    "usage_display": usage_display,
                    "tool_runs": current_tool_runs,
                }
            )
            st.session_state.pending_prompt = None

            def stream_data(text):
                for start in range(0, len(text), 24):
                    yield text[start : start + 24]
                    time.sleep(0.01)

            st.write_stream(stream_data(final_ai_msg))

            # 工具调用轨迹（已有功能）
            if current_tool_runs:
                with st.expander("🛠️ 查看工具调用轨迹"):
                    st.json(current_tool_runs)

            # ▸ 新增：本轮用量摘要（灰色小字，不抢眼）
            parts = [f"{k}: {v}" for k, v in usage_display.items()]
            st.caption(" · ".join(parts))

    st.rerun()
