from __future__ import annotations
import asyncio
import logging
import os
from react_agent.agent import AgentContext, AgentService
from react_agent.conversations import ConversationPersistenceConfig, ConversationService
from react_agent.runtime import (
    ApplicationServices,
    ApplicationStatus,
    close_application_services,
    create_application_services,
    get_application_status,
)


_logger = logging.getLogger(__name__)

# ── 全局单例 ──────────────────────────────────────────────────────────────────
# 和 Streamlit 侧的 @st.cache_resource 类似，但 FastAPI 用模块级单例。
# 整个进程只初始化一次，避免重复编译 LangGraph + 初始化 checkpointer。
_services: ApplicationServices | None = None
_services_lock: asyncio.Lock | None = None


async def _get_services() -> ApplicationServices:
    """Create the application object graph once per process/event loop."""
    global _services, _services_lock
    if _services is not None:
        return _services
    if _services_lock is None:
        _services_lock = asyncio.Lock()
    async with _services_lock:
        if _services is None:
            agent_context = AgentContext(
                model="deepseek/deepseek-chat",
            )
            conversation_config = ConversationPersistenceConfig(
                checkpoint_backend="sqlite",
                checkpoint_db_path=os.getenv(
                    "CHECKPOINT_DB_PATH",
                    "./data/agent-state/agent_checkpoints.sqlite3",
                ),
            )
            _services = await create_application_services(
                agent_context=agent_context,
                conversation_config=conversation_config,
            )
            _logger.info(
                "[API] 应用服务初始化完成，requested_backend=%s effective_backend=%s",
                conversation_config.checkpoint_backend,
                _services.effective_checkpoint_backend,
            )
    return _services


async def get_agent() -> AgentService:
    """
    FastAPI Depends 注入函数。
    注入 Agent 对话用例。保留函数名以兼容现有 FastAPI Depends。

    用法：
        @router.post("/chat/stream")
        async def chat_stream(req: ChatRequest, agent: AgentService = Depends(get_agent)):
            ...
    """
    return (await _get_services()).agent


async def get_conversation_service() -> ConversationService:
    """注入会话管理用例；不会向路由暴露 Agent 或 Checkpointer。"""
    return (await _get_services()).conversations


async def get_runtime_status() -> ApplicationStatus:
    """注入健康检查需要的只读状态，不暴露完整对象容器。"""
    return get_application_status(await _get_services())


async def startup_init() -> None:
    """
    FastAPI lifespan 启动钩子中调用，提前预热 Agent，
    避免第一个请求触发冷启动（模型加载 ~10s）。
    """
    await get_agent()
    _logger.info("[API] Agent 预热完成，服务就绪")


async def shutdown_services() -> None:
    """关闭 Composition Root 持有的外部资源并清空进程内实例。"""
    global _services, _services_lock
    if _services is not None:
        await close_application_services(_services)
    _services = None
    _services_lock = None
