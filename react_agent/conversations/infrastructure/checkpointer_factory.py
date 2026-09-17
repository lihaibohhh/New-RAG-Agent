from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, NoReturn, Optional, Protocol, Tuple

from langgraph.checkpoint.base import BaseCheckpointSaver

from react_agent.conversations.configuration import normalize_checkpoint_backend
from react_agent.conversations.contracts import (
    ConversationPersistenceInitializationError,
)
from react_agent.configuration.settings import resolve_app_path, settings


class CheckpointConfig(Protocol):
    """Configuration shape required to select a Checkpointer backend."""

    checkpoint_backend: str
    checkpoint_db_path: Optional[str]


# 安全默认值：若调用方没有显式覆盖，限制 Checkpoint 反序列化类型。
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

try:
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
except ImportError:
    dict_row = None
    AsyncConnectionPool = None
    AsyncPostgresSaver = None


class CheckpointerFactory:
    """
    Checkpointer 工厂（工程级实现）

    设计原则
    --------
    1. 实例边界：每个顶层 ApplicationServices 持有自己的缓存与生命周期。

    2. 懒初始化锁：asyncio.Lock 在首次调用时创建，绑定到当前 event loop，
       避免 Streamlit 热重载 / FastAPI 多次启动时因 loop 销毁导致的 RuntimeError。

    3. 精确 cache key：在锁外预计算含连接参数的 key（如 "postgres:postgresql://..."），
       锁内做精确 dict.get 查询，杜绝前缀扫描 + 字典无锁迭代的竞态问题。

    4. 持久化语义：显式请求 PostgreSQL 时初始化失败即终止启动，禁止降级到
       MemorySaver；SQLite 失败仍可在本地开发场景共享同一个 MemorySaver。

    5. 生命周期解耦：context manager / connection pool 通过独立注册表 _lifecycle 管理，
       不给第三方对象打补丁（猴子补丁污染命名空间且脆弱）。

    6. 双 key 缓存：允许降级的 backend 同时在原始 key 下缓存 fallback 实例，
       防止后续请求重复尝试已知失败的 SQLite backend。
    """

    _logger = logging.getLogger(__name__)
    _DEFAULT_SQLITE_PATH = "./data/agent-state/agent_checkpoints.sqlite3"

    def __init__(self) -> None:
        # 缓存和外部资源归属于单个 ApplicationServices，不再是进程全局状态。
        self._instances: Dict[str, BaseCheckpointSaver] = {}
        self._lifecycle: Dict[str, Dict[str, Any]] = {}
        self._effective_backends: Dict[int, str] = {}
        # 保持懒创建，避免工厂在 event loop 之外实例化时绑定错误 loop。
        self._lock: Optional[asyncio.Lock] = None

    # ─────────────────────────── 内部工具 ───────────────────────────

    def _get_lock(self) -> asyncio.Lock:
        """
        懒创建锁。

        关键：asyncio.Lock 必须在当前 event loop 中创建，
        若在模块加载时（loop 启动前）或 loop 销毁后创建，会绑定到错误的 loop。
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _log(self, msg: str, level: str = "info") -> None:
        """统一日志出口，禁止 print()（MCP 环境下 print 会污染 stdout 协议流）。"""
        getattr(self._logger, level, self._logger.info)(msg)

    def _normalize_backend(self, backend: str) -> str:
        return normalize_checkpoint_backend(backend)

    def _sqlite_db_path(self, ctx: CheckpointConfig) -> Path:
        raw = getattr(ctx, "checkpoint_db_path", None) or self._DEFAULT_SQLITE_PATH
        # 相对路径固定锚定到应用根目录，不再随启动命令的 cwd 漂移。
        return Path(resolve_app_path(raw))

    def _postgres_conn_str(self) -> str:
        return settings.postgres.POSTGRES_DB_URL.get_secret_value().strip()

    def _compute_cache_key(self, backend: str, ctx: CheckpointConfig) -> str:
        """
        在锁外预计算 cache key。

        含连接参数的精确 key 确保：
        - 同一 backend 不同连接目标不会复用同一实例
        - 进入锁后只需 dict.get(key) 一次，无需迭代
        """
        if backend == "sqlite":
            return f"sqlite:{self._sqlite_db_path(ctx)}"
        if backend == "postgres":
            conn_fingerprint = hashlib.sha256(
                self._postgres_conn_str().encode("utf-8")
            ).hexdigest()[:16]
            return f"postgres:{conn_fingerprint}"
        return backend  # "memory"

    def _register_lifecycle(
        self, key: str, *, cm: Any = None, pool: Any = None
    ) -> None:
        """注册需要在 close() 时清理的外部资源（在锁内调用）。"""
        self._lifecycle[key] = {"cm": cm, "pool": pool}

    # ─────────────────────────── 生命周期管理 ───────────────────────────

    async def close(self) -> None:
        """
        关闭当前工厂已注册的连接资源，并重置实例状态。

        注意：重置 _lock = None 须在退出 async with 块之后，
        否则 __aexit__ 会找不到锁对象。
        """
        lock = self._get_lock()
        async with lock:
            for key, resources in self._lifecycle.items():
                cm = resources.get("cm")
                if cm is not None:
                    try:
                        await cm.__aexit__(None, None, None)
                        self._log(f"[Checkpointer] 已关闭 CM: {key}")
                    except Exception as e:
                        self._log(
                            f"[Checkpointer] 关闭 CM 失败 ({key}): {e}", "warning"
                        )

                pool = resources.get("pool")
                if pool is not None:
                    try:
                        await pool.close()
                        self._log(f"[Checkpointer] 已关闭连接池: {key}")
                    except Exception as e:
                        self._log(
                            f"[Checkpointer] 关闭连接池失败 ({key}): {e}", "warning"
                        )

            self._instances.clear()
            self._lifecycle.clear()
            self._effective_backends.clear()

        # 退出 async with 后再重置锁，确保下次在新 loop 中重建
        self._lock = None

    # ─────────────────────────── 对外入口 ───────────────────────────

    async def create(self, ctx: CheckpointConfig) -> Optional[BaseCheckpointSaver]:
        backend = self._normalize_backend(getattr(ctx, "checkpoint_backend", ""))

        if backend == "none":
            return None

        # 在锁外预计算 key（可能涉及文件系统路径解析 / os.getenv，成本低但避免在锁内做）
        cache_key = self._compute_cache_key(backend, ctx)

        # 快速路径：无锁单 key 查询（CPython dict.get 是原子操作，无需加锁）
        inst = self._instances.get(cache_key)
        if inst is not None:
            return inst

        # 慢路径：加锁创建
        async with self._get_lock():
            # 双重检查（等待锁期间可能已被其他协程创建）
            inst = self._instances.get(cache_key)
            if inst is not None:
                return inst

            instance, effective_key = await self._create_instance(
                backend, ctx, cache_key
            )

            if instance is not None and effective_key:
                self._instances[effective_key] = instance
                self._effective_backends[id(instance)] = effective_key.split(":", 1)[0]
                # 允许降级的 backend 同时在原始 key 下缓存，避免重复初始化。
                # PostgreSQL 初始化失败会抛出异常，不会进入此分支。
                if effective_key != cache_key:
                    self._instances[cache_key] = instance

            return instance

    def effective_backend(self, checkpointer: Optional[BaseCheckpointSaver]) -> str:
        """返回实际生效的后端，而非请求配置值。"""
        if checkpointer is None:
            return "none"
        return self._effective_backends.get(id(checkpointer), "unknown")

    # ─────────────────────────── 各 Backend 创建逻辑 ───────────────────────────

    async def _create_instance(
        self, backend: str, ctx: CheckpointConfig, cache_key: str
    ) -> Tuple[Optional[BaseCheckpointSaver], str]:
        """分发到具体 backend 创建函数（在锁内调用）。"""
        if backend == "memory":
            return self._create_memory(), "memory"
        if backend == "sqlite":
            return await self._create_sqlite(ctx, cache_key)
        if backend == "postgres":
            return await self._create_postgres(cache_key)
        self._log(f"[Checkpointer] 未知 backend '{backend}'，禁用持久化", "warning")
        return None, ""

    def _create_memory(self) -> Optional[BaseCheckpointSaver]:
        try:
            from langgraph.checkpoint.memory import MemorySaver

            self._log(
                "[Checkpointer] 使用 MemorySaver（仅限本地调试，"
                "无持久化，高并发存在 OOM 风险，禁止用于生产环境）",
                "warning",
            )
            inst = MemorySaver()
            self._register_lifecycle("memory")  # 无外部资源，注册空记录以保持注册表完整
            return inst
        except ImportError as e:
            self._log(f"[Checkpointer] ❌ 无法导入 MemorySaver: {e}", "error")
            return None

    def _fallback_to_memory(
        self, reason: str
    ) -> Tuple[Optional[BaseCheckpointSaver], str]:
        """
        统一降级入口（必须在锁内调用）。

        仅供允许降级的本地 backend 使用。显式请求 PostgreSQL 时不得调用本方法。
        多次 SQLite 降级共享同一个实例，避免同一 thread_id 状态分裂。
        """
        self._log(f"[Checkpointer] ⚠️ {reason}，降级到 MemorySaver", "warning")
        existing = self._instances.get("memory")
        if existing is not None:
            return existing, "memory"
        inst = self._create_memory()
        return (inst, "memory") if inst is not None else (None, "")

    def _raise_postgres_unavailable(
        self,
        reason: str,
        *,
        cause: BaseException | None = None,
    ) -> NoReturn:
        message = (
            f"PostgreSQL Checkpointer 不可用：{reason}。"
            "已禁止降级到 MemorySaver，应用启动已终止。"
        )
        self._log(f"[Checkpointer] ❌ {message}", "error")
        error = ConversationPersistenceInitializationError(message)
        if cause is not None:
            raise error from cause
        raise error

    async def _create_sqlite(
        self, ctx: CheckpointConfig, ok_key: str
    ) -> Tuple[Optional[BaseCheckpointSaver], str]:
        """创建 SQLite checkpointer。"""
        db_path = self._sqlite_db_path(ctx)

        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as e:
            return self._fallback_to_memory(f"AsyncSqliteSaver 不可用: {e}")

        db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._log(f"[Checkpointer] 初始化 SQLite: {db_path}")
            cm = AsyncSqliteSaver.from_conn_string(str(db_path))
            checkpointer = await cm.__aenter__()
            # 显式幂等建表，与 Postgres / Redis 行为保持一致（不依赖 __aenter__ 副作用）
            await checkpointer.setup()

            self._register_lifecycle(ok_key, cm=cm)
            self._log(f"[Checkpointer] ✅ SQLite 已启用: {db_path}")
            return checkpointer, ok_key

        except Exception as e:
            self._log(
                f"[Checkpointer] ❌ SQLite 初始化失败: {type(e).__name__}: {e}", "error"
            )
            return self._fallback_to_memory("SQLite 初始化失败")

    async def _create_postgres(
        self, ok_key: str
    ) -> Tuple[Optional[BaseCheckpointSaver], str]:
        """创建 PostgreSQL checkpointer。"""
        if (
            AsyncPostgresSaver is None
            or AsyncConnectionPool is None
            or dict_row is None
        ):
            self._raise_postgres_unavailable(
                "缺少依赖：pip install langgraph-checkpoint-postgres psycopg[binary,pool]"
            )

        conn_str = self._postgres_conn_str()
        if not conn_str:
            self._raise_postgres_unavailable("POSTGRES_DB_URL 未配置")

        pg_cfg = settings.postgres
        min_size = pg_cfg.POSTGRES_POOL_MIN_SIZE
        max_size = pg_cfg.POSTGRES_POOL_MAX_SIZE
        connect_timeout = pg_cfg.POSTGRES_CONNECT_TIMEOUT_SECONDS

        pool = None
        try:
            self._log("[Checkpointer] 初始化 Postgres 连接池...")
            pool = AsyncConnectionPool(
                conn_str,
                min_size=min_size,
                max_size=max_size,
                open=False,
                kwargs={"autocommit": True, "row_factory": dict_row},
            )
            # 使用 asyncio.wait_for 包裹 pool.open()：
            # 直接调用无超时保护，Postgres 不可达时会永久挂起整个进程。
            # wait_for 兼容所有 psycopg_pool 版本，无需依赖 open(timeout=...) 参数。
            await asyncio.wait_for(pool.open(wait=True), timeout=connect_timeout)

            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()

            self._register_lifecycle(ok_key, pool=pool)
            self._log("[Checkpointer] ✅ Postgres 已启用")
            return checkpointer, ok_key

        except asyncio.TimeoutError:
            if pool is not None:
                try:
                    await pool.close()
                except Exception:
                    pass
            self._raise_postgres_unavailable(
                f"连接池初始化超时（{connect_timeout}s），请检查连接配置和网络连通性"
            )
        except Exception as e:
            if pool is not None:
                try:
                    await pool.close()
                except Exception:
                    pass
            self._raise_postgres_unavailable(
                f"初始化失败（{type(e).__name__}）",
                cause=e,
            )
