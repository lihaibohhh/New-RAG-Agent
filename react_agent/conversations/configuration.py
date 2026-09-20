"""Conversation persistence configuration loading."""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import SecretStr

from react_agent.configuration.settings import settings
from react_agent.conversations.contracts import ConversationPersistenceConfig


DEFAULT_CHECKPOINT_BACKEND = "sqlite"
DEFAULT_CHECKPOINT_DB_PATH = "./data/agent-state/agent_checkpoints.sqlite3"


def normalize_checkpoint_backend(backend: str) -> str:
    """Normalize backend aliases for selection and readiness comparison."""
    value = (backend or "").strip().lower()
    aliases = {
        "": "memory",
        "mem": "memory",
        "sqlite3": "sqlite",
        "aiosqlite": "sqlite",
        "postgresql": "postgres",
        "pg": "postgres",
        "off": "none",
        "false": "none",
        "0": "none",
    }
    normalized = aliases.get(value, value)
    if normalized not in {"memory", "sqlite", "postgres", "none"}:
        raise ValueError(
            f"不支持的 CHECKPOINT_BACKEND: {backend!r}；"
            "可选 memory / sqlite / postgres / none"
        )
    return normalized


def load_conversation_persistence_config(
    environ: Mapping[str, str] | None = None,
) -> ConversationPersistenceConfig:
    """Load the process-wide conversation persistence selection.

    PostgreSQL is enabled explicitly with ``CHECKPOINT_BACKEND=postgres``.
    SQLite remains the safe local fallback when the variable is absent.
    """
    source = os.environ if environ is None else environ
    requested_backend = (
        source.get("CHECKPOINT_BACKEND", DEFAULT_CHECKPOINT_BACKEND).strip().lower()
        or DEFAULT_CHECKPOINT_BACKEND
    )
    backend = normalize_checkpoint_backend(requested_backend)
    db_path = (
        source.get("CHECKPOINT_DB_PATH", DEFAULT_CHECKPOINT_DB_PATH).strip()
        or DEFAULT_CHECKPOINT_DB_PATH
    )
    if environ is None:
        postgres = settings.postgres
        postgres_db_url = postgres.POSTGRES_DB_URL
        pool_min_size = postgres.POSTGRES_POOL_MIN_SIZE
        pool_max_size = postgres.POSTGRES_POOL_MAX_SIZE
        connect_timeout = postgres.POSTGRES_CONNECT_TIMEOUT_SECONDS
    else:
        # Explicit mapping is a self-contained test/script input; never mix in
        # credentials or pool settings from the host process.
        postgres_db_url = SecretStr(source.get("POSTGRES_DB_URL", "").strip())
        pool_min_size = int(source.get("POSTGRES_POOL_MIN_SIZE", "1"))
        pool_max_size = int(source.get("POSTGRES_POOL_MAX_SIZE", "10"))
        connect_timeout = float(source.get("POSTGRES_CONNECT_TIMEOUT_SECONDS", "5"))
    return ConversationPersistenceConfig(
        checkpoint_backend=backend,
        checkpoint_db_path=db_path,
        postgres_db_url=postgres_db_url,
        postgres_pool_min_size=pool_min_size,
        postgres_pool_max_size=pool_max_size,
        postgres_connect_timeout_seconds=connect_timeout,
    )


__all__ = [
    "DEFAULT_CHECKPOINT_BACKEND",
    "DEFAULT_CHECKPOINT_DB_PATH",
    "load_conversation_persistence_config",
    "normalize_checkpoint_backend",
]
