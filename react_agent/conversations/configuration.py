"""Conversation persistence configuration loading."""

from __future__ import annotations

import os
from collections.abc import Mapping

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
    return aliases.get(value, value)


def load_conversation_persistence_config(
    environ: Mapping[str, str] | None = None,
) -> ConversationPersistenceConfig:
    """Load the process-wide conversation persistence selection.

    PostgreSQL is enabled explicitly with ``CHECKPOINT_BACKEND=postgres``.
    SQLite remains the safe local fallback when the variable is absent.
    """
    source = os.environ if environ is None else environ
    backend = (
        source.get("CHECKPOINT_BACKEND", DEFAULT_CHECKPOINT_BACKEND).strip().lower()
        or DEFAULT_CHECKPOINT_BACKEND
    )
    db_path = (
        source.get("CHECKPOINT_DB_PATH", DEFAULT_CHECKPOINT_DB_PATH).strip()
        or DEFAULT_CHECKPOINT_DB_PATH
    )
    return ConversationPersistenceConfig(
        checkpoint_backend=backend,
        checkpoint_db_path=db_path,
    )


__all__ = [
    "DEFAULT_CHECKPOINT_BACKEND",
    "DEFAULT_CHECKPOINT_DB_PATH",
    "load_conversation_persistence_config",
    "normalize_checkpoint_backend",
]
