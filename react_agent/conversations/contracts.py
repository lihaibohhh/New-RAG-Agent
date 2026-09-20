"""Public contracts for conversation management."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import SecretStr


@dataclass(frozen=True)
class ConversationPersistenceConfig:
    """Complete, immutable configuration injected into checkpoint infrastructure."""

    checkpoint_backend: str = "sqlite"
    checkpoint_db_path: Optional[str] = "./data/agent-state/agent_checkpoints.sqlite3"
    postgres_db_url: SecretStr = field(default_factory=lambda: SecretStr(""), repr=False)
    postgres_pool_min_size: int = 1
    postgres_pool_max_size: int = 10
    postgres_connect_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.postgres_db_url, str):
            object.__setattr__(self, "postgres_db_url", SecretStr(self.postgres_db_url))
        if self.postgres_pool_min_size < 1 or self.postgres_pool_max_size < 1:
            raise ValueError("PostgreSQL 连接池大小必须大于零")
        if self.postgres_pool_min_size > self.postgres_pool_max_size:
            raise ValueError("POSTGRES_POOL_MIN_SIZE 不能大于 POSTGRES_POOL_MAX_SIZE")
        if self.postgres_connect_timeout_seconds <= 0:
            raise ValueError("POSTGRES_CONNECT_TIMEOUT_SECONDS 必须大于零")


@dataclass(frozen=True)
class ConversationMessage:
    """Typed, storage-independent message view returned to conversation clients."""

    type: str
    content: str
    id: str | None = None
    has_tool_calls: bool = False


class ConversationDeleteStatus(str, Enum):
    """Stable deletion outcomes exposed by the conversation use case."""

    DELETED = "deleted"
    NOT_FOUND = "not_found"
    UNSUPPORTED = "unsupported"


class ConversationRepositoryError(RuntimeError):
    """Raised when the configured conversation store cannot complete an operation."""


class ConversationPersistenceInitializationError(RuntimeError):
    """Raised when a required durable conversation backend cannot start."""
