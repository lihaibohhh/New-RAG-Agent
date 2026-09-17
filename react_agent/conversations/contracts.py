"""Public contracts for conversation management."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


@dataclass(frozen=True)
class ConversationPersistenceConfig:
    """Configuration consumed by conversation persistence infrastructure."""

    checkpoint_backend: str = "memory"
    checkpoint_db_path: Optional[str] = None


class ConversationDeleteStatus(str, Enum):
    """Stable deletion outcomes exposed by the conversation use case."""

    DELETED = "deleted"
    NOT_FOUND = "not_found"
    UNSUPPORTED = "unsupported"


class ConversationRepositoryError(RuntimeError):
    """Raised when the configured conversation store cannot complete an operation."""


class ConversationPersistenceInitializationError(RuntimeError):
    """Raised when a required durable conversation backend cannot start."""
