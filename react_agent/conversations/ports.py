"""Outbound ports owned by the conversation module."""
from __future__ import annotations

from typing import Protocol

from react_agent.conversations.contracts import (
    ConversationDeleteStatus,
    ConversationMessage,
)


class ConversationRepositoryPort(Protocol):
    """Persistence capability required by ``ConversationService``."""

    async def get_messages(self, thread_id: str) -> list[ConversationMessage] | None:
        """Return messages in the latest checkpoint, or ``None`` if absent."""

    async def delete(self, thread_id: str) -> ConversationDeleteStatus:
        """Delete all persisted state associated with a thread."""
