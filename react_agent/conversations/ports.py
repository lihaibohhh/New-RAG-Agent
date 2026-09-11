"""Outbound ports owned by the conversation module."""
from __future__ import annotations

from typing import Any, List, Optional, Protocol

from react_agent.conversations.contracts import ConversationDeleteStatus


class ConversationRepositoryPort(Protocol):
    """Persistence capability required by ``ConversationService``."""

    async def get_messages(self, thread_id: str) -> Optional[List[Any]]:
        """Return the latest message list, or ``None`` when the thread is absent."""

    async def delete(self, thread_id: str) -> ConversationDeleteStatus:
        """Delete all persisted state associated with a thread."""

