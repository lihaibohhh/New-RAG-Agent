"""Conversation-management use cases."""
from __future__ import annotations

from typing import Any, List, Optional

from react_agent.conversations.contracts import ConversationDeleteStatus
from react_agent.conversations.ports import ConversationRepositoryPort


class ConversationService:
    """Owns conversation history and deletion use cases."""

    def __init__(self, repository: ConversationRepositoryPort):
        self._repository = repository

    async def get_history(self, thread_id: str) -> Optional[List[Any]]:
        return await self._repository.get_messages(_require_thread_id(thread_id))

    async def delete(self, thread_id: str) -> ConversationDeleteStatus:
        return await self._repository.delete(_require_thread_id(thread_id))


def _require_thread_id(thread_id: str) -> str:
    normalized = (thread_id or "").strip()
    if not normalized:
        raise ValueError("thread_id 不能为空")
    return normalized

