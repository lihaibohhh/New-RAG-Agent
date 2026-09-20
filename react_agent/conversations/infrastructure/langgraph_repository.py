"""LangGraph Checkpointer adapter for conversation-management use cases."""
from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver

from react_agent.conversations.contracts import (
    ConversationDeleteStatus,
    ConversationMessage,
    ConversationRepositoryError,
)

_logger = logging.getLogger(__name__)


class LangGraphConversationRepository:
    """Expose conversation operations without leaking Checkpointer details upstream."""

    def __init__(self, checkpointer: Optional[BaseCheckpointSaver]):
        self._checkpointer = checkpointer

    async def get_messages(self, thread_id: str) -> list[ConversationMessage] | None:
        if self._checkpointer is None:
            return None

        try:
            checkpoint = await self._checkpointer.aget_tuple(_config(thread_id))
        except Exception as exc:
            _logger.error(
                "conversation_history_failed | thread_id=%s | err_type=%s",
                thread_id,
                type(exc).__name__,
            )
            raise ConversationRepositoryError("读取会话历史失败") from exc

        if checkpoint is None:
            return None
        values = checkpoint.checkpoint.get("channel_values", {})
        return [_message_view(message) for message in values.get("messages", [])]

    async def delete(self, thread_id: str) -> ConversationDeleteStatus:
        if self._checkpointer is None:
            return ConversationDeleteStatus.NOT_FOUND

        try:
            checkpoint = await self._checkpointer.aget_tuple(_config(thread_id))
        except Exception as exc:
            _logger.error(
                "conversation_lookup_failed | thread_id=%s | err_type=%s",
                thread_id,
                type(exc).__name__,
            )
            raise ConversationRepositoryError("确认会话是否存在时失败") from exc

        if checkpoint is None:
            return ConversationDeleteStatus.NOT_FOUND

        public_delete = getattr(self._checkpointer, "adelete_thread", None)
        if callable(public_delete):
            try:
                await public_delete(thread_id)
                return ConversationDeleteStatus.DELETED
            except NotImplementedError:
                pass
            except Exception as exc:
                _logger.error(
                    "conversation_delete_failed | thread_id=%s | err_type=%s",
                    thread_id,
                    type(exc).__name__,
                )
                raise ConversationRepositoryError("删除会话失败") from exc

        return ConversationDeleteStatus.UNSUPPORTED


def _message_view(message: Any) -> ConversationMessage:
    """Project messages without exposing LangGraph/LangChain object types."""
    if isinstance(message, dict):
        message_type = message.get("type", "unknown")
        content = message.get("content", "")
        message_id = message.get("id")
        has_tool_calls = bool(
            message.get("tool_calls") or message.get("invalid_tool_calls")
        )
    else:
        message_type = getattr(message, "type", "unknown")
        content = getattr(message, "content", "")
        message_id = getattr(message, "id", None)
        has_tool_calls = bool(
            getattr(message, "tool_calls", None)
            or getattr(message, "invalid_tool_calls", None)
        )
    return ConversationMessage(
        type=str(message_type),
        content=content if isinstance(content, str) else str(content),
        id=str(message_id) if message_id is not None else None,
        has_tool_calls=has_tool_calls,
    )


def _config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}
