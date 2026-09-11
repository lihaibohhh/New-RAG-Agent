"""LangGraph Checkpointer adapter for conversation-management use cases."""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver

from react_agent.conversations.contracts import (
    ConversationDeleteStatus,
    ConversationRepositoryError,
)

_logger = logging.getLogger(__name__)


class LangGraphConversationRepository:
    """Expose conversation operations without leaking Checkpointer details upstream."""

    def __init__(self, checkpointer: Optional[BaseCheckpointSaver]):
        self._checkpointer = checkpointer

    async def get_messages(self, thread_id: str) -> Optional[List[Any]]:
        if self._checkpointer is None:
            return None

        try:
            checkpoint = await self._checkpointer.aget_tuple(_config(thread_id))
        except Exception as exc:
            _logger.error("conversation_history_failed | thread_id=%s | err=%s", thread_id, exc)
            raise ConversationRepositoryError("读取会话历史失败") from exc

        if checkpoint is None:
            return None
        values = checkpoint.checkpoint.get("channel_values", {})
        return list(values.get("messages", []))

    async def delete(self, thread_id: str) -> ConversationDeleteStatus:
        if self._checkpointer is None:
            return ConversationDeleteStatus.NOT_FOUND

        try:
            checkpoint = await self._checkpointer.aget_tuple(_config(thread_id))
        except Exception as exc:
            _logger.error("conversation_lookup_failed | thread_id=%s | err=%s", thread_id, exc)
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
                _logger.error("conversation_delete_failed | thread_id=%s | err=%s", thread_id, exc)
                raise ConversationRepositoryError("删除会话失败") from exc

        # Compatibility for older AsyncSqliteSaver versions without adelete_thread().
        connection = getattr(self._checkpointer, "conn", None)
        if connection is not None:
            try:
                for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                    await connection.execute(
                        f"DELETE FROM {table} WHERE thread_id = ?",  # noqa: S608
                        (thread_id,),
                    )
                await connection.commit()
                return ConversationDeleteStatus.DELETED
            except Exception as exc:
                _logger.error("conversation_sqlite_delete_failed | thread_id=%s | err=%s", thread_id, exc)
                raise ConversationRepositoryError("删除 SQLite 会话失败") from exc

        # Compatibility for older MemorySaver versions without adelete_thread().
        storage = getattr(self._checkpointer, "storage", None)
        if storage is not None:
            try:
                keys = [key for key in list(storage.keys()) if key[0] == thread_id]
                for key in keys:
                    del storage[key]
                return ConversationDeleteStatus.DELETED
            except Exception as exc:
                _logger.error("conversation_memory_delete_failed | thread_id=%s | err=%s", thread_id, exc)
                raise ConversationRepositoryError("删除内存会话失败") from exc

        return ConversationDeleteStatus.UNSUPPORTED


def _config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}

