"""Conversation configuration, message views, and checkpoint adapter boundaries."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from api.routes.v1 import sessions
from react_agent.conversations.configuration import (
    load_conversation_persistence_config,
    normalize_checkpoint_backend,
)
from react_agent.conversations.contracts import (
    ConversationDeleteStatus,
    ConversationMessage,
    ConversationPersistenceConfig,
)
from react_agent.conversations.infrastructure.checkpointer_factory import (
    CheckpointerFactory,
)
from react_agent.conversations.infrastructure.langgraph_repository import (
    LangGraphConversationRepository,
)
from react_agent.conversations.service import ConversationService
from react_agent.conversations.projection import project_display_messages


@pytest.mark.parametrize("backend", ["bad", "mysql", "postgresql-extra"])
def test_unknown_checkpoint_backend_is_rejected(backend: str) -> None:
    with pytest.raises(ValueError, match="不支持的 CHECKPOINT_BACKEND"):
        normalize_checkpoint_backend(backend)
    with pytest.raises(ValueError, match="不支持的 CHECKPOINT_BACKEND"):
        load_conversation_persistence_config({"CHECKPOINT_BACKEND": backend})


@pytest.mark.asyncio
async def test_factory_rejects_invalid_explicit_configuration() -> None:
    factory = CheckpointerFactory()
    with pytest.raises(ValueError, match="不支持的 CHECKPOINT_BACKEND"):
        await factory.create(ConversationPersistenceConfig(checkpoint_backend="bad"))
    assert factory._instances == {}


def test_postgres_configuration_is_injected_and_secret_is_redacted() -> None:
    config = load_conversation_persistence_config(
        {
            "CHECKPOINT_BACKEND": "pg",
            "POSTGRES_DB_URL": "postgresql://test:fake-password@localhost/test",
            "POSTGRES_POOL_MIN_SIZE": "2",
            "POSTGRES_POOL_MAX_SIZE": "5",
            "POSTGRES_CONNECT_TIMEOUT_SECONDS": "3.5",
        }
    )
    assert config.checkpoint_backend == "postgres"
    assert config.postgres_pool_min_size == 2
    assert config.postgres_pool_max_size == 5
    assert config.postgres_connect_timeout_seconds == 3.5
    assert "fake-password" not in repr(config)

    factory = CheckpointerFactory()
    key = factory._compute_cache_key("postgres", config)
    another = ConversationPersistenceConfig(
        checkpoint_backend="postgres",
        postgres_db_url=config.postgres_db_url,
        postgres_pool_max_size=6,
    )
    assert key != factory._compute_cache_key("postgres", another)
    assert "fake-password" not in key


def test_sqlite_path_is_anchored_to_application_root() -> None:
    config = ConversationPersistenceConfig(
        checkpoint_db_path="./data/agent-state/test-checkpoints.sqlite3"
    )
    expected = (
        Path(__file__).resolve().parents[1]
        / "data/agent-state/test-checkpoints.sqlite3"
    ).resolve()
    assert CheckpointerFactory()._sqlite_db_path(config) == expected


class FakeSaver:
    def __init__(self, messages: list[object], *, supports_delete: bool = True) -> None:
        self.messages = messages
        self.supports_delete = supports_delete
        self.deleted: list[str] = []
        self.storage = {("user:bucket:session", "namespace"): "must-not-touch"}

    async def aget_tuple(self, config: dict) -> SimpleNamespace:
        assert config["configurable"]["thread_id"] == "user:bucket:session"
        return SimpleNamespace(checkpoint={"channel_values": {"messages": self.messages}})

    async def adelete_thread(self, thread_id: str) -> None:
        if not self.supports_delete:
            raise NotImplementedError
        self.deleted.append(thread_id)


@pytest.mark.asyncio
async def test_repository_returns_typed_message_views() -> None:
    saver = FakeSaver(
        [
            HumanMessage(content="问题", id="human-1"),
            AIMessage(content="回答", id="ai-1", tool_calls=[]),
            {"type": "tool", "content": {"ok": True}, "id": "tool-1"},
        ]
    )
    service = ConversationService(LangGraphConversationRepository(saver))

    messages = await service.get_history(" user:bucket:session ")

    assert messages == [
        ConversationMessage(type="human", content="问题", id="human-1"),
        ConversationMessage(type="ai", content="回答", id="ai-1"),
        ConversationMessage(type="tool", content="{'ok': True}", id="tool-1"),
    ]
    assert all(type(message) is ConversationMessage for message in messages)


def test_display_projection_excludes_tools_and_tool_call_plans() -> None:
    messages = [
        ConversationMessage(type="human", content="问题", id="h1"),
        ConversationMessage(
            type="ai",
            content="准备检索",
            id="a1",
            has_tool_calls=True,
        ),
        ConversationMessage(type="tool", content="结果", id="t1"),
        ConversationMessage(type="ai", content="最终回答", id="a2"),
    ]

    assert project_display_messages(messages) == [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "最终回答"},
    ]


@pytest.mark.asyncio
async def test_repository_marks_assistant_tool_calls_for_display_filtering() -> None:
    saver = FakeSaver(
        [
            AIMessage(
                content="准备检索",
                id="ai-tool",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "query_internal_knowledge",
                        "args": {"query": "测试"},
                    }
                ],
            )
        ]
    )

    messages = await LangGraphConversationRepository(saver).get_messages(
        "user:bucket:session"
    )

    assert messages is not None
    assert messages[0].has_tool_calls is True


@pytest.mark.asyncio
async def test_repository_deletes_only_through_public_checkpoint_api() -> None:
    saver = FakeSaver([])
    repository = LangGraphConversationRepository(saver)
    assert await repository.delete("user:bucket:session") is ConversationDeleteStatus.DELETED
    assert saver.deleted == ["user:bucket:session"]

    unsupported = FakeSaver([], supports_delete=False)
    repository = LangGraphConversationRepository(unsupported)
    assert await repository.delete("user:bucket:session") is ConversationDeleteStatus.UNSUPPORTED
    assert unsupported.storage == {
        ("user:bucket:session", "namespace"): "must-not-touch"
    }


@pytest.mark.asyncio
async def test_session_route_paginates_message_views(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sessions, "get_rate_limit_key", lambda _key, _request: "bucket")
    saver = FakeSaver(
        [HumanMessage(content="第一条"), AIMessage(content="第二条"), AIMessage(content="第三条")]
    )
    service = ConversationService(LangGraphConversationRepository(saver))

    response = await sessions.get_session_history(
        "session",
        None,
        page=1,
        page_size=2,
        conversations=service,
        api_key="fake-key",
    )

    assert response.total == 3
    assert response.has_more is True
    assert [(message.type, message.content) for message in response.messages] == [
        ("human", "第一条"),
        ("ai", "第二条"),
    ]
