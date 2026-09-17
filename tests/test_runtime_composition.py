from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from react_agent.agent.configuration.context import AgentContext
from react_agent.agent.contracts.dependencies import AgentDependencies
from react_agent.agent.application.service import AgentService
from react_agent.conversations import load_conversation_persistence_config
from react_agent.conversations.contracts import (
    ConversationPersistenceConfig,
    ConversationPersistenceInitializationError,
)
from react_agent.conversations.infrastructure import (
    checkpointer_factory as checkpointer_factory_module,
)
from react_agent.conversations.infrastructure.checkpointer_factory import (
    CheckpointerFactory,
)
from react_agent.configuration.settings import LLMConfig, SearchConfig, Settings
from react_agent.mcp_server.rag_tools import execute_query_financial_reports
from react_agent.rag.contracts import (
    RetrievedChunk,
    RetrievalResult,
    WarmupStatus,
)
from react_agent.rag.operations import RagWarmupManager
from react_agent.rag.runtime import create_rag_runtime
from react_agent.rag.runtime_ports import AgentRagRuntimePort
from react_agent.runtime import container
from react_agent.runtime.container import (
    ApplicationStatus,
    close_application_services,
    create_application_services,
    get_application_status,
)
from react_agent.tools.rag import create_rag_tool
from react_agent.tools.markdown import create_markdown_tool
from react_agent.tools.search import create_search_tool


class _FakeGraph:
    def __init__(self) -> None:
        self.context = None
        self.config = None
        self.version = None

    async def ainvoke(self, inputs, *, context, config):
        self.context = context
        return {"messages": list(inputs["messages"]), "config": config}

    async def astream_events(self, inputs, *, context, config, version):
        self.context = context
        self.config = config
        self.version = version
        yield {"event": "on_chat_model_stream", "data": {"inputs": inputs}}


@pytest.mark.asyncio
async def test_agent_service_passes_injected_dependencies_to_graph() -> None:
    dependencies = AgentDependencies(
        config=AgentContext(),
        model_provider=lambda: object(),
        tools=(),
    )
    graph = _FakeGraph()
    service = AgentService(dependencies, graph)

    await service.invoke([], thread_id="user:test")

    assert graph.context is dependencies


@pytest.mark.asyncio
async def test_agent_service_stream_events_preserves_v1_streaming_contract() -> None:
    dependencies = AgentDependencies(
        config=AgentContext(recursion_limit=40),
        model_provider=lambda: object(),
        tools=(),
    )
    graph = _FakeGraph()
    service = AgentService(dependencies, graph)

    events = [
        event async for event in service.stream_events([], thread_id="user:stream")
    ]

    assert events[0]["event"] == "on_chat_model_stream"
    assert graph.context is dependencies
    assert graph.config == {
        "recursion_limit": 40,
        "configurable": {"thread_id": "user:stream"},
    }
    assert graph.version == "v2"


@pytest.mark.asyncio
async def test_rag_tool_uses_only_injected_retrieval_service() -> None:
    calls: list[tuple[str, int]] = []

    class FakeRetrievalService:
        async def search(self, query: str, *, top_k: int) -> RetrievalResult:
            calls.append((query, top_k))
            return RetrievalResult(
                query=query,
                chunks=(
                    RetrievedChunk(
                        content="正文",
                        source_file="report.pdf",
                        source_page=7,
                        chunk_id="chunk-7",
                    ),
                ),
                stage="fake",
            )

    service = FakeRetrievalService()
    rag_tool = create_rag_tool(
        retrieval_service_provider=lambda: service,
        max_retries=0,
        timeout=1,
    )

    payload = json.loads(await rag_tool.ainvoke({"query": "  测试查询  "}))

    assert calls == [("测试查询", 3)]
    assert payload["ok"] is True
    assert payload["data"]["results"][0]["chunk_id"] == "chunk-7"
    assert payload["data"]["results"][0]["page"] == 7


@pytest.mark.asyncio
async def test_search_tool_uses_runtime_selected_client() -> None:
    requested_sizes: list[int] = []

    class FakeSearchClient:
        async def ainvoke(self, _query):
            return {"results": [{"title": "result", "url": "https://example.test"}]}

    search_tool = create_search_tool(
        client_provider=lambda size: (
            requested_sizes.append(size) or FakeSearchClient()
        ),
        max_results=7,
        api_key_configured=True,
        max_retries=0,
        timeout=1,
    )

    payload = json.loads(await search_tool.ainvoke({"query": "test"}))

    assert requested_sizes == [7]
    assert payload["ok"] is True
    assert payload["meta"]["max_results"] == 7


def test_artifact_tool_uses_runtime_selected_output_dir(tmp_path) -> None:
    markdown_tool = create_markdown_tool(output_dir=tmp_path)

    payload = markdown_tool.invoke(
        {
            "title": "runtime artifact",
            "sections": [{"heading": "section", "content": "body"}],
            "filename": "runtime-artifact",
            "mode": "overwrite",
        }
    )

    output_path = Path(payload["data"]["path"])
    assert output_path.parent == tmp_path
    assert output_path.read_text(encoding="utf-8").startswith("# runtime artifact")


@pytest.mark.asyncio
async def test_mcp_query_uses_explicit_service_provider() -> None:
    service_calls = 0

    class FakeRetrievalService:
        async def search(self, query: str, *, top_k: int) -> RetrievalResult:
            return RetrievalResult(query=query, stage=f"top-{top_k}")

    service = FakeRetrievalService()

    def provide_service():
        nonlocal service_calls
        service_calls += 1
        return service

    result = await execute_query_financial_reports(
        service_provider=provide_service,
        warmup=lambda _wait: _ready(),
        query="evidence",
        top_k=2,
    )

    assert service_calls == 1
    assert result["ok"] is True
    assert result["meta"]["stage"] == "top-2"


async def _ready() -> dict:
    return {
        "ready": True,
        "stage": "ready",
        "waited_seconds": 0,
        "warmup_status": {"state": "done"},
    }


@pytest.mark.asyncio
async def test_warmup_state_is_instance_scoped() -> None:
    class FakeRetrievalService:
        async def warmup(self) -> WarmupStatus:
            return WarmupStatus(ready=True, timings={"fake": 0.0})

    service = FakeRetrievalService()
    first = RagWarmupManager(lambda: service)
    second = RagWarmupManager(lambda: service)

    assert (await first.ensure_ready(1))["ready"] is True
    assert second.get_status()["warmup_status"]["state"] == "not_started"
    await first.close()
    await second.close()


@pytest.mark.asyncio
async def test_admin_health_composition_does_not_build_query_pipeline() -> None:
    rag_runtime = create_rag_runtime()

    rag_runtime.get_admin_service()

    assert rag_runtime._retrieval_service is None
    assert rag_runtime._embedding_provider is None
    await rag_runtime.close()


def test_runtime_selects_model_and_tool_catalog(monkeypatch) -> None:
    monkeypatch.setattr(container.settings.llm, "model", "provider/model")
    monkeypatch.setattr(
        container,
        "load_chat_model",
        lambda model_ref: ("model", model_ref),
    )

    dependencies = container._compose_agent_dependencies(
        AgentContext(),
        create_rag_runtime(),
    )

    assert dependencies.resolve_model() == ("model", "provider/model")
    assert dependencies.model_ref == "provider/model"
    assert [tool.name for tool in dependencies.tools] == [
        "search",
        "make_excel_table",
        "query_internal_knowledge",
        "docx_tool",
        "md_tool",
    ]


def test_model_and_search_adapter_settings_own_their_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MODEL", "provider/env-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.4")
    monkeypatch.setenv("MAX_SEARCH_RESULTS", "7")

    model_settings = LLMConfig()
    search_settings = SearchConfig()

    assert model_settings.model == "provider/env-model"
    assert model_settings.llm_temperature == 0.4
    assert search_settings.max_search_results == 7


def test_global_settings_do_not_duplicate_agent_behavior_context() -> None:
    global_settings = Settings()

    assert not hasattr(global_settings, "runtime")
    assert not hasattr(global_settings, "core")


def test_conversation_persistence_configuration_has_one_environment_loader() -> None:
    config = load_conversation_persistence_config(
        {
            "CHECKPOINT_BACKEND": "POSTGRES",
            "CHECKPOINT_DB_PATH": "./custom/checkpoints.sqlite3",
        }
    )
    defaults = load_conversation_persistence_config({})

    assert config == ConversationPersistenceConfig(
        checkpoint_backend="postgres",
        checkpoint_db_path="./custom/checkpoints.sqlite3",
    )
    assert defaults == ConversationPersistenceConfig(
        checkpoint_backend="sqlite",
        checkpoint_db_path="./data/agent-state/agent_checkpoints.sqlite3",
    )


@pytest.mark.parametrize(
    ("requested", "effective", "ready"),
    [
        ("sqlite", "sqlite", True),
        ("sqlite", "memory", False),
        ("postgres", "postgres", True),
        ("postgres", "memory", False),
        ("memory", "memory", True),
        ("none", "none", True),
    ],
)
def test_readiness_requires_the_selected_checkpoint_backend(
    requested: str,
    effective: str,
    ready: bool,
) -> None:
    status = ApplicationStatus(
        agent_initialized=True,
        requested_checkpoint_backend=requested,
        checkpoint_backend=effective,
    )

    assert status.ready is ready


def test_runtime_status_normalizes_checkpoint_backend_aliases() -> None:
    services = SimpleNamespace(
        agent=SimpleNamespace(initialized=True),
        conversation_persistence=ConversationPersistenceConfig(
            checkpoint_backend="postgresql"
        ),
        effective_checkpoint_backend="postgres",
    )

    status = get_application_status(services)

    assert status.requested_checkpoint_backend == "postgres"
    assert status.ready is True


def test_application_entrypoints_use_the_shared_persistence_loader() -> None:
    project_root = Path(__file__).parent.parent
    api_source = (project_root / "api" / "dependencies.py").read_text(encoding="utf-8")
    streamlit_source = (project_root / "tests" / "test_agent.py").read_text(
        encoding="utf-8"
    )

    for source in (api_source, streamlit_source):
        assert "load_conversation_persistence_config()" in source
        assert 'checkpoint_backend="sqlite"' not in source
        assert 'checkpoint_backend="postgres"' not in source


def test_agent_dependencies_manage_an_immutable_active_tool_catalog() -> None:
    dependencies = AgentDependencies(
        config=AgentContext(enable_web_search=False),
        model_provider=lambda: object(),
        tools=[
            SimpleNamespace(name="search"),
            SimpleNamespace(name="query_internal_knowledge"),
        ],
    )

    active, active_names = dependencies.active_tools()

    assert isinstance(dependencies.tools, tuple)
    assert [tool.name for tool in active] == ["query_internal_knowledge"]
    assert active_names == frozenset({"query_internal_knowledge"})


def test_agent_dependencies_reject_duplicate_or_unnamed_tools() -> None:
    with pytest.raises(ValueError, match="不能重复"):
        AgentDependencies(
            config=AgentContext(),
            model_provider=lambda: object(),
            tools=[SimpleNamespace(name="same"), SimpleNamespace(name="same")],
        )

    with pytest.raises(ValueError, match="非空 name"):
        AgentDependencies(
            config=AgentContext(),
            model_provider=lambda: object(),
            tools=[SimpleNamespace(name="")],
        )


@pytest.mark.asyncio
async def test_checkpointer_lifecycle_is_instance_scoped() -> None:
    first = CheckpointerFactory()
    second = CheckpointerFactory()
    config = ConversationPersistenceConfig(checkpoint_backend="memory")

    first_checkpointer = await first.create(config)
    second_checkpointer = await second.create(config)

    assert first_checkpointer is not second_checkpointer
    assert first.effective_backend(first_checkpointer) == "memory"
    assert second.effective_backend(second_checkpointer) == "memory"

    await first.close()

    assert first.effective_backend(first_checkpointer) == "unknown"
    assert second.effective_backend(second_checkpointer) == "memory"
    await second.close()


@pytest.mark.asyncio
async def test_postgres_missing_dependencies_never_falls_back_to_memory(
    monkeypatch,
) -> None:
    monkeypatch.setattr(checkpointer_factory_module, "AsyncPostgresSaver", None)
    factory = CheckpointerFactory()
    monkeypatch.setattr(
        factory,
        "_postgres_conn_str",
        lambda: "postgresql://configured-but-not-used",
    )

    with pytest.raises(
        ConversationPersistenceInitializationError,
        match="已禁止降级到 MemorySaver",
    ):
        await factory.create(
            ConversationPersistenceConfig(checkpoint_backend="postgres")
        )

    assert factory._instances == {}


@pytest.mark.asyncio
async def test_postgres_missing_connection_string_stops_startup(
    monkeypatch,
) -> None:
    factory = CheckpointerFactory()
    monkeypatch.setattr(factory, "_postgres_conn_str", lambda: "")

    with pytest.raises(
        ConversationPersistenceInitializationError,
        match="POSTGRES_DB_URL 未配置",
    ):
        await factory.create(
            ConversationPersistenceConfig(checkpoint_backend="postgres")
        )

    assert factory._instances == {}


@pytest.mark.asyncio
async def test_application_services_report_effective_backend(monkeypatch) -> None:
    monkeypatch.setenv("RAG_RUNTIME_MODE", "local")
    monkeypatch.delenv("KNOWLEDGE_SERVICE_URL", raising=False)
    dependencies = AgentDependencies(
        config=AgentContext(),
        model_provider=lambda: object(),
        tools=(),
    )
    monkeypatch.setattr(
        container,
        "_compose_agent_dependencies",
        lambda _context, _rag_runtime: dependencies,
    )

    services = await create_application_services(
        agent_context=dependencies.config,
        conversation_config=ConversationPersistenceConfig(checkpoint_backend="memory"),
    )

    assert get_application_status(services).checkpoint_backend == "memory"
    await close_application_services(services)


def test_agent_and_rag_adapter_have_no_runtime_service_locator_imports() -> None:
    package_root = Path(container.__file__).parent.parent
    agent_nodes = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (package_root / "agent" / "workflow" / "nodes").glob("*.py")
    )
    rag_adapter = (package_root / "tools" / "rag.py").read_text(encoding="utf-8")
    mcp_adapter = (package_root / "mcp_server" / "rag_tools.py").read_text(
        encoding="utf-8"
    )
    application_runtime = (package_root / "runtime" / "container.py").read_text(
        encoding="utf-8"
    )

    assert "react_agent.utils.llm" not in agent_nodes
    assert "react_agent.tools" not in agent_nodes
    assert "from react_agent.rag.runtime import" not in rag_adapter
    assert "from react_agent.rag.runtime import" not in mcp_adapter
    assert "get_rag_runtime_profile" not in application_runtime


def test_agent_rag_runtime_port_exposes_only_query_lifecycle_capabilities() -> None:
    public_members = AgentRagRuntimePort.__dict__

    assert "operations" in public_members["__annotations__"]
    assert "get_retrieval_service" in public_members
    assert "close" in public_members
    assert "get_admin_service" not in public_members
    assert "get_ingestion_service" not in public_members
    assert "get_evaluation_retrieval_service" not in public_members


def test_legacy_agent_stream_and_chat_router_are_removed() -> None:
    package_root = Path(container.__file__).parent.parent
    project_root = package_root.parent
    api_main = (project_root / "api" / "main.py").read_text(encoding="utf-8")

    assert "stream" not in AgentService.__dict__
    assert "api.routes.chat" not in api_main
    assert "include_router(chat_router)" not in api_main
    assert not (project_root / "api" / "routes" / "chat.py").exists()
