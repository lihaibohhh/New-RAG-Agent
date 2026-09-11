from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from react_agent.agent.context import AgentContext
from react_agent.agent.dependencies import AgentDependencies
from react_agent.agent.service import AgentService
from react_agent.conversations.contracts import ConversationPersistenceConfig
from react_agent.conversations.infrastructure.checkpointer_factory import (
    CheckpointerFactory,
)
from react_agent.mcp_server.rag_tools import execute_query_financial_reports
from react_agent.rag.contracts import (
    RetrievedChunk,
    RetrievalResult,
    WarmupStatus,
)
from react_agent.rag.operations import RagWarmupManager
from react_agent.rag.runtime import create_rag_runtime
from react_agent.runtime import container
from react_agent.runtime.container import (
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

    async def ainvoke(self, inputs, *, context, config):
        self.context = context
        return {"messages": list(inputs["messages"]), "config": config}


@pytest.mark.asyncio
async def test_agent_service_passes_injected_dependencies_to_graph() -> None:
    dependencies = AgentDependencies(
        context=AgentContext(),
        model_provider=lambda: object(),
        tools=(),
    )
    graph = _FakeGraph()
    service = AgentService(dependencies, graph)

    await service.invoke([], thread_id="user:test")

    assert graph.context is dependencies


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
    monkeypatch.setattr(
        container,
        "get_rag_runtime_profile",
        lambda: SimpleNamespace(timeout=2),
    )
    monkeypatch.setattr(
        container,
        "load_chat_model",
        lambda model_ref: ("model", model_ref),
    )

    dependencies = container._compose_agent_dependencies(
        AgentContext(model="provider/model"),
        create_rag_runtime(),
    )

    assert dependencies.get_model() == ("model", "provider/model")
    assert [tool.name for tool in dependencies.tools] == [
        "search",
        "make_excel_table",
        "query_internal_knowledge",
        "docx_tool",
        "md_tool",
    ]


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
async def test_application_services_report_effective_backend(monkeypatch) -> None:
    dependencies = AgentDependencies(
        context=AgentContext(),
        model_provider=lambda: object(),
        tools=(),
    )
    monkeypatch.setattr(
        container,
        "_compose_agent_dependencies",
        lambda _context, _rag_runtime: dependencies,
    )

    services = await create_application_services(
        agent_context=dependencies.context,
        conversation_config=ConversationPersistenceConfig(
            checkpoint_backend="memory"
        ),
    )

    assert get_application_status(services).checkpoint_backend == "memory"
    await close_application_services(services)


def test_agent_and_rag_adapter_have_no_runtime_service_locator_imports() -> None:
    package_root = Path(container.__file__).parent.parent
    agent_nodes = (package_root / "agent" / "nodes.py").read_text(
        encoding="utf-8"
    )
    rag_adapter = (package_root / "tools" / "rag.py").read_text(
        encoding="utf-8"
    )
    mcp_adapter = (package_root / "mcp_server" / "rag_tools.py").read_text(
        encoding="utf-8"
    )

    assert "react_agent.utils.llm" not in agent_nodes
    assert "react_agent.tools" not in agent_nodes
    assert "react_agent.rag.runtime" not in rag_adapter
    assert "react_agent.rag.runtime" not in mcp_adapter
