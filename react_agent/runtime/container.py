"""Application composition root for process-level services and adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any

from react_agent.agent.configuration.context import AgentContext
from react_agent.agent.contracts.dependencies import AgentDependencies
from react_agent.agent.workflow.graph import compile_agent_graph
from react_agent.agent.application.service import AgentService
from react_agent.conversations.configuration import normalize_checkpoint_backend
from react_agent.conversations.contracts import ConversationPersistenceConfig
from react_agent.conversations.infrastructure.checkpointer_factory import (
    CheckpointerFactory,
)
from react_agent.conversations.infrastructure.langgraph_repository import (
    LangGraphConversationRepository,
)
from react_agent.conversations.service import ConversationService
from react_agent.configuration.settings import settings
from react_agent.rag.runtime import (
    create_configured_rag_runtime,
)
from react_agent.rag.runtime_ports import AgentRagRuntimePort
from react_agent.tools.excel import create_excel_tool
from react_agent.tools.make_docx import create_docx_tool
from react_agent.tools.markdown import create_markdown_tool
from react_agent.tools.rag import create_rag_tool
from react_agent.tools.search import create_search_tool
from react_agent.models import load_chat_model
from react_agent.metering.pricing import estimate_configured_model_cost


@dataclass(frozen=True)
class ApplicationServices:
    """Fully composed application services sharing one Checkpointer instance."""

    agent: AgentService
    conversations: ConversationService
    conversation_persistence: ConversationPersistenceConfig
    effective_checkpoint_backend: str
    _checkpointer_factory: CheckpointerFactory = field(repr=False, compare=False)
    _rag_runtime: AgentRagRuntimePort = field(repr=False, compare=False)


@dataclass(frozen=True)
class ApplicationStatus:
    """Read-only process status exposed to health-check adapters."""

    agent_initialized: bool
    requested_checkpoint_backend: str
    checkpoint_backend: str

    @property
    def ready(self) -> bool:
        """Only report ready when the selected persistence backend is active."""
        return self.agent_initialized and (
            self.requested_checkpoint_backend == self.checkpoint_backend
        )


async def create_application_services(
    *,
    agent_context: AgentContext,
    conversation_config: ConversationPersistenceConfig,
) -> ApplicationServices:
    """Create, configure and connect the complete application object graph."""
    checkpointer_factory = CheckpointerFactory()
    checkpointer = await checkpointer_factory.create(conversation_config)
    rag_runtime = create_configured_rag_runtime()
    dependencies = _compose_agent_dependencies(agent_context, rag_runtime)
    graph = compile_agent_graph(checkpointer)
    return ApplicationServices(
        agent=AgentService(dependencies, graph),
        conversations=ConversationService(
            LangGraphConversationRepository(checkpointer)
        ),
        conversation_persistence=conversation_config,
        effective_checkpoint_backend=checkpointer_factory.effective_backend(
            checkpointer
        ),
        _checkpointer_factory=checkpointer_factory,
        _rag_runtime=rag_runtime,
    )


def _compose_agent_dependencies(
    agent_context: AgentContext,
    rag_runtime: AgentRagRuntimePort,
) -> AgentDependencies:
    """Select the model adapter and the exact tool set injected into Agent."""
    rag_tool = create_rag_tool(
        retrieval_service_provider=rag_runtime.get_retrieval_service,
        max_retries=settings.tools.rag.max_retries,
        timeout=settings.tools.rag.client_timeout,
    )
    search_tool = create_search_tool(
        client_provider=_create_tavily_client,
        max_results=settings.tools.search.max_search_results,
        api_key_configured=bool(settings.secrets.TAVILY_API_KEY.strip()),
        max_retries=settings.tools.search.max_retries,
        timeout=settings.tools.search.timeout,
    )
    output_dir = settings.storage.OUTPUT_DIR
    excel_tool = create_excel_tool(
        output_dir=output_dir,
        default_mode=settings.tools.excel.mode,
        keep_backup=settings.tools.excel.keep_backup,
    )
    docx_tool = create_docx_tool(output_dir=output_dir)
    markdown_tool = create_markdown_tool(output_dir=output_dir)
    tools: tuple[Any, ...] = (
        search_tool,
        excel_tool,
        rag_tool,
        docx_tool,
        markdown_tool,
    )
    return AgentDependencies(
        config=agent_context,
        model_provider=partial(load_chat_model, settings.llm.model),
        tools=tools,
        model_ref=settings.llm.model,
        cost_estimator=estimate_configured_model_cost,
    )


def _create_tavily_client(max_results: int) -> Any:
    """Lazily instantiate the concrete search adapter selected by Runtime."""
    from langchain_tavily import TavilySearch

    return TavilySearch(max_results=max_results)


def get_application_status(services: ApplicationServices) -> ApplicationStatus:
    """Project the object graph to the minimal health-check contract."""
    return ApplicationStatus(
        agent_initialized=services.agent.initialized,
        requested_checkpoint_backend=normalize_checkpoint_backend(
            services.conversation_persistence.checkpoint_backend
        ),
        checkpoint_backend=services.effective_checkpoint_backend,
    )


async def close_application_services(services: ApplicationServices) -> None:
    """Release only the resources owned by this application object graph."""
    try:
        await services._rag_runtime.close()
    finally:
        await services._checkpointer_factory.close()


async def warmup_application_services(
    services: ApplicationServices,
    *,
    wait_seconds: int | float = 120,
) -> dict[str, Any]:
    """Trigger the RAG operational use case without exposing its runtime container."""
    return await services._rag_runtime.operations.ensure_ready(wait_seconds)
