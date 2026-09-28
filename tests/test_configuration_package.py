"""Application configuration assets stay colocated with their loader."""

from importlib.resources import files

from agent_tools.config import AgentToolsConfig
from api.settings import APISettings
from react_agent.configuration.settings import Settings
from react_agent.metering.pricing import pricing_path


def test_configuration_assets_are_available_in_package(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_PRICING_PATH", raising=False)
    monkeypatch.delenv("KNOWLEDGE_SERVICE_TIMEOUT", raising=False)

    package = files("react_agent.configuration")
    assert package.joinpath("config.yaml").is_file()
    assert package.joinpath("deepseek_pricing.yaml").is_file()
    assert pricing_path() == package.joinpath("deepseek_pricing.yaml")
    configured = Settings()
    assert isinstance(configured.tools, AgentToolsConfig)
    assert configured.tools.rag.client_timeout == 150


def test_agent_settings_ignore_knowledge_runtime_tuning_environment(
    monkeypatch,
) -> None:
    """服务端调优变量不得进入 Agent 配置模型或阻断 Agent 启动。"""
    monkeypatch.delenv("KNOWLEDGE_SERVICE_TIMEOUT", raising=False)
    monkeypatch.setenv("RAG_CPU_RERANK_CANDIDATES", "0")
    monkeypatch.setenv("RAG_RERANKER_CONCURRENCY", "not-an-integer")
    monkeypatch.setenv("RAG_RERANKER_BATCH_SIZE", "999")

    configured = Settings()

    assert configured.tools.rag.max_retries == 2
    assert configured.tools.rag.client_timeout == 150
    assert not hasattr(configured.tools.rag, "cpu_rerank_candidates")


def test_api_settings_own_api_redis_configuration(monkeypatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://api-redis:6380/2")
    monkeypatch.setenv("REDIS_MAX_CONNECTIONS", "7")

    api_settings = APISettings()
    agent_settings = Settings()

    assert api_settings.redis_url == "redis://api-redis:6380/2"
    assert api_settings.redis_max_connections == 7
    assert not hasattr(agent_settings, "redis")
