from __future__ import annotations

from pathlib import Path

import pytest

import check_tables
from knowledge.contracts import IngestionReport
from knowledge.client import (
    KnowledgeClientConfig,
    RemoteIngestionRuntime,
    RemoteRagRuntime,
    create_remote_ingestion_runtime,
    create_remote_rag_runtime,
    load_client_config,
)
from knowledge.server.runtime import create_knowledge_runtime_config


def test_knowledge_namespace_has_no_legacy_top_level_packages() -> None:
    project_root = Path(__file__).parent.parent

    assert not (project_root / "knowledge_client").exists()
    assert not (project_root / "knowledge_service").exists()


def test_server_runtime_owns_its_environment_projection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RAG_DEVICE", "cpu")
    monkeypatch.setenv("RAG_CPU_RERANK_CANDIDATES", "33")
    monkeypatch.setenv("REDIS_MAX_CONNECTIONS", "7")
    monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("DOCLING_ENABLED", "true")
    monkeypatch.setenv("RAG_INGESTION_BATCH_SIZE", "64")

    config = create_knowledge_runtime_config()

    assert config.shared.requested_device == "cpu"
    assert config.rag.tuning.cpu_rerank_candidates == 33
    assert config.rag.redis.max_connections == 7
    assert config.storage.chroma_db_path == str((tmp_path / "chroma").resolve())
    assert config.ingestion.docling.enabled is True
    assert config.ingestion.service.batch_size == 64


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RAG_MAX_CONTENT_CHARS", "199"),
        ("RAG_RERANK_TOP_N", "not-an-integer"),
        ("RAG_CPU_RERANK_CANDIDATES", "0"),
        ("RAG_RERANKER_CONCURRENCY", "9"),
        ("RAG_RERANKER_BATCH_SIZE", "129"),
        ("REDIS_MAX_CONNECTIONS", "0"),
        ("SEMANTIC_CACHE_TTL", "0"),
        ("SEMANTIC_CACHE_THRESHOLD", "1.1"),
        ("CACHE_MIN_SCORE", "-0.1"),
        ("DOCLING_REQUEST_TIMEOUT", "0"),
        ("DOCLING_SEGMENT_RETRIES", "-1"),
        ("DOCLING_AUTO_MIN_IMAGE_AREA_RATIO", "1.5"),
        ("DOCLING_LOCAL_MIN_PAGE_COVERAGE", "-0.1"),
        ("RAG_INGESTION_WORKERS", "17"),
        ("RAG_INGESTION_BATCH_SIZE", "20001"),
        ("RERANKER_THRESHOLD", "nan"),
    ],
)
def test_server_runtime_rejects_invalid_numeric_environment(
    monkeypatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError):
        create_knowledge_runtime_config()


@pytest.mark.parametrize(
    "name",
    [
        "DOCLING_ENABLED",
        "DOCLING_STRICT_MODE",
        "RAG_INGESTION_FAIL_FAST",
        "RERANKER_DEBUG",
    ],
)
def test_server_runtime_rejects_invalid_boolean_environment(
    monkeypatch,
    name: str,
) -> None:
    monkeypatch.setenv(name, "sometimes")

    with pytest.raises(ValueError):
        create_knowledge_runtime_config()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RAG_DEVICE", "tpu"),
        ("DOCLING_PARSER_MODE", "automatic"),
    ],
)
def test_server_runtime_rejects_invalid_enum_environment(
    monkeypatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError):
        create_knowledge_runtime_config()


def test_client_config_is_explicit_and_normalized() -> None:
    config = KnowledgeClientConfig(
        base_url=" https://knowledge.example/ ",
        api_key=" secret ",
        timeout=30.0,
    )

    assert config.base_url == "https://knowledge.example"
    assert config.api_key == "secret"
    assert config.timeout == 30.0


def test_load_client_config_from_supplied_environment() -> None:
    config = load_client_config(
        {
            "RAG_RUNTIME_MODE": "remote",
            "KNOWLEDGE_SERVICE_URL": "http://knowledge.test/",
            "KNOWLEDGE_SERVICE_API_KEY": "test-secret",
            "KNOWLEDGE_SERVICE_TIMEOUT": "12.5",
        }
    )

    assert config == KnowledgeClientConfig(
        base_url="http://knowledge.test",
        api_key="test-secret",
        timeout=12.5,
    )


@pytest.mark.asyncio
async def test_explicit_factory_creates_remote_only_runtime() -> None:
    runtime = create_remote_rag_runtime(
        KnowledgeClientConfig(base_url="http://knowledge.test")
    )

    assert isinstance(runtime, RemoteRagRuntime)
    assert not hasattr(runtime, "_embedding_provider")
    await runtime.close()


@pytest.mark.asyncio
async def test_ingestion_factory_exposes_only_ingestion_runtime() -> None:
    runtime = create_remote_ingestion_runtime(
        KnowledgeClientConfig(base_url="http://knowledge.test")
    )

    assert isinstance(runtime, RemoteIngestionRuntime)
    assert hasattr(runtime, "get_ingestion_service")
    assert not hasattr(runtime, "get_retrieval_service")
    assert not hasattr(runtime, "get_admin_service")
    await runtime.close()


@pytest.mark.parametrize(
    ("environ", "error_type", "message"),
    [
        ({"RAG_RUNTIME_MODE": "local"}, RuntimeError, "只允许 Knowledge Service"),
        ({"RAG_RUNTIME_MODE": "automatic"}, ValueError, "仅支持 remote 或 local"),
        (
            {"RAG_RUNTIME_MODE": "remote"},
            RuntimeError,
            "必须配置 KNOWLEDGE_SERVICE_URL",
        ),
    ],
)
def test_client_config_rejects_non_remote_or_incomplete_modes(
    environ: dict[str, str],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        load_client_config(environ)


def test_build_script_closes_remote_runtime(monkeypatch) -> None:
    class IngestionService:
        async def ingest(self, relative_path: str) -> IngestionReport:
            return IngestionReport(data_dir=relative_path, status="completed")

    class Runtime:
        def __init__(self) -> None:
            self.closed = False

        def get_ingestion_service(self) -> IngestionService:
            return IngestionService()

        async def close(self) -> None:
            self.closed = True

    runtime = Runtime()
    monkeypatch.setattr(
        check_tables,
        "create_configured_ingestion_runtime",
        lambda: runtime,
    )

    report = check_tables.build_vector_db("reports")

    assert report["data_dir"] == "reports"
    assert report["status"] == "completed"
    assert runtime.closed is True
