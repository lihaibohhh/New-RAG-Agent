from __future__ import annotations

from pathlib import Path

import pytest

from knowledge.contracts import StoredChunk
from knowledge.ingestion import IngestionConfig, IngestionService
from knowledge.runtime import KnowledgeRuntimeConfig
from knowledge.runtime import create_knowledge_runtime


def test_legacy_agent_rag_package_is_removed() -> None:
    package_root = Path(__file__).parent.parent

    assert not (package_root / "react_agent" / "rag").exists()


def test_knowledge_root_contains_only_public_source_contracts() -> None:
    package_root = Path(__file__).parent.parent / "knowledge"
    root_modules = {path.name for path in package_root.glob("*.py")}

    assert root_modules == {"__init__.py", "contracts.py", "runtime_ports.py"}
    assert (package_root / "ingestion" / "source.py").is_file()


def test_rag_owns_internal_contracts_ports_and_adapters() -> None:
    package_root = Path(__file__).parent.parent / "knowledge"
    public_contracts = (package_root / "contracts.py").read_text(encoding="utf-8")
    rag_ports = (package_root / "rag" / "ports.py").read_text(encoding="utf-8")

    assert "class RetrievalRequest" not in public_contracts
    assert "class CandidateRetrievalTrace" not in public_contracts
    assert "class SemanticCachePort" in rag_ports
    assert "class HybridRetrieverPort" in rag_ports
    assert "class RerankerPort" in rag_ports
    assert "class ChunkReaderPort" in rag_ports
    assert "class KnowledgeBaseInspectorPort" in rag_ports
    assert (package_root / "rag" / "contracts.py").is_file()
    assert (package_root / "rag" / "ports.py").is_file()
    assert (package_root / "rag" / "infrastructure").is_dir()
    assert (package_root / "rag" / "admin" / "service.py").is_file()
    assert (
        package_root / "rag" / "infrastructure" / "storage" / "chroma_knowledge_base.py"
    ).is_file()
    assert not list((package_root / "admin").rglob("*.py"))
    assert not (package_root / "foundation" / "ports.py").exists()
    assert not (package_root / "foundation" / "redis.py").exists()
    assert not (package_root / "foundation" / "chroma.py").exists()
    assert not (
        package_root / "foundation" / "storage" / "chroma_knowledge_base.py"
    ).exists()
    assert not (package_root / "runtime" / "offline.py").exists()


def test_ingestion_owns_internal_contracts_ports_and_adapters() -> None:
    package_root = Path(__file__).parent.parent / "knowledge"
    public_contracts = (package_root / "contracts.py").read_text(encoding="utf-8")

    assert "class ParseRequest" not in public_contracts
    assert "class ParsedChunk" not in public_contracts
    assert "class ParseResult" not in public_contracts
    assert "OcrPolicy" not in public_contracts
    assert "class KnowledgeDocument" in public_contracts
    assert "class RagDocument" not in public_contracts
    assert not (package_root / "ports.py").exists()
    assert not list((package_root / "infrastructure").rglob("*.py"))
    assert (package_root / "ingestion" / "contracts.py").is_file()
    assert (package_root / "ingestion" / "ports.py").is_file()
    assert (package_root / "ingestion" / "infrastructure").is_dir()


def test_top_runtime_only_connects_module_runtimes_and_shared_resources() -> None:
    package_root = Path(__file__).parent.parent / "knowledge"
    container_source = (package_root / "runtime" / "container.py").read_text(
        encoding="utf-8"
    )
    rag_runtime_source = (package_root / "rag" / "runtime.py").read_text(
        encoding="utf-8"
    )
    ingestion_runtime_source = (package_root / "ingestion" / "runtime.py").read_text(
        encoding="utf-8"
    )
    resources_source = (package_root / "runtime" / "resources.py").read_text(
        encoding="utf-8"
    )
    server_source = (package_root / "server" / "app.py").read_text(encoding="utf-8")

    assert "DoclingServiceAdapter" not in container_source
    assert "RedisSemanticCacheAdapter" not in container_source
    assert "KnowledgeRuntimeConfig" not in rag_runtime_source
    assert "KnowledgeRuntimeConfig" not in ingestion_runtime_source
    assert "class LocalRagRuntime" in rag_runtime_source
    assert "class LocalIngestionRuntime" in ingestion_runtime_source
    assert "class SharedKnowledgeResources" in resources_source
    assert "def get_retrieval_service" not in container_source
    assert "def get_evaluation_retrieval_service" not in container_source
    assert "def get_admin_service" not in container_source
    assert "def get_ingestion_service" not in container_source
    assert "def operations" not in container_source
    assert 'getattr(runtime, "rag_runtime", runtime)' not in server_source
    assert 'getattr(runtime, "ingestion_runtime", runtime)' not in server_source


def test_module_runtime_configs_expose_only_owned_settings() -> None:
    package_root = Path(__file__).parent.parent / "knowledge"
    config = KnowledgeRuntimeConfig()

    assert (package_root / "rag" / "config.py").is_file()
    assert (package_root / "ingestion" / "config.py").is_file()
    assert not hasattr(config.rag, "docling")
    assert not hasattr(config.rag, "service")
    assert not hasattr(config.ingestion, "redis")
    assert not hasattr(config.ingestion, "reranker")
    assert not hasattr(config.shared, "reranker_model")


def test_offline_chunk_reader_does_not_create_full_runtime(monkeypatch) -> None:
    from knowledge.rag import offline

    observed: dict[str, object] = {}

    class Reader:
        def __init__(self, *, chroma_dir: str) -> None:
            observed["chroma_dir"] = chroma_dir

        async def list_chunks(self, **kwargs):
            observed.update(kwargs)
            return [StoredChunk("chunk-1", "content")]

    monkeypatch.setattr(offline, "ChromaKnowledgeBaseAdapter", Reader)

    chunks = offline.read_chunks_sync(
        chroma_dir="test-chroma",
        source_file="report.pdf",
        offset=2,
        limit=3,
    )

    assert chunks == (StoredChunk("chunk-1", "content"),)
    assert observed == {
        "chroma_dir": "test-chroma",
        "source_file": "report.pdf",
        "offset": 2,
        "limit": 3,
    }
    offline_source = (Path(offline.__file__)).read_text(encoding="utf-8")
    assert "KnowledgeRuntime" not in offline_source
    assert "knowledge.runtime" not in offline_source
    assert "EmbeddingProviderAdapter" not in offline_source
    dataset_source = (
        Path(__file__).parent.parent / "eval" / "dataset" / "sources.py"
    ).read_text(encoding="utf-8")
    assert "knowledge.rag.offline" in dataset_source
    assert "knowledge.runtime" not in dataset_source


def test_http_transport_codecs_are_shared_by_client_and_server() -> None:
    package_root = Path(__file__).parent.parent / "knowledge"
    client_source = (package_root / "client" / "remote.py").read_text(encoding="utf-8")
    server_source = (package_root / "server" / "app.py").read_text(encoding="utf-8")

    assert "RetrievedChunk(" not in client_source
    assert "EvaluationCandidate(" not in client_source
    assert "StoredChunk(" not in client_source
    assert "asdict(" not in server_source
    assert "_retrieval_payload" not in server_source
    assert "_evaluation_retrieval_payload" not in server_source


def test_ingestion_config_rejects_invalid_resource_limits() -> None:
    for kwargs in (
        {"max_pdf_pages": 0},
        {"batch_size": 0},
        {"workers": 0},
    ):
        try:
            IngestionConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"配置应被拒绝: {kwargs}")


@pytest.mark.asyncio
async def test_ingestion_service_uses_injected_pdf_limit(tmp_path: Path) -> None:
    class Manifest:
        def load(self) -> dict[str, str]:
            return {}

        def save(self, _record: dict[str, str]) -> None:
            raise AssertionError("被排除的文件不应更新清单")

    class PageCounter:
        def count(self, _file_path: str) -> int:
            return 2

    class UnusedDependency:
        def __getattr__(self, name: str):
            raise AssertionError(f"被排除的文件不应调用 {name}")

    class CacheInvalidator:
        async def notify_index_changed(self) -> None:
            raise AssertionError("没有写入 chunk 时不应失效缓存")

    (tmp_path / "oversized.pdf").write_bytes(b"")
    service = IngestionService(
        writer=UnusedDependency(),
        manifest=Manifest(),
        page_counter=PageCounter(),
        preflight=UnusedDependency(),
        document_parser=UnusedDependency(),
        index_changed=CacheInvalidator(),
        config=IngestionConfig(max_pdf_pages=1),
    )

    report = await service.ingest(str(tmp_path))

    assert report.status == "no_eligible_files"
    assert report.files_skipped == 1
    assert report.chunks_written == 0
    assert report.cache_invalidated is False


@pytest.mark.asyncio
async def test_ingestion_publishes_index_change_after_committed_write() -> None:
    class IndexChanged:
        def __init__(self) -> None:
            self.calls = 0

        async def notify_index_changed(self) -> None:
            self.calls += 1

    observer = IndexChanged()
    service = IngestionService(
        writer=object(),
        manifest=object(),
        page_counter=object(),
        preflight=object(),
        document_parser=object(),
        index_changed=observer,
    )
    service._build_sync = lambda _path: {
        "status": "completed",
        "data_dir": "reports",
        "chunks_written": 2,
    }

    report = await service.ingest("reports")

    assert observer.calls == 1
    assert report.cache_invalidated is True


@pytest.mark.asyncio
async def test_local_runtime_requires_explicit_configuration() -> None:
    config = KnowledgeRuntimeConfig()
    runtime = create_knowledge_runtime(config)

    assert runtime._config is config
    await runtime.close()
