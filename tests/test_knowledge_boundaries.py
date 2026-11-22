from __future__ import annotations

from pathlib import Path

import pytest

import knowledge.ingestion.service as ingestion_service_module
from knowledge.contracts import KnowledgeDocument, StoredChunk
from knowledge.ingestion import IngestionBatchSettings, IngestionPipeline
from knowledge.runtime import create_knowledge_services
from knowledge.settings import KnowledgeSettings


def test_legacy_agent_rag_package_is_removed() -> None:
    package_root = Path(__file__).parent.parent

    assert not (package_root / "react_agent" / "rag").exists()


def test_knowledge_root_contains_only_public_source_contracts() -> None:
    package_root = Path(__file__).parent.parent / "knowledge"
    root_modules = {path.name for path in package_root.glob("*.py")}

    assert root_modules == {
        "__init__.py",
        "contracts.py",
        "services.py",
        "settings.py",
    }
    assert (package_root / "ingestion" / "source.py").is_file()


def test_rag_owns_internal_contracts_and_adapters() -> None:
    package_root = Path(__file__).parent.parent / "knowledge"
    public_contracts = (package_root / "contracts.py").read_text(encoding="utf-8")
    rag_contracts = (package_root / "rag" / "contracts.py").read_text(
        encoding="utf-8"
    )

    assert "class RetrievalRequest" not in public_contracts
    assert "class CandidateRetrievalTrace" not in public_contracts
    assert "class SemanticCachePort" in rag_contracts
    assert "class HybridRetrieverPort" not in rag_contracts
    assert "class RerankerPort" in rag_contracts
    assert "class ChunkReaderPort" in rag_contracts
    assert "class KnowledgeBaseInspectorPort" not in rag_contracts
    assert (package_root / "rag" / "contracts.py").is_file()
    assert not (package_root / "rag" / "ports.py").exists()
    assert (package_root / "rag" / "infrastructure").is_dir()
    assert (package_root / "rag" / "application.py").is_file()
    assert not (package_root / "rag" / "query" / "service.py").exists()
    assert not (package_root / "rag" / "query" / "result_builder.py").exists()
    assert not (package_root / "rag" / "admin.py").exists()
    assert not (package_root / "rag" / "evaluation.py").exists()
    assert not (package_root / "rag" / "offline.py").exists()
    assert not (package_root / "rag" / "operations.py").exists()
    infrastructure = package_root / "rag" / "infrastructure"
    assert {path.name for path in infrastructure.glob("*.py")} >= {
        "bm25.py",
        "cache_codec.py",
        "chroma.py",
        "corpus.py",
        "reranker.py",
        "semantic_cache.py",
        "vector.py",
    }
    assert not list((package_root / "rag" / "retrieval").glob("*.py"))
    assert not list((infrastructure / "cache").glob("*.py"))
    assert not list((infrastructure / "retrieval").glob("*.py"))
    assert not list((infrastructure / "storage").glob("*.py"))
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
    assert "class LocalRagService" in rag_runtime_source
    assert "class LocalIngestionService(IngestionPipeline)" in ingestion_runtime_source
    assert "_get_ingestion_pipeline" not in ingestion_runtime_source
    assert "_ingestion_service" not in ingestion_runtime_source
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
    config = KnowledgeSettings()

    assert (package_root / "settings.py").is_file()
    assert not (package_root / "rag" / "config.py").exists()
    assert not (package_root / "ingestion" / "config.py").exists()
    assert not hasattr(config.rag, "docling")
    assert not hasattr(config.rag, "service")
    assert not hasattr(config.ingestion, "redis")
    assert not hasattr(config.ingestion, "reranker")
    assert not hasattr(config.shared, "reranker_model")


def test_offline_chunk_reader_does_not_create_full_runtime(monkeypatch) -> None:
    from knowledge.rag import runtime as rag_runtime

    observed: dict[str, object] = {}

    class Reader:
        def __init__(self, *, chroma_dir: str) -> None:
            observed["chroma_dir"] = chroma_dir

        async def list_chunks(self, **kwargs):
            observed.update(kwargs)
            return [StoredChunk("chunk-1", "content")]

    monkeypatch.setattr(rag_runtime, "ChromaKnowledgeBaseAdapter", Reader)

    chunks = rag_runtime.read_chunks_sync(
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
    runtime_source = (Path(rag_runtime.__file__)).read_text(encoding="utf-8")
    assert "KnowledgeRuntime" not in runtime_source
    assert "knowledge.runtime" not in runtime_source
    assert "EmbeddingProviderAdapter" not in runtime_source
    dataset_source = (
        Path(__file__).parent.parent / "eval" / "dataset" / "sources.py"
    ).read_text(encoding="utf-8")
    assert "knowledge.rag.runtime" in dataset_source
    assert "knowledge.runtime" not in dataset_source


def test_http_transport_codecs_are_shared_by_client_and_server() -> None:
    package_root = Path(__file__).parent.parent / "knowledge"
    client_source = (package_root / "client" / "http.py").read_text(encoding="utf-8")
    server_source = (package_root / "server" / "app.py").read_text(encoding="utf-8")

    assert "RetrievedChunk(" not in client_source
    assert "EvaluationCandidate(" not in client_source
    assert "StoredChunk(" not in client_source
    assert "asdict(" not in server_source
    assert "_retrieval_payload" not in server_source
    assert "_evaluation_retrieval_payload" not in server_source


def test_rag_search_chain_has_no_forwarding_service_layers() -> None:
    package_root = Path(__file__).parent.parent / "knowledge"
    client_source = (package_root / "client" / "http.py").read_text(encoding="utf-8")
    runtime_source = (package_root / "rag" / "runtime.py").read_text(
        encoding="utf-8"
    )

    assert "class HttpRagService(_KnowledgeHttpTransport)" in client_source
    assert "self._client = KnowledgeServiceClient" not in client_source
    assert "class LocalRagService(RetrievalApplicationService)" in runtime_source
    assert "self._retrieval_service" not in runtime_source
    assert "_get_retrieval_service(" not in runtime_source


def test_ingestion_chain_has_no_forwarding_runtime_or_worker_bridge() -> None:
    package_root = Path(__file__).parent.parent / "knowledge" / "ingestion"
    runtime_source = (package_root / "runtime.py").read_text(encoding="utf-8")
    service_source = (package_root / "service.py").read_text(encoding="utf-8")
    document_source = (package_root / "document_service.py").read_text(
        encoding="utf-8"
    )

    assert "class LocalIngestionService(IngestionPipeline)" in runtime_source
    assert "_get_ingestion_pipeline" not in runtime_source
    assert "_ingestion_service" not in runtime_source
    assert "parse_file_to_documents" not in service_source
    assert "parse_file_to_documents" not in document_source


def test_ingestion_config_rejects_invalid_resource_limits() -> None:
    for kwargs in (
        {"max_pdf_pages": 0},
        {"batch_size": 0},
        {"workers": 0},
    ):
        try:
            IngestionBatchSettings(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"配置应被拒绝: {kwargs}")


def test_single_worker_ingestion_does_not_create_process_pool(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class Writer:
        def __init__(self) -> None:
            self.documents: list[KnowledgeDocument] = []

        def write(self, documents, **_kwargs) -> None:
            self.documents.extend(documents)

    class Manifest:
        def __init__(self) -> None:
            self.saved: dict[str, str] = {}

        def save(self, record: dict[str, str]) -> None:
            self.saved = dict(record)

    def reject_pool(*_args, **_kwargs):
        raise AssertionError("workers=1 时不应创建进程池")

    def parse_one(args):
        file_path, fingerprint, _source_root, _parser = args
        return (
            file_path,
            fingerprint,
            [KnowledgeDocument("content")],
            None,
            0.01,
        )

    monkeypatch.setattr(ingestion_service_module, "Pool", reject_pool)
    monkeypatch.setattr(ingestion_service_module, "_parse_one", parse_one)
    writer = Writer()
    manifest = Manifest()
    pipeline = IngestionPipeline(
        writer=writer,
        manifest=manifest,
        page_counter=object(),
        preflight=object(),
        document_parser=object(),
        index_changed=object(),
        config=IngestionBatchSettings(workers=1),
    )
    source = tmp_path / "report.txt"

    result = pipeline._run_pipeline(
        data_path=tmp_path,
        data_dir=str(tmp_path),
        all_files=[source],
        new_files=[(source, "fingerprint")],
        skipped_files=[],
        hash_record={},
    )

    assert result["status"] == "completed"
    assert result["files_processed"] == 1
    assert result["chunks_written"] == 1
    assert len(writer.documents) == 1
    assert manifest.saved == {str(source): "fingerprint"}


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
    service = IngestionPipeline(
        writer=UnusedDependency(),
        manifest=Manifest(),
        page_counter=PageCounter(),
        preflight=UnusedDependency(),
        document_parser=UnusedDependency(),
        index_changed=CacheInvalidator(),
        config=IngestionBatchSettings(max_pdf_pages=1),
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
    service = IngestionPipeline(
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
async def test_local_services_require_explicit_configuration() -> None:
    config = KnowledgeSettings()
    runtime = create_knowledge_services(config)

    assert runtime._config is config
    await runtime.close()
