from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager

from knowledge.server.app import create_app
from knowledge.settings import KnowledgeServiceSettings, KnowledgeSettings
from knowledge.contracts import (
    EvaluationCandidate,
    EvaluationRetrievalResult,
    IngestionReport,
    RagHealthStatus,
    RetrievedChunk,
    RetrievalResult,
    StoredChunk,
    KnowledgeDocument,
)
from knowledge.rag.infrastructure.retrieval.chunk_corpus import (
    MigratingChunkCorpusAdapter,
)
from knowledge.foundation.chunk_store import SQLiteChunkStoreAdapter
from knowledge.client import (
    HttpIngestionService,
    HttpRagService,
    create_configured_ingestion_service,
    create_configured_rag_service,
)


@pytest.mark.parametrize(
    "name",
    [
        "KNOWLEDGE_SERVICE_REQUIRE_API_KEY",
        "KNOWLEDGE_SERVICE_WARMUP_ON_START",
        "KNOWLEDGE_SERVICE_EVALUATION_API_ENABLED",
    ],
)
def test_service_settings_reject_invalid_boolean_environment_values(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setenv(name, "tru")

    with pytest.raises(ValueError, match=name):
        KnowledgeSettings.from_env()


class FakeOperations:
    def __init__(self) -> None:
        self.started = False

    async def start(self, force: bool = False) -> dict:
        self.started = True
        return {"stage": "started", "force": force}

    async def get_status(self) -> dict:
        return {
            "task_state": "done",
            "warmup_status": {"state": "done", "timings": {"fake": 0.1}},
        }

    async def ensure_ready(self, wait_seconds: float = 20) -> dict:
        return {
            "ready": True,
            "stage": "ready",
            "waited_seconds": 0,
            "warmup_status": {"state": "done", "timings": {"fake": 0.1}},
        }


class FakeRetrievalService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def search(self, query: str, **kwargs) -> RetrievalResult:
        self.calls.append({"query": query, **kwargs})
        return RetrievalResult(
            query=query,
            chunks=(
                RetrievedChunk(
                    content="证据正文",
                    source_file="report.pdf",
                    source_page=8,
                    chunk_id="chunk-8",
                    score=0.91,
                    doc_type="text",
                    industry="半导体",
                ),
            ),
            stage="fake_retrieve",
            candidates_count=4,
            reranked_count=3,
            top_score=0.91,
            timings={"total": 0.2},
        )


class FakeEvaluationRetrievalService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def search(self, query: str, **kwargs) -> EvaluationRetrievalResult:
        self.calls.append({"query": query, **kwargs})
        candidate = EvaluationCandidate(
            rank=1,
            chunk_id="chunk-8",
            source_file="report.pdf",
            source_page=8,
            content_chars=4,
            doc_type="text",
            industry="半导体",
        )
        return EvaluationRetrievalResult(
            query=query,
            retrieval_mode=str(kwargs.get("retrieval_mode") or "hybrid"),
            chunks=(
                RetrievedChunk(
                    content="证据正文",
                    source_file="report.pdf",
                    source_page=8,
                    chunk_id="chunk-8",
                ),
            ),
            stages={
                "bm25": (candidate,),
                "vector": (),
                "fusion": (candidate,),
                "filtered": (candidate,),
                "reranker_input": (candidate,),
                "reranked": (candidate,),
                "final": (candidate,),
            },
            timings={"bm25": 0.01, "reranker": 0.02, "evaluation_total": 0.03},
            configuration={"query_cache_enabled": False, "device": "cpu"},
        )


class FakeAdminService:
    def __init__(self) -> None:
        self.invalidated: list[str] = []

    async def health(self) -> RagHealthStatus:
        return RagHealthStatus(
            ready=True,
            state="knowledge_base_ready",
            details={"knowledge_base": {"chunk_count": 25911}},
        )

    async def invalidate(self, knowledge_base_id: str = "default") -> None:
        self.invalidated.append(knowledge_base_id)

    async def read_chunks(
        self,
        *,
        source_file: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> tuple[StoredChunk, ...]:
        all_chunks = (
            StoredChunk(
                chunk_id="chunk-8",
                content="证据正文",
                metadata={"source_file": source_file or "report.pdf", "page": 8},
            ),
        )
        return all_chunks[offset : offset + limit if limit is not None else None]


class _FakeIngestionPipeline:
    def __init__(self) -> None:
        self.paths: list[str] = []

    async def ingest(self, data_dir: str) -> IngestionReport:
        self.paths.append(data_dir)
        return IngestionReport(
            data_dir=data_dir,
            status="completed",
            files_discovered=1,
            files_selected=1,
            files_processed=1,
            chunks_written=2,
            cache_invalidated=True,
        )


class FakeRagService:
    def __init__(self) -> None:
        self.operations = FakeOperations()
        self.retrieval = FakeRetrievalService()
        self.evaluation_retrieval = FakeEvaluationRetrievalService()
        self.admin = FakeAdminService()
        self.closed = False

    async def search(self, query: str, **kwargs) -> RetrievalResult:
        return await self.retrieval.search(query, **kwargs)

    async def evaluation_search(
        self,
        query: str,
        **kwargs,
    ) -> EvaluationRetrievalResult:
        return await self.evaluation_retrieval.search(query, **kwargs)

    async def start_warmup(self, force: bool = False) -> dict:
        return await self.operations.start(force=force)

    async def get_warmup_status(self) -> dict:
        return await self.operations.get_status()

    async def ensure_ready(self, wait_seconds: float = 20) -> dict:
        return await self.operations.ensure_ready(wait_seconds)

    async def health(self) -> RagHealthStatus:
        return await self.admin.health()

    async def invalidate(self, knowledge_base_id: str = "default") -> None:
        await self.admin.invalidate(knowledge_base_id)

    async def read_chunks(self, **kwargs) -> tuple[StoredChunk, ...]:
        return await self.admin.read_chunks(**kwargs)

    async def close(self) -> None:
        self.closed = True


class FakeIngestionService:
    def __init__(self) -> None:
        self.delegate = _FakeIngestionPipeline()
        self.closed = False

    async def ingest(self, path: str) -> IngestionReport:
        return await self.delegate.ingest(path)

    async def close(self) -> None:
        self.closed = True


class FakeServices:
    def __init__(self) -> None:
        self.rag = FakeRagService()
        self.ingestion = FakeIngestionService()
        self.closed = False

    async def close(self) -> None:
        await self.ingestion.close()
        await self.rag.close()
        self.closed = True


@pytest.fixture
def service(tmp_path: Path):
    runtime = FakeServices()
    app = create_app(
        runtime_factory=lambda: runtime,
        service_settings=KnowledgeServiceSettings(
            api_key="test-secret",
            ingestion_root=tmp_path,
            evaluation_api_enabled=True,
        ),
    )
    return app, runtime, tmp_path


@pytest.mark.asyncio
async def test_search_requires_key_and_preserves_traceability(service) -> None:
    app, runtime, _ = service
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://knowledge.test",
        ) as client:
            unauthorized = await client.post(
                "/api/v1/retrieval/search",
                json={"query": "台积电资本开支"},
            )
            response = await client.post(
                "/api/v1/retrieval/search",
                headers={"X-Knowledge-Service-Key": "test-secret"},
                json={"query": "台积电资本开支", "top_k": 3},
            )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert response.json()["chunks"][0] == {
        "content": "证据正文",
        "source_file": "report.pdf",
        "source_page": 8,
        "chunk_id": "chunk-8",
        "score": 0.91,
        "doc_type": "text",
        "industry": "半导体",
    }
    assert runtime.rag.retrieval.calls[0]["retrieval_mode"] == "hybrid"
    assert runtime.closed is True


@pytest.mark.asyncio
async def test_evaluation_trace_uses_dedicated_authenticated_endpoint(service) -> None:
    app, runtime, _ = service
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://knowledge.test",
        ) as client:
            unauthorized = await client.post(
                "/api/v1/evaluation/retrieval/trace",
                json={"query": "台积电资本开支"},
            )
            response = await client.post(
                "/api/v1/evaluation/retrieval/trace",
                headers={"X-Knowledge-Service-Key": "test-secret"},
                json={
                    "query": "台积电资本开支",
                    "top_k": 3,
                    "retrieval_mode": "hybrid",
                },
            )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    payload = response.json()
    assert payload["chunks"][0]["content"] == "证据正文"
    assert payload["trace"]["stages"]["bm25"][0] == {
        "rank": 1,
        "chunk_id": "chunk-8",
        "source_file": "report.pdf",
        "source_page": 8,
        "content_chars": 4,
        "doc_type": "text",
        "industry": "半导体",
    }
    assert "content" not in payload["trace"]["stages"]["bm25"][0]
    assert payload["trace"]["configuration"]["query_cache_enabled"] is False
    assert (
        runtime.rag.evaluation_retrieval.calls[0]["retrieval_mode"] == "hybrid"
    )
    assert runtime.rag.retrieval.calls == []


@pytest.mark.asyncio
async def test_evaluation_trace_is_disabled_outside_enabled_profiles(
    tmp_path: Path,
) -> None:
    runtime = FakeServices()
    app = create_app(
        runtime_factory=lambda: runtime,
        service_settings=KnowledgeServiceSettings(
            api_key="test-secret",
            ingestion_root=tmp_path,
            evaluation_api_enabled=False,
        ),
    )
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://knowledge.test",
        ) as client:
            response = await client.post(
                "/api/v1/evaluation/retrieval/trace",
                headers={"X-Knowledge-Service-Key": "test-secret"},
                json={"query": "只有评测才能调用"},
            )

    assert response.status_code == 404
    assert runtime.rag.evaluation_retrieval.calls == []
    assert runtime.rag.retrieval.calls == []


@pytest.mark.asyncio
async def test_ingestion_is_confined_to_server_root(service) -> None:
    app, runtime, root = service
    (root / "allowed").mkdir()
    headers = {"X-Knowledge-Service-Key": "test-secret"}
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://knowledge.test",
            headers=headers,
        ) as client:
            escaped = await client.post(
                "/api/v1/ingestions",
                json={"relative_path": "../outside"},
            )
            windows_absolute = await client.post(
                "/api/v1/ingestions",
                json={"relative_path": "C:\\private\\reports"},
            )
            accepted = await client.post(
                "/api/v1/ingestions",
                json={"relative_path": "allowed"},
            )

    assert escaped.status_code == 400
    assert windows_absolute.status_code == 400
    assert accepted.status_code == 200
    assert runtime.ingestion.delegate.paths == [
        str((root / "allowed").resolve())
    ]


@pytest.mark.asyncio
async def test_remote_runtime_uses_http_service_without_local_chroma(service) -> None:
    app, server_runtime, _ = service
    async with LifespanManager(app):
        remote = HttpRagService(
            base_url="http://knowledge.test",
            api_key="test-secret",
            transport=httpx.ASGITransport(app=app),
        )
        assert not hasattr(remote, "ingest")
        ingestion_remote = HttpIngestionService(
            base_url="http://knowledge.test",
            api_key="test-secret",
            transport=httpx.ASGITransport(app=app),
        )
        result = await remote.search(
            "测试查询",
            top_k=2,
            use_query_cache=False,
            retrieval_mode="bm25",
        )
        traced = await remote.evaluation_search(
            "评测查询",
            top_k=3,
            retrieval_mode="hybrid",
        )
        health = await remote.health()
        chunks = await remote.read_chunks(limit=1)
        report = await ingestion_remote.ingest("allowed")
        await remote.close()
        await ingestion_remote.close()

    assert result.chunks[0].chunk_id == "chunk-8"
    assert traced.stages["bm25"][0].chunk_id == "chunk-8"
    assert traced.configuration["query_cache_enabled"] is False
    assert result.chunks[0].source_page == 8
    assert health.ready is True
    assert chunks[0].metadata.source_page == 8
    assert report.chunks_written == 2
    assert server_runtime.rag.retrieval.calls[0]["retrieval_mode"] == "bm25"


@pytest.mark.asyncio
async def test_chunk_store_migrates_legacy_corpus_only_once(tmp_path: Path) -> None:
    class LegacyCorpus:
        def __init__(self) -> None:
            self.calls = 0

        def list_documents(self) -> list[KnowledgeDocument]:
            self.calls += 1
            return [
                KnowledgeDocument(
                    content="规范正文",
                    metadata={
                        "source_file": "legacy.pdf",
                        "page": 3,
                        "chunk_id": "legacy-3",
                    },
                )
            ]

    legacy = LegacyCorpus()
    store = SQLiteChunkStoreAdapter(database_path=str(tmp_path / "chunks.sqlite3"))
    corpus = MigratingChunkCorpusAdapter(store=store, legacy_source=legacy)

    first = corpus.list_documents()
    second = await corpus.list_chunks(limit=10)

    assert legacy.calls == 1
    assert first[0].metadata.chunk_id == "legacy-3"
    assert second[0].metadata.source_file == "legacy.pdf"
    assert second[0].metadata.source_page == 3
    assert store.snapshot_complete() is True


@pytest.mark.asyncio
async def test_configured_factory_selects_http_rag_service(monkeypatch) -> None:
    monkeypatch.setenv("RAG_RUNTIME_MODE", "remote")
    monkeypatch.setenv("KNOWLEDGE_SERVICE_URL", "http://knowledge.test")
    monkeypatch.setenv("KNOWLEDGE_SERVICE_API_KEY", "test-secret")

    service = create_configured_rag_service()

    assert isinstance(service, HttpRagService)
    assert not hasattr(service, "_embedding_provider")
    await service.close()


@pytest.mark.asyncio
async def test_configured_ingestion_service_is_separate(monkeypatch) -> None:
    monkeypatch.setenv("RAG_RUNTIME_MODE", "remote")
    monkeypatch.setenv("KNOWLEDGE_SERVICE_URL", "http://knowledge.test")

    service = create_configured_ingestion_service()

    assert isinstance(service, HttpIngestionService)
    assert not hasattr(service, "search")
    await service.close()


def test_configured_remote_runtime_requires_service_url(monkeypatch) -> None:
    monkeypatch.delenv("RAG_RUNTIME_MODE", raising=False)
    monkeypatch.delenv("KNOWLEDGE_SERVICE_URL", raising=False)

    with pytest.raises(RuntimeError, match="必须配置 KNOWLEDGE_SERVICE_URL"):
        create_configured_rag_service()


def test_configured_runtime_rejects_unknown_mode(monkeypatch) -> None:
    monkeypatch.setenv("RAG_RUNTIME_MODE", "automatic")

    with pytest.raises(ValueError, match="仅支持 remote 或 local"):
        create_configured_rag_service()


def test_configured_runtime_rejects_local_mode_outside_service(monkeypatch) -> None:
    monkeypatch.setenv("RAG_RUNTIME_MODE", "local")
    monkeypatch.delenv("KNOWLEDGE_SERVICE_URL", raising=False)

    with pytest.raises(RuntimeError, match="只允许 Knowledge Service"):
        create_configured_rag_service()
