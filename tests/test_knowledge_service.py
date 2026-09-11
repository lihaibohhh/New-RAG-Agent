from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager

from knowledge_service.app import create_app
from knowledge_service.settings import KnowledgeServiceSettings
from react_agent.rag.contracts import (
    IngestionReport,
    RagHealthStatus,
    RetrievedChunk,
    RetrievalResult,
    StoredChunk,
    RagDocument,
)
from react_agent.rag.infrastructure.retrieval.chunk_corpus import (
    MigratingChunkCorpusAdapter,
)
from react_agent.rag.infrastructure.storage.sqlite_chunk_store import (
    SQLiteChunkStoreAdapter,
)
from react_agent.rag.runtime.remote import RemoteRagRuntime
from react_agent.rag.runtime.access import create_configured_rag_runtime


class FakeOperations:
    def __init__(self) -> None:
        self.started = False

    def start(self, force: bool = False) -> dict:
        self.started = True
        return {"stage": "started", "force": force}

    def get_status(self) -> dict:
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


class FakeIngestionService:
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


class FakeRuntime:
    def __init__(self) -> None:
        self.operations = FakeOperations()
        self.retrieval = FakeRetrievalService()
        self.admin = FakeAdminService()
        self.ingestion = FakeIngestionService()
        self.closed = False

    def get_retrieval_service(self) -> FakeRetrievalService:
        return self.retrieval

    def get_admin_service(self) -> FakeAdminService:
        return self.admin

    def get_ingestion_service(self) -> FakeIngestionService:
        return self.ingestion

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def service(tmp_path: Path):
    runtime = FakeRuntime()
    app = create_app(
        runtime_factory=lambda: runtime,
        service_settings=KnowledgeServiceSettings(
            api_key="test-secret",
            ingestion_root=tmp_path,
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
    assert runtime.retrieval.calls[0]["retrieval_mode"] == "hybrid"
    assert runtime.closed is True


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
    assert runtime.ingestion.paths == [str((root / "allowed").resolve())]


@pytest.mark.asyncio
async def test_remote_runtime_uses_http_service_without_local_chroma(service) -> None:
    app, server_runtime, _ = service
    async with LifespanManager(app):
        remote = RemoteRagRuntime(
            base_url="http://knowledge.test",
            api_key="test-secret",
            transport=httpx.ASGITransport(app=app),
        )
        result = await remote.get_retrieval_service().search(
            "测试查询",
            top_k=2,
            use_query_cache=False,
            retrieval_mode="bm25",
        )
        health = await remote.get_admin_service().health()
        chunks = await remote.get_admin_service().read_chunks(limit=1)
        report = await remote.get_ingestion_service().ingest("allowed")
        await remote.close()

    assert result.chunks[0].chunk_id == "chunk-8"
    assert result.chunks[0].source_page == 8
    assert health.ready is True
    assert chunks[0].metadata.source_page == 8
    assert report.chunks_written == 2
    assert server_runtime.retrieval.calls[0]["retrieval_mode"] == "bm25"


@pytest.mark.asyncio
async def test_chunk_store_migrates_legacy_corpus_only_once(tmp_path: Path) -> None:
    class LegacyCorpus:
        def __init__(self) -> None:
            self.calls = 0

        def list_documents(self) -> list[RagDocument]:
            self.calls += 1
            return [
                RagDocument(
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
async def test_configured_runtime_selects_remote_client(monkeypatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_SERVICE_URL", "http://knowledge.test")
    monkeypatch.setenv("KNOWLEDGE_SERVICE_API_KEY", "test-secret")

    runtime = create_configured_rag_runtime()

    assert isinstance(runtime, RemoteRagRuntime)
    assert not hasattr(runtime, "_embedding_provider")
    await runtime.close()
