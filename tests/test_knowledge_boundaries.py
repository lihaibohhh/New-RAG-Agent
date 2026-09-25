from __future__ import annotations

from pathlib import Path

import pytest

from knowledge.ingestion import IngestionConfig, IngestionService
from knowledge.runtime import KnowledgeRuntimeConfig
from knowledge.runtime import create_knowledge_runtime


def test_legacy_agent_rag_package_is_removed() -> None:
    package_root = Path(__file__).parent.parent

    assert not (package_root / "react_agent" / "rag").exists()


def test_knowledge_contract_package_does_not_depend_on_service_adapters() -> None:
    package_root = Path(__file__).parent.parent / "knowledge"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in package_root.rglob("*.py")
        if path.relative_to(package_root).parts[0] not in {"client", "server"}
    )

    assert "react_agent" not in source
    assert "from fastapi" not in source
    assert "mcp_server" not in source


def test_rag_and_ingestion_use_cases_are_independent() -> None:
    package_root = Path(__file__).parent.parent / "knowledge"
    rag_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (package_root / "rag").rglob("*.py")
    )
    ingestion_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (package_root / "ingestion").rglob("*.py")
    )

    assert "knowledge.ingestion" not in rag_source
    assert "knowledge.rag" not in ingestion_source


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
