from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import mcp_service.main as mcp_main
from knowledge.contracts import RetrievalResult
from mcp_service.app import create_mcp_server
from mcp_service.bootstrap import McpServiceApplication
from mcp_service.rag_tools import execute_query_financial_reports
from mcp_service.responses import ensure_jsonable, mcp_err, mcp_ok


def test_mcp_service_is_independent_from_agent() -> None:
    project_root = Path(__file__).parent.parent
    package_root = project_root / "mcp_service"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in package_root.rglob("*.py")
    )

    assert package_root.is_dir()
    assert not (project_root / "react_agent" / "mcp_server").exists()
    assert "react_agent" not in source


def test_mcp_stdio_modules_do_not_print_to_stdout() -> None:
    project_root = Path(__file__).parent.parent
    paths = [project_root / "mcp_rag_server.py"]
    paths.extend((project_root / "mcp_service").rglob("*.py"))

    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        print_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        assert not print_calls, f"stdio MCP 模块禁止 print(): {path}"


def test_mcp_response_envelope_remains_json_safe() -> None:
    assert mcp_ok(data={"values": {1, 2}})["ok"] is True
    assert sorted(ensure_jsonable({1, 2})) == [1, 2]
    assert mcp_err("failed", error_type="TestError") == {
        "ok": False,
        "data": None,
        "error": {"type": "TestError", "message": "failed"},
        "meta": {},
    }


@pytest.mark.asyncio
async def test_mcp_query_uses_only_injected_rag_capabilities() -> None:
    service_calls = 0

    class FakeRetrievalService:
        async def search(self, query: str, *, top_k: int) -> RetrievalResult:
            return RetrievalResult(query=query, stage=f"top-{top_k}")

    service = FakeRetrievalService()

    def provide_service() -> FakeRetrievalService:
        nonlocal service_calls
        service_calls += 1
        return service

    async def ready(_wait: int | float) -> dict:
        return {
            "ready": True,
            "stage": "ready",
            "waited_seconds": 0,
            "warmup_status": {"state": "done"},
        }

    result = await execute_query_financial_reports(
        service_provider=provide_service,
        warmup=ready,
        query="evidence",
        top_k=2,
    )

    assert service_calls == 1
    assert result["ok"] is True
    assert result["meta"]["stage"] == "top-2"
    assert result["meta"]["status"] == "ok"
    assert len(result["meta"]["request_id"]) == 32
    assert result["meta"]["elapsed_ms"] >= 0


@pytest.mark.asyncio
async def test_mcp_query_does_not_resolve_service_when_warmup_is_not_ready() -> None:
    def unexpected_service():
        raise AssertionError("预热未完成时不应解析检索服务")

    async def warming(_wait: int | float) -> dict:
        return {
            "ready": False,
            "stage": "warming_up",
            "retry_after_seconds": 2,
            "warmup_status": {"state": "running"},
        }

    result = await execute_query_financial_reports(
        service_provider=unexpected_service,
        warmup=warming,
        query="evidence",
    )

    assert result["ok"] is True
    assert result["data"]["status"] == "warming_up"
    assert result["data"]["retry_after_seconds"] == 2
    assert result["meta"]["status"] == "not_ready"


class _FakeOperations:
    def start(self, force: bool = False) -> dict:
        return {"stage": "started", "force": force}

    def get_status(self) -> dict:
        return {"task_state": "none", "warmup_status": {"state": "not_started"}}

    async def ensure_ready(self, _wait_seconds: int | float = 20) -> dict:
        return {"ready": True, "stage": "ready"}

    async def close(self) -> None:
        return None


class _FakeRagRuntime:
    operations = _FakeOperations()

    def get_retrieval_service(self):
        return object()

    def get_admin_service(self):
        return object()

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_mcp_server_registers_minimal_tool_catalog(monkeypatch) -> None:
    monkeypatch.delenv("MCP_EXPOSE_ADMIN_TOOLS", raising=False)

    server = create_mcp_server(rag_runtime=_FakeRagRuntime())
    tools = await server.list_tools()

    assert {tool.name for tool in tools} == {
        "query_financial_reports",
        "server_info",
    }


@pytest.mark.asyncio
async def test_mcp_server_registers_admin_tools_only_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("MCP_EXPOSE_ADMIN_TOOLS", "true")

    server = create_mcp_server(rag_runtime=_FakeRagRuntime())
    tools = await server.list_tools()

    assert {tool.name for tool in tools} == {
        "check_knowledge_base",
        "get_rag_singleton_warmup_status",
        "query_financial_reports",
        "server_info",
        "start_rag_singleton_warmup",
    }


def test_compatibility_entrypoint_has_no_import_time_runtime(monkeypatch) -> None:
    monkeypatch.delenv("KNOWLEDGE_SERVICE_URL", raising=False)
    sys.modules.pop("mcp_rag_server", None)

    module = __import__("mcp_rag_server")

    assert callable(module.main)
    assert not hasattr(module, "server")
    assert not hasattr(module, "rag_runtime")


def test_mcp_runtime_root_preserves_source_checkout_location() -> None:
    project_root = Path(__file__).parent.parent.resolve()

    assert mcp_main._runtime_root() == project_root


def test_mcp_runtime_root_uses_cwd_for_installed_package(tmp_path: Path) -> None:
    installed_main = tmp_path / "site-packages" / "mcp_service" / "main.py"
    installed_main.parent.mkdir(parents=True)
    installed_main.touch()
    working_directory = tmp_path / "workspace"
    working_directory.mkdir()

    runtime_root = mcp_main._runtime_root(
        installed_main,
        cwd=working_directory,
    )

    assert runtime_root == working_directory.resolve()
    assert "site-packages" not in str(runtime_root)


def test_mcp_env_file_can_be_explicit_and_relative(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MCP_ENV_FILE", "config/mcp.env")

    assert mcp_main._resolve_env_path(tmp_path) == tmp_path / "config" / "mcp.env"


def test_mcp_logging_falls_back_to_stderr_when_file_is_not_writable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    configured: list[dict] = []

    def deny_file_handler(*_args, **_kwargs):
        raise PermissionError("read-only installation")

    monkeypatch.delenv("MCP_LOG_PATH", raising=False)
    monkeypatch.setattr(mcp_main.logging, "FileHandler", deny_file_handler)
    monkeypatch.setattr(
        mcp_main.logging,
        "basicConfig",
        lambda **kwargs: configured.append(kwargs),
    )

    assert mcp_main._configure_logging(tmp_path) is None
    assert len(configured) == 1
    assert len(configured[0]["handlers"]) == 1


def test_application_closes_runtime_when_stdio_server_stops() -> None:
    class StoppingServer:
        def run(self, transport: str = "stdio") -> None:
            assert transport == "stdio"
            raise RuntimeError("server stopped")

    class Runtime:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    runtime = Runtime()
    application = McpServiceApplication(
        server=StoppingServer(),
        rag_runtime=runtime,
    )

    with pytest.raises(RuntimeError, match="server stopped"):
        application.run_stdio()

    assert runtime.closed is True


@pytest.mark.asyncio
async def test_stdio_subprocess_initialize_and_list_tools(tmp_path: Path) -> None:
    project_root = Path(__file__).parent.parent
    environment = dict(os.environ)
    environment.update(
        {
            "MCP_EXPOSE_ADMIN_TOOLS": "0",
            "MCP_LOG_PATH": str(tmp_path / "mcp.log"),
            "PYTHONIOENCODING": "utf-8",
            "RAG_RUNTIME_MODE": "remote",
            "KNOWLEDGE_SERVICE_URL": "http://knowledge.invalid",
        }
    )
    stderr_path = tmp_path / "stderr.log"
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(project_root / "mcp_rag_server.py")],
        cwd=project_root,
        env=environment,
    )

    with stderr_path.open("w+", encoding="utf-8") as stderr:
        with anyio.fail_after(20):
            async with stdio_client(parameters, errlog=stderr) as (read, write):
                async with ClientSession(read, write) as session:
                    initialized = await session.initialize()
                    tools = await session.list_tools()
        stderr.seek(0)
        stderr_output = stderr.read()

    assert initialized.serverInfo.name == "financial-rag"
    assert {tool.name for tool in tools.tools} == {
        "query_financial_reports",
        "server_info",
    }
    assert "Traceback" not in stderr_output
    assert (tmp_path / "mcp.log").is_file()
