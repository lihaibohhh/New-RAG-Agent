"""MCP stdio 进程初始化与正式命令入口。"""
from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from mcp_service.bootstrap import create_configured_mcp_application


logger = logging.getLogger(__name__)
_TRACING_ENV_NAMES = (
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGSMITH_TRACING",
)
_SOURCE_CHECKOUT_MARKERS = ("mcp_rag_server.py", "pyproject.toml")
_LOG_FORMAT = (
    "%(asctime)s - pid=%(process)d - %(name)s - "
    "%(levelname)s - %(message)s"
)


def _runtime_root(
    package_file: str | Path | None = None,
    *,
    cwd: str | Path | None = None,
) -> Path:
    """Resolve runtime files without treating site-packages as a project root."""
    package_path = Path(package_file or __file__).resolve()
    source_root = package_path.parent.parent
    if all((source_root / marker).is_file() for marker in _SOURCE_CHECKOUT_MARKERS):
        return source_root
    return Path(cwd or Path.cwd()).resolve()


def _configure_utf8_streams() -> None:
    if sys.platform != "win32":
        return
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
        )
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer,
            encoding="utf-8",
            errors="replace",
        )


def _resolve_env_path(runtime_root: Path) -> Path:
    configured = os.getenv("MCP_ENV_FILE", "").strip()
    if not configured:
        return runtime_root / ".env"
    path = Path(configured).expanduser()
    return path if path.is_absolute() else runtime_root / path


def _configure_environment(runtime_root: Path) -> Path:
    env_path = _resolve_env_path(runtime_root).resolve()
    load_dotenv(env_path)
    for name in _TRACING_ENV_NAMES:
        os.environ[name] = "false"
    return env_path


def _resolve_log_path(runtime_root: Path) -> Path:
    configured = os.getenv("MCP_LOG_PATH", "").strip()
    if not configured:
        return runtime_root / "logs" / "mcp_debug.log"
    path = Path(configured).expanduser()
    return path if path.is_absolute() else runtime_root / path


def _configure_logging(runtime_root: Path) -> Path | None:
    """Configure stderr logging and add a file handler when it is writable."""
    log_path = _resolve_log_path(runtime_root).resolve()
    stream_handler = logging.StreamHandler(sys.stderr)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
    except OSError as exc:
        logging.basicConfig(
            level=logging.INFO,
            format=_LOG_FORMAT,
            handlers=[stream_handler],
            force=True,
        )
        logger.warning(
            "[RAG-MCP] 日志文件不可写，已降级为仅 stderr "
            "log_path=%s error_type=%s",
            log_path,
            type(exc).__name__,
        )
        return None

    logging.basicConfig(
        level=logging.INFO,
        format=_LOG_FORMAT,
        handlers=[stream_handler, file_handler],
        force=True,
    )
    return log_path


def main() -> None:
    """初始化 stdio 进程并运行，退出时总是关闭远程 Runtime。"""
    runtime_root = _runtime_root()
    _configure_utf8_streams()
    env_path = _configure_environment(runtime_root)
    log_path = _configure_logging(runtime_root)
    logger.info(
        "[RAG-MCP] 运行环境已配置 runtime_root=%s env_path=%s log_path=%s",
        runtime_root,
        env_path,
        log_path or "stderr-only",
    )

    application = create_configured_mcp_application()
    application.run_stdio()


__all__ = ["main"]


if __name__ == "__main__":
    main()
