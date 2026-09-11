"""
mcp_rag_server.py — 金融研报 RAG MCP Server 薄启动入口

运行方式（stdio 模式，供 Claude Desktop / Cursor / Claude Code 等客户端接入）：
    python mcp_rag_server.py

测试方式（MCP Inspector）：
    npx -y @modelcontextprotocol/inspector -- python mcp_rag_server.py
"""

from __future__ import annotations

import sys
import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ── 路径锚定：无论从哪个工作目录启动，都能正确找到 src 下的项目包 ─────
# 脚本位于 <project_root>/src/mcp_rag_server.py
_HERE = Path(__file__).resolve().parent  # .../src
project_root = _HERE
log_path = project_root / "logs" / "mcp_debug.log"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── Windows UTF-8 流修复 ──────────────────────────────────────
# 必须在 logging.basicConfig 之前执行：
# basicConfig 会捕获当前 sys.stderr 的引用，若之后才替换则 handler 仍可能使用旧编码。
if sys.platform == "win32":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ── 必须在导入项目模块之前加载 .env ──────────────────────────
# 使用绝对路径加载，避免工作目录不确定导致找不到 .env
load_dotenv(_HERE / ".env")

# MCP RAG 进程默认关闭 LangSmith/LangChain 追踪，避免首次 retriever.invoke()
# 触发运行时元数据采集时执行 git describe 并拖慢冷启动查询。
for _tracing_env in (
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGSMITH_TRACING",
):
    os.environ[_tracing_env] = "false"

from react_agent.mcp_server.app import create_mcp_server  # noqa: E402
from react_agent.rag.runtime import create_configured_rag_runtime  # noqa: E402


logger = logging.getLogger(__name__)

# ── 创建 MCP Server 实例 ─────────────────────────────────────
# 具体工具注册在 react_agent.mcp_server.app.create_mcp_server() 内完成。
rag_runtime = create_configured_rag_runtime()
server = create_mcp_server(rag_runtime=rag_runtime)

if __name__ == "__main__":
    # Windows 下 ProactorEventLoop 与部分异步 I/O 库存在兼容问题，
    # 显式切换为 SelectorEventLoop 可降低事件循环冲突概率。
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - pid=%(process)d - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),  # stderr 可写日志，不污染 stdout 协议通道
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True
    )
    logger.info("[RAG-MCP] 日志已配置 log_path=%s", log_path)

    # 不要在这里预热模型。
    # MCP stdio 模式要求服务启动后尽快响应握手；
    # 嵌入模型、精排器、Chroma、Redis 等重资源应在具体工具调用时懒加载。
    try:
        server.run(transport="stdio")
    finally:
        asyncio.run(rag_runtime.close())
