"""
app.py:
    创建 FastMCP 实例，注册各组工具
"""
from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP
from react_agent.mcp_server.info_tools import register_info_tools
from react_agent.mcp_server.health_tools import register_health_tools
from react_agent.mcp_server.warmup_tools import register_warmup_tools
from react_agent.mcp_server.rag_tools import register_rag_tools
from react_agent.rag.runtime import RagRuntimePort

MCP_SERVER_NAME = "financial-rag"
ADMIN_TOOLS_ENV = "MCP_EXPOSE_ADMIN_TOOLS"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _server_instructions(expose_admin_tools: bool) -> str:
    instructions = (
        "这是一个金融研报私有知识库检索服务。"
        "当用户需要从已入库研报中查找公司财务数据证据、研报结论、行业观点、政策条款、券商评级、目标价或盈利预测时，"
        "优先且通常只需调用 query_financial_reports 工具检索证据片段；它不是实时数据或结构化 SQL 查询工具。"
        "query_financial_reports 会自动检查 RAG 单例预热状态，必要时触发后台预热并短时间(10-40s)等待；"
        "如果仍未就绪，会返回 warming_up 与 retry_after_seconds，稍后重试同一查询即可。"
        "检索结果会尽量返回来源文档、页码和 chunk_id，具体完整性取决于入库元数据。"
        "当需要了解本 MCP Server 能力边界时，调用 server_info。"
    )

    if expose_admin_tools:
        instructions += (
            "当前已暴露诊断工具：check_knowledge_base、start_rag_singleton_warmup、"
            "get_rag_singleton_warmup_status；这些工具仅用于诊断、显式预热和冷启动排障，"
            "不应作为普通业务检索的必选步骤。"
        )

    return instructions


def create_mcp_server(*, rag_runtime: RagRuntimePort) -> FastMCP:
    """
    创建 MCP Server 实例，并集中注册 MCP 工具。

    mcp_rag_server.py 只负责薄启动、环境初始化和 stdio 运行；
    具体工具注册统一在 react_agent.mcp_server.app 中完成。
    """
    expose_admin_tools = _env_bool(ADMIN_TOOLS_ENV, default=False)
    server = FastMCP(
        name=MCP_SERVER_NAME,
        instructions=_server_instructions(expose_admin_tools),
    )
    register_info_tools(server, expose_admin_tools=expose_admin_tools)

    if expose_admin_tools:
        register_health_tools(
            server,
            service_provider=rag_runtime.get_admin_service,
        )
        register_warmup_tools(server, manager=rag_runtime.operations)

    register_rag_tools(
        server,
        service_provider=rag_runtime.get_retrieval_service,
        warmup=rag_runtime.operations.ensure_ready,
    )

    return server
