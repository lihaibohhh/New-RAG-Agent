"""
info.py:
    注册 server_info
"""
from __future__ import annotations

from typing import Any
from mcp.server.fastmcp import FastMCP
from react_agent.mcp_server.responses import mcp_ok


def register_info_tools(server: FastMCP, expose_admin_tools: bool = False) -> None:
    """
    注册 MCP Server 信息类工具。

    这类工具不依赖 RAG / Chroma / Redis，适合作为模块化后的第一个工具。
    """
    available_tools: list[dict[str, Any]] = [
        {
            "name": "server_info",
            "purpose": "查看 MCP Server 基本信息、工具列表和适用边界。",
        },
        {
            "name": "query_financial_reports",
            "purpose": (
                "检索已入库金融研报私有知识库，返回证据片段；尽量附带来源文档、页码和 chunk_id。"
                "不是实时数据或结构化 SQL 查询工具。该工具会自动处理 RAG 单例预热状态。"
            ),
            "best_query_format": "公司名 + 指标/主题 + 报告期或关键词",
            "example": "比亚迪 毛利率 2023Q3",
        },
    ]

    if expose_admin_tools:
        available_tools.extend(
            [
                {
                    "name": "check_knowledge_base",
                    "purpose": "轻量检查 Chroma 向量库是否可访问，返回 chunk 数量；不验证 embedding、reranker、BM25、Redis 或真实检索链路。",
                },
                {
                    "name": "start_rag_singleton_warmup",
                    "purpose": "后台触发 RAG 单例预热，立即返回；只加载 retriever/reranker 两个进程内单例。",
                },
                {
                    "name": "get_rag_singleton_warmup_status",
                    "purpose": "查询后台 RAG 单例预热状态；以 warmup_status.state 判断 not_started/running/done/error/cancelled。",
                },
            ]
        )

    @server.tool(description="返回 financial-rag MCP Server 的基本信息、可用工具和适用边界")
    async def server_info() -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": "financial-rag",
            "version": "0.1.0",
            "transport": "stdio",
            "purpose": "把金融研报私有知识库检索能力暴露给 Claude Code / Claude Desktop / Cursor 等 MCP 客户端。",
            "available_tools": available_tools,
            "best_for": [
                "已入库研报中的公司财务指标证据片段检索",
                "研报结论检索",
                "行业研究观点查找",
                "券商评级、目标价、盈利预测查找",
                "已入库政策文件和监管条款检索",
            ],
            "not_for": [
                "实时股价",
                "实时新闻",
                "非金融研报相关闲聊",
                "没有入库到私有知识库的外部资料",
            ],
        }

        if expose_admin_tools:
            data["admin_tools"] = {
                "exposed": True,
                "env": "MCP_EXPOSE_ADMIN_TOOLS",
                "purpose": "诊断、显式预热和冷启动排障。",
            }

        return mcp_ok(
            data=data,
            meta={
                "tool": "server_info",
                "stage": "static_info",
            }
        )
