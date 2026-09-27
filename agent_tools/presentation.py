"""具体工具适配器的默认展示元数据。"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


TOOL_PROGRESS_LABELS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "query_internal_knowledge": (
            "正在查询内部知识库……",
            "知识库查询完成，正在分析结果……",
        ),
        "search": (
            "正在搜索公开信息……",
            "公开信息搜索完成，正在分析结果……",
        ),
        "make_excel_table": (
            "正在生成 Excel 文件……",
            "Excel 文件生成完成，正在整理回答……",
        ),
        "docx_tool": (
            "正在生成 Word 报告……",
            "Word 报告生成完成，正在整理回答……",
        ),
        "md_tool": (
            "正在生成 Markdown 文档……",
            "Markdown 文档生成完成，正在整理回答……",
        ),
        "sql_tool": (
            "正在查询结构化财务数据……",
            "财务数据查询完成，正在分析结果……",
        ),
    }
)


__all__ = ["TOOL_PROGRESS_LABELS"]
