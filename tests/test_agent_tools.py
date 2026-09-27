from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

from agent_tools.contracts.retrieval import (
    decode_retrieval_outcome,
    retrieval_meta,
)
from agent_tools.contracts.results import tool_success
from agent_tools.sql import create_sql_tool


class _SqlModel:
    def invoke(self, _prompt: str) -> SimpleNamespace:
        return SimpleNamespace(
            content=(
                "SELECT company, metric, value, unit, year, is_estimate, "
                "source_file, source_page FROM financial_metrics"
            )
        )


def test_retrieval_contract_normalizes_adapter_payload() -> None:
    payload = tool_success(
        tool_name="renamed_private_search",
        query="示例查询",
        data={
            "results": [
                {
                    "content": "可引用正文",
                    "source": "report.pdf",
                    "page": 3,
                    "chunk_id": "report::3",
                }
            ]
        },
        meta=retrieval_meta(
            has_relevant_content=True,
            total_results=1,
        ),
    )

    outcome = decode_retrieval_outcome(payload)

    assert outcome is not None
    assert outcome.ok is True
    assert outcome.has_relevant_content is True
    assert outcome.evidence[0].source_file == "report.pdf"
    assert outcome.evidence[0].source_page == 3
    assert outcome.evidence[0].chunk_id == "report::3"


def test_sql_tool_uses_injected_model_and_database(tmp_path) -> None:
    db_path = tmp_path / "financials.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE financial_metrics (
                company TEXT,
                metric TEXT,
                value REAL,
                unit TEXT,
                year TEXT,
                is_estimate INTEGER,
                source_file TEXT,
                source_page INTEGER
            )
            """
        )
        connection.execute(
            """
            INSERT INTO financial_metrics
            VALUES ('示例公司', '收入', 100, '亿元', '2026Q2', 0, '财报.pdf', 8)
            """
        )

    sql_tool = create_sql_tool(
        db_path=db_path,
        model_provider=_SqlModel,
    )

    payload = json.loads(sql_tool.invoke({"query": "查询示例公司收入"}))

    assert payload["ok"] is True
    assert payload["meta"]["row_count"] == 1
    assert payload["data"]["rows"][0]["source_file"] == "财报.pdf"
    assert payload["data"]["rows"][0]["source_page"] == 8
