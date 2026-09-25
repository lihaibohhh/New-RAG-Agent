"""兼容建库脚本：所有写库与索引变更通知统一交给 IngestionService。"""
from __future__ import annotations

import argparse
import asyncio

from knowledge.client import create_configured_ingestion_runtime


async def _ingest_knowledge_base(data_dir: str | None):
    runtime = create_configured_ingestion_runtime()
    try:
        return await runtime.get_ingestion_service().ingest(data_dir or ".")
    finally:
        await runtime.close()


def build_vector_db(data_dir: str | None = None) -> dict:
    """保留原脚本函数名，返回可序列化的建库报告。"""
    return asyncio.run(_ingest_knowledge_base(data_dir)).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="增量构建 RAG 向量知识库")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=None,
        help="待入库文档目录；省略时使用项目配置",
    )
    args = parser.parse_args()
    print(build_vector_db(args.data_dir))


if __name__ == "__main__":
    main()
