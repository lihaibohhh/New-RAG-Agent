"""评测管道的回答生成阶段。"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def generate_answers_sync(records: list[dict], llm: Any) -> list[dict]:
    """使用检索上下文逐条生成答案。"""
    updated = []
    for index, record in enumerate(records, 1):
        question = record["question"]
        context_text = "\n\n".join(record.get("contexts", [])) or "（无检索结果）"
        prompt = (
            "根据以下参考资料，简洁准确地回答问题。"
            "如果资料中没有相关信息，请如实说明。\n\n"
            f"参考资料：\n{context_text}\n\n问题：{question}\n\n答案："
        )
        try:
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.warning("LLM 生成失败 [%s...]: %s", question[:30], exc)
            answer = ""
        updated.append({**record, "answer": answer})
        if index % 10 == 0 or index == len(records):
            logger.info("  生成进度：%s/%s", index, len(records))
    return updated


# 兼容旧测试和调用方。
_generate_answers_sync = generate_answers_sync
