"""RAG 离线评测包。

日常检索改造先运行不调用 LLM 的确定性评测：

``conda run -n new_agent python -m eval.run_eval --deterministic_only``

数据集标签审计：

``conda run -n new_agent python -m eval.audit_dataset --strict``

旧数据集标签补全（默认另存为 ``eval_dataset_v2.jsonl``）：

``conda run -n new_agent python -m eval.enrich_dataset --sqlite-only``

端到端 RAGAS 评测会访问真实模型，仅在配置齐备且明确需要时运行：

``conda run -n new_agent python -m eval.run_eval --n 150``
"""
