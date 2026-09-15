"""RAG 离线评测包。

日常检索改造先运行不调用 LLM 的确定性评测：

``conda run -n new_agent python -m eval.run_eval --deterministic_only``

生成当前知识库的候选数据集：

``conda run -n new_agent python -m eval.dataset_generator``

数据集标签审计：

``conda run -n new_agent python -m eval.audit_dataset --strict``

端到端评测会运行 RAGAS 和回答/拒答裁判并访问真实模型，仅在配置齐备且
明确需要时运行：

``conda run -n new_agent python -m eval.run_eval --n 150``
"""
