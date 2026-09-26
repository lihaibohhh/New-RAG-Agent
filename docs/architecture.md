# 系统架构说明

本文说明 ReAct-RAG-Agent 的主要模块、依赖边界和运行链路。项目首页与最小启动方式见
[README](../README.md)。

## 总体结构

```text
Streamlit / FastAPI / MCP
          │
          ▼
Application Runtime
    ├── AgentService
    ├── ConversationService
    ├── Tool Adapters
    └── Checkpointer
          │
          ▼
Knowledge Service HTTP API
    ├── RAG Runtime
    └── Ingestion Runtime
```

Agent 负责研究编排、工具选择、上下文和会话；Knowledge Service 是在线知识库资源的
唯一所有者，负责检索、建库、预热和缓存失效。

## Agent 与会话

| 模块 | 职责 |
|---|---|
| `react_agent/agent/service.py` | 对外对话用例，校验 `thread_id`，封装 invoke/stream |
| `react_agent/agent/workflow/` | LangGraph 拓扑、条件路由和节点适配 |
| `react_agent/agent/contracts/` | 图状态和运行时依赖契约 |
| `react_agent/agent/context_management/` | 轮次分段、Token 预算、协议修复、证据索引和历史压缩 |
| `react_agent/agent/policies.py` | 模型轮次、工具批次、RAG 配额和递归安全策略 |
| `react_agent/agent/model_execution.py` | 模型绑定、调用、响应和用量更新 |
| `react_agent/agent/tool_flow/` | 工具调用解析、执行、结果归一化和预算统计 |
| `react_agent/skills/` | 专业工作流的选择、缓存加载和瞬态提示注入 |
| `react_agent/conversations/` | 历史读取、删除和 Checkpointer 适配 |
| `react_agent/runtime/container.py` | 模型、工具、会话和共享资源的应用组合根 |

`AgentService` 不负责历史管理；`ConversationService` 不负责对话执行。FastAPI 的
Chat 和 Sessions 路由分别注入对应服务，但通过组合根共享同一个 Checkpointer。

外部调用必须提供明确的 `thread_id`。API 层再叠加用户命名空间，避免不同用户共享
默认线程或越权读取会话。

## 单轮执行链路

```text
prepare_turn
  → 按当前用户请求选择 Skill
  → call_model
  → tools
  → postprocess_tools
  → call_model / finalize_model
```

Skill 只在当前用户轮中选择和注入，不扩大 Runtime 已注册的工具权限。用户需要在同一轮
明确提出研究和报告交付要求。

正常终止由模型轮次、工具批次、工具重试和 RAG 调用预算控制。`RECURSION_LIMIT` 是图
异常循环的最后熔断器，不作为正常业务预算。

当预算不足以完成下一次工具链时，Agent 会禁用工具并进入最终总结。若模型已经产生工具
调用，则先写入对应 ToolMessage 闭合协议，再执行无工具的 `finalize_model`。

## 上下文与证据

每次模型调用都由 `context_management` 从完整 State 构造独立输入：

- 系统提示词和当前轮始终优先。
- 历史按完整用户轮次裁剪，避免拆散 tool call/tool result。
- RAG 工具正文可被投影或截断，但来源、页码和 chunk ID 尽量保留。
- 当前轮证据按 `chunk_id` 合并为有界证据索引。
- 跨轮来源索引不复制全文，只保存查询和溯源信息。
- 可选历史压缩只替换单次模型输入，不覆盖 Checkpoint 中的原始消息。

模型输入、输出和上下文窗口之间的具体预算关系见
[配置指南](configuration.md)。

## Knowledge 模块边界

`knowledge/` 根层只保存跨模块公共契约、能力接口和组合设施：

| 模块 | 边界 |
|---|---|
| `knowledge/rag/` | 在线查询、混合召回、精排、缓存、评测和预热 |
| `knowledge/ingestion/` | 文档解析、OCR 路由、增量建库和索引写入 |
| `knowledge/foundation/` | 双方确实共享且不依赖内部类型的基础设施 |
| `knowledge/runtime/` | 顶层资源生命周期与跨模块事件连接 |
| `knowledge/transport/` | HTTP Schema 与领域对象 Codec |
| `knowledge/client/` | 独立 HTTP 客户端及远程 RAG/Ingestion Runtime |
| `knowledge/server/` | ASGI 入口、服务端配置和环境组合根 |

依赖约束：

- RAG 与 Ingestion 不互相依赖。
- Transport 不依赖 FastAPI、httpx、Client、Server 或模块内部类型。
- Agent 和 MCP 仅依赖 Knowledge 公共契约或远程客户端。
- 只有 Knowledge Service 在线进程可以打开本地 Chroma、BM25 和 Chunk Store。
- 静态导入方向由 `tests/architecture/` 使用 Python AST 校验。

## RAG 查询链路

```text
Query
  → Semantic Cache
  → BM25 + Vector 并行召回
  → RRF 融合与去重
  → Cross-Encoder Reranker
  → Top-N chunks
  → 来源归一化
```

逻辑 Chunk 同时保存在 SQLite Chunk Store。Chroma/HNSW 是可重建的向量派生索引；
BM25 可从 Chunk Store 重建，避免依赖向量库导出全文。

RAG 结果尽量保留：

- `source_file`
- `source_page`
- `chunk_id`
- 检索和精排阶段所需的分数或轨迹

## 建库链路

```text
PDF / DOCX / TXT / Markdown / CSV / Excel
  → DocumentParsingService
  → PDF Router / Basic Parser / Table Parser
  → ParseResult / ParsedChunk
  → KnowledgeDocument
  → Chunk Store + Chroma
  → 发布索引变化事件
  → RAG 缓存失效或重建
```

普通文本型 PDF 使用本地解析；复杂、扫描、表格密集或多栏 PDF 可以路由到 Docling。
建库提交后由顶层 Runtime 连接事件，不让 Ingestion 反向依赖 RAG。

## API 与 MCP

FastAPI 提供：

- `/api/v1/chat/invoke`：非流式对话
- `/api/v1/chat/stream`：SSE 流式对话
- 会话 history/delete
- API Key、限流和每日 Token 预算
- problem+json 错误体、request_id 和 Prometheus 指标

MCP 使用 stdio 协议。默认只暴露信息查询与 RAG 能力，诊断和预热工具通过
`MCP_EXPOSE_ADMIN_TOOLS=1` 显式开启。stdout 只承载协议消息，日志进入 stderr 或
`MCP_LOG_PATH`。

## 持久化与共享资源

- Agent Checkpoint：PostgreSQL、SQLite 或 Memory。
- Redis：语义缓存、BM25 缓存、限流和预算，不保存 Agent 对话。
- Chunk Store：RAG 逻辑语料快照。
- Chroma：可重建的向量索引。
- PostgreSQL/SQLite Checkpointer：用户会话状态。

显式选择 PostgreSQL 时，依赖缺失、连接失败或初始化失败会阻止应用启动。SQLite 在
本地开发场景可以降级为 Memory，但 readiness 会报告真实后端不一致。
