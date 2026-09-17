# ReAct-RAG-Agent

> 面向金融研报问答、私有知识库检索与多工具分析生成的企业级 ReAct-RAG Agent。  
> 核心目标：让用户用自然语言完成「跨研报检索 → 可信溯源 → 联网补充 → Word / Excel / Markdown 输出 → 多轮会话延续」。

---

## 1. 项目定位

金融分析师、投研人员或行业研究团队每天需要阅读大量研报、年报、公告和政策材料。传统流程通常是逐篇打开 PDF、手动搜索关键词、复制表格、再整理成报告，效率低且容易遗漏来源。

本项目实现了一个面向金融研报场景的 ReAct-RAG Agent，能够：

- 在私有研报知识库中检索公司、行业、指标、事件相关内容；
- 对回答中的关键结论标注来源文件和页码，便于追溯和复核；
- 结合联网搜索补充公开信息；
- 调用工具生成 Excel、Word、Markdown 等结构化交付物；
- 通过 Checkpoint 持久化保存会话上下文，支持多轮追问；
- 通过 Streamlit UI 和 FastAPI REST / SSE 双入口接入前端或外部系统；
- 通过 MCP Server 将金融研报 RAG 能力接入支持 MCP 的客户端。

> 本仓库不包含私有研报数据、向量库、密钥、运行日志和本地数据库。请通过 `.env.example` 配置环境变量，并将自己的 PDF / 数据文件放入本地忽略目录中。

---

## 2. 关于这个项目

| 方向 | 已完成能力 |
|---|---|
| **Agent 架构** | LangGraph `StateGraph` + ReAct 循环，支持模型推理、工具调用、观察、反思与动态路由 |
| **RAG 检索** | PyMuPDF + pdfplumber 金融研报解析，BM25 + 向量双路召回，RRF 融合，Cross-Encoder 精排 |
| **可信溯源** | RAG 结果保留来源文件和页码，回答可追溯、可复核 |
| **多工具输出** | Web 搜索、知识库检索、Excel、Word、Markdown，Text2SQL 模块可按需启用 |
| **记忆与隔离** | Checkpointer 支持 PostgreSQL / SQLite / Memory，多用户 thread 命名空间隔离 |
| **服务化** | FastAPI v1 API，token 级 SSE 流式，断连取消上游 LLM，统一错误信封，request_id 日志 |
| **MCP 接入** | stdio 模式 MCP Server，默认向 Claude Desktop / Cursor / Claude Code / MCP Inspector 暴露 RAG 查询；诊断与预热工具通过 admin 模式显式开启 |
| **安全与成本控制** | API Key 鉴权，匿名 IP 限流，Redis 固定窗口限流，每日 token 预算，fail-open 降级 |
| **可观测性** | Prometheus 指标采集：HTTP 延迟、in-flight、TTFT、tokens、成本、429 / 401 |
| **测试与 CI** | FastAPI 回归测试使用 FakeAgent + fakeredis，避免真实 LLM / Redis 调用，做到零烧钱测试 |
| **评测体系** | RAGAS 自动评测管道，150 条样本、6 个行业分组，记录检索与端到端指标 |

---

## 3. 已验证使用场景

- **跨文档研报问答**：从数百份 PDF 中定位涉及特定公司、行业、财务指标或政策事件的段落，并返回来源页码。
- **财务指标精查**：`sql.py` 提供 Text2SQL 路径，可对 SQLite 财务指标库进行自然语言查询，并返回 `source_file` / `source_page`。
- **联网搜索 + 研究报告生成**：先搜索公开事件，再生成带章节结构和来源标注的 `.docx` 报告。
- **同一会话多格式输出**：一轮分析后继续追问「整理成 Word」或「生成公司对比 Excel」，Agent 会基于当前上下文调用对应工具。
- **外部系统接入**：通过 REST API / SSE 接入 n8n、自定义前端、第三方服务或自动化工作流。

---

## 4. 系统架构

![Architecture](img_1.png)

```mermaid
flowchart LR
    U[User / Frontend] --> S[Streamlit UI]
    U --> A[FastAPI REST + SSE]

    RT[react_agent/runtime<br/>Composition Root] -. creates / injects .-> G
    RT -. creates / injects .-> CS
    RT -. selects / injects .-> T1
    RT -. owns lifecycle .-> CP

    S --> G[AgentService]
    A --> G
    A --> CS[ConversationService]

    G --> LG[LangGraph ReAct Workflow]
    LG --> T1[RAG Tool / HTTP Client]
    LG --> T2[Web Search Tool]
    LG --> T3[Excel / Word / Markdown Tools]

    T1 --> KS[Knowledge Service API]
    KS --> P[PDF Parser]
    P --> R[BM25 + Vector Retriever]
    R --> RR[RRF Fusion]
    RR --> CE[Cross-Encoder Reranker]
    CE --> C[Semantic Cache]

    G --> CP[Checkpoint Store]
    CS --> CP
    C --> Redis[(Redis)]
    R --> Chroma[(Chroma / HNSW)]
    R --> Chunk[(SQLite Chunk Store)]
    CP --> DB[(PostgreSQL / SQLite / Memory)]
```

### 核心链路

```text
用户请求
  → Streamlit / FastAPI
  → AgentService
  → 初始化当前轮模型/工具/恢复预算
  → LangGraph ReAct 循环
  → 工具调用：RAG / Search / Excel / Word / Markdown
  → 工具结果整理与失败恢复
  → 预算耗尽时闭合未执行的 tool calls
  → 无工具 finalize_model 生成最终答复
  → 带来源、工具轨迹、token / cost 信息的响应
```

---

## 5. 核心模块

### 5.1 Agent 与 Conversations

| 文件 | 职责 |
|---|---|
| `react_agent/agent/application/` | 对外 Agent 对话用例；校验 `thread_id`，封装 invoke / stream |
| `react_agent/agent/workflow/` | LangGraph 拓扑、条件路由与节点适配层；`nodes/` 按生命周期、模型和工具节点拆分 |
| `react_agent/agent/contracts/` | 图状态与运行时依赖契约，不包含模型或工具实现 |
| `react_agent/agent/configuration/` | 仅管理图内行为的 `AgentContext`，以及环境变量覆盖与类型转换 |
| `react_agent/agent/policies/` | 模型轮次、工具批次、重试和递归安全预算的纯策略判断 |
| `react_agent/agent/modeling/` | 模型绑定/调用、消息历史清洗与截断，以及当前对话轮次的模型消息选择；不解析价格卡 |
| `react_agent/agent/prompting/` | 默认系统提示词，以及失败恢复/主动收口的瞬态控制指令 |
| `react_agent/agent/tool_flow/` | Agent 内部工具调用解析、执行、结果归一化、限幅与预算统计 |
| `react_agent/agent/support/` | 时间等不包含业务决策的通用辅助能力 |
| `react_agent/agent/__init__.py` | 稳定公共 API；根目录不再存放实现或旧模块兼容门面 |
| `react_agent/configuration/` | 应用级配置模型、环境/YAML 加载及非敏感工具默认值、价格卡；不负责 Agent 图内行为 |
| `react_agent/tooling/` | Agent 与工具适配器共享的 ToolResult 信封和重试执行契约 |
| `react_agent/models/` | LLM Provider 解析、创建与缓存 |
| `react_agent/metering/` | 统一模型 token 归一化、人民币计价及累计状态的轮次用量差值；不决定 Agent 对话边界 |
| `react_agent/infrastructure/` | Redis 等跨用例共享的技术资源适配器 |
| `react_agent/observability/` | 已计算用量的结构化日志、会话汇总与界面展示；不负责重新计量或计价 |
| `react_agent/conversations/contracts.py` | 会话删除结果与持久化错误契约 |
| `react_agent/conversations/ports.py` | `ConversationRepositoryPort` 出站端口 |
| `react_agent/conversations/service.py` | 历史读取与会话删除用例 |
| `react_agent/conversations/infrastructure/` | LangGraph Checkpointer Adapter 及 PostgreSQL / SQLite / Memory 工厂 |
| `react_agent/runtime/container.py` | 选择 LLM Adapter 与 Agent 工具集，创建共享 Checkpointer，完成依赖注入并管理实例生命周期 |

全局配置入口是 `react_agent/configuration/settings.py`。工具默认值和价格卡随
`react_agent.configuration` 一起打包；`.env` / `.env.example` 留在应用根目录，
供本地运行和 Docker Compose 注入环境变量。`pyproject.toml`、Compose 文件与
`pytest.ini` 保留在根目录，供相应构建、部署和测试工具发现。

`AgentService` 不提供历史读取或删除接口；FastAPI Chat 路由只注入
`AgentService`，Sessions 路由只注入 `ConversationService`。两者不互相依赖，
由 Composition Root 共享同一个 Checkpointer。`AgentContext` 只管理提示词、
能力开关、业务预算、上下文控制和图安全熔断。模型选择/推理参数由
`LLMConfig` 管理，具体工具参数由各工具配置管理；模型提供者、计价策略与工具
集合通过 `AgentDependencies` 显式传入，Agent 节点不再自行导入模型工厂或
全局工具表。会话标识由调用方提供，会话后端与数据库路径由
`ConversationPersistenceConfig` 独立管理；FastAPI 和 Streamlit 都通过
`create_application_services(...)` 完成组装，并把返回的服务实例传给
`close_application_services(...)` 精确释放本实例资源。健康检查展示的是实际
生效后端，因此 SQLite/PostgreSQL 降级到 Memory 时不会继续误报原配置值。
FastAPI 与 Streamlit 均通过 `load_conversation_persistence_config()` 读取
`CHECKPOINT_BACKEND` 和 `CHECKPOINT_DB_PATH`；未配置 backend 时安全回退到
SQLite，不再由各启动入口分别硬编码持久化后端。
当显式配置 `CHECKPOINT_BACKEND=postgres` 时，依赖缺失、连接串缺失、连接超时
或建表失败都会终止应用启动，不会再静默降级到易失的 MemorySaver。SQLite
初始化失败时仍保留面向本地开发的 MemorySaver 降级能力。
`/api/v1/health/ready` 比较请求与实际启用的后端：例如请求 SQLite 却降级
MemorySaver 时返回 503；显式 Memory 模式则正常就绪。原有 `/api/v1/health`
和 `/health` 保留存活检查语义，不因后端不一致而返回 503。

Agent 包内部依赖方向固定为：`application → workflow → policies / modeling / tool_flow`
，其中 `contracts` 和 `configuration` 是被依赖的契约与配置层。模型、工具和
持久化的具体 Adapter 只能由 `react_agent/runtime/` 注入，禁止反向导入到
Agent 包。内部调用方统一使用子包路径；外部入口统一从 `react_agent.agent`
导入公共对象。旧的根目录模块路径已经移除，LangGraph 显式节点名称保持不变。

Agent 的正常终止由 `MAX_MODEL_ROUNDS`、`MAX_TOOL_BATCHES` 和
`MAX_TOOL_RETRIES` 控制；`RECURSION_LIMIT` 只作为图异常循环的最后熔断器，
并在 `AgentContext` 初始化时校验其足以覆盖所配置的业务预算。若模型在预算
耗尽时已经生成工具调用，图会先写入 `TOOL_BUDGET_EXHAUSTED` ToolMessage
闭合调用协议，再进入不绑定工具的 `finalize_model`，避免持久化悬空调用。

### 5.2 RAG：私有知识库检索

| 阶段 | 实现 |
|---|---|
| PDF 解析 | PyMuPDF 提取带 bbox 坐标文本块，处理多栏阅读顺序；pdfplumber 提取表格并转 Markdown |
| 分块 | 语法感知分块；PDF 表格预分块透传，避免二次切分破坏表格 |
| 召回 | BM25 + 向量双路 Top-K 召回 |
| 融合 | RRF 融合，合并去重得到候选池 |
| 精排 | Cross-Encoder reranker，返回 `(docs, top_score)` |
| 缓存 | Redis 语义缓存，基于 reranker 置信度门控，低置信结果不固化 |

检索流程：

```text
PDF / DOCX / TXT / Markdown / CSV / Excel
  → DocumentParsingService（唯一格式分派入口）
      ├─ PDF → PdfParserRouter
      │          ├─ 简单文字型 → LocalPdfParserAdapter
      │          └─ 复杂/扫描型 → DoclingServiceAdapter（AUTO/FORCE OCR）
      ├─ 普通文档 → BasicDocumentParserAdapter
      └─ 结构化表格 → StructuredTableParserAdapter
  → 统一 ParseResult / ParsedChunk 契约
  → 内部 RagDocument 边界转换
  → BM25 Top-10 + Vector Top-10
  → RRF 融合到最多 20 个候选
  → Cross-Encoder 精排
  → Top-3 返回给 Agent
```

RAG 采用端口与适配器分层：`query/RetrievalService` 是缓存、召回、精排和
来源归一化的唯一在线编排入口；`ingestion/IngestionService` 负责增量建库后
统一失效查询缓存；Chroma、Redis、Retriever 和 Reranker 位于
`rag/infrastructure/`，由 Knowledge Service 内的实例级 `RagRuntime` 组装。Agent 与 MCP 的 RAG
Adapter 都在注册时接收 `RetrievalService` 提供者，只负责协议转换，执行过程中
不再访问 RAG Service Locator。

`knowledge_service/` 是唯一允许直接打开 `CHROMA_DB_PATH` 的进程边界，提供
检索、预热、真实健康检查、缓存失效、分页 Chunk 读取和受限目录建库 API。
Agent、MCP、Windows CLI 和评测任务使用 `RAG_RUNTIME_MODE=remote` 并配置
`KNOWLEDGE_SERVICE_URL`，通过 `RemoteRagRuntime` 访问服务，不加载
embedding/reranker，也不直接读取 HNSW 文件。远程模式缺少 URL 时启动会立即失败，
不会静默回退为本地 Chroma 所有者。`RAG_RUNTIME_MODE=local` 只用于 Knowledge
Service 本身和显式离线维护。服务端固定单 Worker，避免嵌入式 Chroma 与进程内
BM25 出现多写者或多份失效状态。

评测任务使用独立的 `POST /api/v1/evaluation/retrieval/trace` 管道，
不修改普通 `POST /api/v1/retrieval/search` 的请求、返回值或缓存语义。
专用管道强制绕过查询缓存，除了最终 chunk 外，返回
`bm25`/`vector`/`fusion`/`filtered`/`reranker_input`/`reranked`/`final`
各阶段的排名、chunk ID、溯源元数据与耗时。阶段快照不包含正文，
既能计算召回互补、融合增益和 reranker 损益，又避免报告重复携带大段私有语料。
该接口仍要求 Knowledge Service API Key，并受
`KNOWLEDGE_SERVICE_EVALUATION_API_ENABLED` 开关控制：基础 Compose 默认关闭，
`docker-compose.dev.yml` 仅在开发环境开启。

逻辑 Chunk 同时写入独立 SQLite Chunk Store。旧库首次预热时会从 Chroma
一次性生成事务型快照；快照完成后，BM25 全量重建和评测分页读取都以 Chunk Store
为语料来源。Chroma/HNSW 因此成为可重建的派生向量索引，而不是唯一正文来源。
`rag/operations/` 继续统一管理预热顺序、状态机、幂等、失败、等待和取消规则。
`runtime/offline.py` 继续提供显式知识库路径的离线管理工厂。

文档解析同样只有一条调用链。`ingestion/DocumentParsingService` 只判断文件
类型并委托独立解析器；PDF 专属 `PdfParserRouter` 只做文档级画像、解析器选择、
质量门槛和失败降级。扫描型 PDF 使用 Docling `force OCR`，其他复杂 PDF 使用
`auto OCR`，简单文字型 PDF 使用本地解析器。当前不做逐页混合解析，避免同一
文档多解析器结果拼接造成阅读顺序、页码和重复块问题。各解析器只返回
`ParseResult`，应用端口统一传递 `RagDocument`；LangChain `Document`
只允许出现在 BM25、Chroma 等具体基础设施适配器内部。

### 5.3 Tools：Agent 工具层

| 工具 | 说明 |
|---|---|
| `query_internal_knowledge` | 调用 RAG 管道，返回带来源文件和页码的知识库结果 |
| `search` | Tavily Web 搜索，补充公开信息 |
| `make_excel_table` | 生成 Excel 表格，支持 timestamp / overwrite / append 模式 |
| `docx_tool` | 生成正式 Word 报告，支持标题、正文、列表、表格、页脚等样式 |
| `md_tool` | 生成 Markdown 文档，适合 GitHub、飞书、Notion 等场景 |
| `sql.py` | Text2SQL 财务精查模块，当前可按需加入 Runtime 的 Agent 工具集合 |

> 当前 Agent 默认工具集合由 `react_agent/runtime/container.py` 在组装时显式确定。
> `react_agent/tools/__init__.py` 只导出工具与工厂，不再维护进程级 `TOOLS`。
> RAG、Tavily、Excel、Word 与 Markdown 均由工具工厂接收 Runtime 选择的服务、
> 重试参数、搜索条数或输出目录；工具模块不再自行读取这些运行配置。
> `sql.py` 已实现但默认未注入，因此不会被 Agent 自动调用。

### 5.4 API：FastAPI 服务层

FastAPI 与 Streamlit 并存，共享同一套 Agent 实例与持久化存储。

| 能力 | 说明 |
|---|---|
| 版本化 API | 对话与会话接口统一位于 `/api/v1/*` |
| token 级流式 | `/api/v1/chat/stream` 基于 `astream_events(version="v2")` 输出 token / tool_call / tool_result / done / error |
| 断连取消 | 客户端断开后取消并 await upstream task，避免 LLM 继续消耗 |
| 统一错误 | RFC 7807 风格 problem+json，错误体与响应头均包含 request_id |
| 鉴权 | 支持 `X-API-Key` 与 `Authorization: Bearer` |
| 限流与预算 | Redis 固定分钟窗口限流 + per-user 每日 token 预算 |
| 可观测 | Prometheus 指标采集 HTTP 延迟、TTFT、tokens、cost、in-flight、429 / 401 |
| 会话管理 | history / delete session，bucket_key 命名空间隔离，防 IDOR 越权读取 |


---

### 5.5 MCP Server：外部 MCP 客户端接入

本项目新增 stdio 模式 MCP Server，用于将金融研报 RAG 能力接入 Claude Desktop、Cursor、Claude Code、MCP Inspector 等支持 MCP 的客户端。

| 文件 | 职责 |
|---|---|
| `src/mcp_rag_server.py` | MCP stdio 薄启动入口；负责路径锚定、Windows UTF-8 流修复、加载 `.env`、关闭 LangSmith/LangChain tracing、配置日志并启动 server |
| `react_agent/mcp_server/app.py` | MCP Server 创建与工具注册入口；默认注册 info + rag，`MCP_EXPOSE_ADMIN_TOOLS=1` 时注册 health + warmup |
| `react_agent/mcp_server/info_tools.py` | 注册 `server_info`，返回当前真实可用工具与能力边界；默认不暴露 admin tools |
| `react_agent/mcp_server/health_tools.py` | 注册 `check_knowledge_base`，轻量检查知识库与向量库可用性，可选 admin 工具 |
| `react_agent/rag/operations/warmup.py` | 实例级 RAG 预热状态机，拥有顺序、幂等、失败、等待与取消规则 |
| `react_agent/mcp_server/warmup_tools.py` | 注册 `start_rag_singleton_warmup` / `get_rag_singleton_warmup_status`，可选 admin 工具 |
| `react_agent/mcp_server/rag_tools.py` | 注册 `query_financial_reports`，对外暴露金融研报 RAG 查询 |
| `react_agent/mcp_server/responses.py` | 统一 MCP 工具返回结构，如 `mcp_ok` / `mcp_err` |

MCP 启动方式：

```bash
cd src
python mcp_rag_server.py
```

使用 MCP Inspector 测试：

```bash
cd src
npx -y @modelcontextprotocol/inspector -- python mcp_rag_server.py
```

使用 Codex 外接测试
```bash
codex mcp add financial-rag --env MCP_EXPOSE_ADMIN_TOOLS=0 -- "E:\anaconda\envs\agent_base\python.exe" "E:\transformer_program\nanoGPT_program\AI_Agent\agent_v0\react-agent-main\src\mcp_rag_server.py"
# 确认是否注册：
/mcp
query: 根据内部数据库，2026 年 1 月，国内领先的芯片设计企业兆易创新和谁正式签署战略合作协议？
```

MCP 启动入口遵循“薄启动”原则：启动阶段不预热 embedding、reranker、Chroma、Redis 等重资源，避免 stdio 握手阶段阻塞。普通业务查询只需调用 `query_financial_reports`；首次查询会自动触发 retriever / reranker 单例预热并进行有限等待。诊断或显式预热场景可设置 `MCP_EXPOSE_ADMIN_TOOLS=1`，再调用 `start_rag_singleton_warmup` / `get_rag_singleton_warmup_status`。
MCP 启动时根据 `RAG_RUNTIME_MODE` 创建 Runtime；默认 `remote`，此时必须配置
`KNOWLEDGE_SERVICE_URL`，不会隐式打开本地 Chroma。注册函数只接收显式的 Query、
Admin 和 Operations 提供者；关闭 stdio Server 后由入口释放客户端连接，不依赖
模块级 RAG 服务单例。

---

## 6. API 示例

### 健康检查

```bash
curl http://localhost:8000/health
```

### 非流式调用

```bash
curl -X POST http://localhost:8000/api/v1/chat/invoke \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "message": "总结新能源行业最近的投资主线，并给出来源",
    "session_id": "demo-session"
  }'
```

### SSE 流式调用

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "message": "检索半导体行业最近的供需变化，并生成要点",
    "session_id": "demo-session"
  }'
```

流式事件类型：

```text
token | tool_call | tool_result | usage | done | error
```

`usage.cost` 是本次模型调用的人民币估算值；未计价时为 `null`，并通过
`cost_status` 标明状态。`done.total_cost` 仅累计已计价调用，新增的
`currency=CNY` 与 `unpriced_model_count` 用于避免把未知价格误认为零费用。
流式和非流式均按当前轮全部模型调用累计 token，再写入 API 每日 token 预算。
流式请求若在模型返回 usage 前断连，进行中调用的精确 token 数可能不可得；
预算仅记录截至断连已收到的模型结束事件用量。
Prometheus 费用指标为 `llm_cost_cny_total`；原 USD 指标已停用，已有监控面板
需要切换查询名。旧 Checkpoint 中的 `estimated_cost_usd` 仅作为兼容字段保留，
新调用只写入 `estimated_cost_cny`。

### Prometheus 指标

```bash
curl http://localhost:8000/api/v1/metrics
```

---

## 7. 快速启动

### 7.1 环境准备

推荐 Python 3.10+。

```bash
git clone https://github.com/lihaibohhh/ReAct-RAG-Agent.git
cd ReAct-RAG-Agent/src
```

使用项目指定的 Conda 环境，以 `pyproject.toml` 安装运行与开发依赖：

```powershell
conda run -n new_agent python -m pip install -e . --group dev
```

### 7.2 配置环境变量

```bash
cp .env.example .env
```

根据自己的模型和工具服务填写：

```env
MODEL=deepseek/deepseek-flash
DEEPSEEK_API_KEY=your_deepseek_api_key

TAVILY_API_KEY=your_tavily_api_key

# Redis 只承担缓存、限流与预算；Agent 会话使用 PostgreSQL
REDIS_URL=redis://localhost:6379/0
CHECKPOINT_BACKEND=postgres
POSTGRES_DB_URL=postgresql://user:password@127.0.0.1:5432/finance_agent
```

如果 Agent 在 Docker 容器中运行，而 PostgreSQL 安装在 Windows 本机，另设：

```env
POSTGRES_DOCKER_DB_URL=postgresql://user:password@host.docker.internal:5432/finance_agent
```

连接串只保存在本地 `.env`，不要提交 GitHub。`data_sql/financials.db`
仍是 SQL 工具的旧实验数据源，不属于 Agent 会话库；当前保留但不迁移，后续可独立重构或废弃。

DeepSeek 费用估算由 `react_agent/configuration/deepseek_pricing.yaml` 独立管理。价格卡使用人民币、
按北京时间区分峰时与闲时，并把旧模型名归一化到实际计费模型。API 返回未知模型
时费用会显示为“未计价”并写入告警，不会静默记为零。修改价格卡后重启
`agent-app` 即可生效；模型名变化仍通过 `.env` 的 `MODEL` 配置，并使用
`docker compose ... up -d --force-recreate agent-app` 让新环境变量进入容器。

### 7.3 使用 Docker Compose 启动应用

应用镜像以非 root 用户运行。Compose 分别启动 Streamlit 与单 Worker
Knowledge Service；后者是 Chroma、BM25、embedding、reranker 和 Chunk Store
的唯一所有者。本地数据库、研报、向量库和日志不会写入镜像，而是在运行时挂载。

Compose 只把 `chroma_db`、`data/knowledge`、模型缓存和私有研报挂载给 Knowledge
Service；Agent 只挂载专用的 `data/agent-state` 会话目录，并通过
`http://knowledge-service:8001` 访问知识库。当前为兼容既有数据仍使用宿主 bind
mount，但必须停止 Windows 本地 Chroma 进程和旧 Agent 容器，避免同时打开同一
目录。后续完成备份恢复演练后可把该挂载替换为 Docker named volume。
混合检索使用 `rank-bm25`，首次生成 Chunk Store 后，Redis 缓存失效会从独立
SQLite 语料快照重建 BM25。BM25 建库和查询统一使用 Jieba 搜索模式，
同时执行 NFKC/大小写规范化、金融科技词典和英文数字型号保护。
分词策略指纹会随 Redis 索引一起保存；旧分词或词典变化后的缓存会被自动拒绝并
从 Chunk Store 重建。查询最高 BM25 分数不大于 0 时返回空候选，不再把固定语料顺序
误当作相关结果。

```powershell
docker compose config
docker compose build knowledge-service agent-app
docker compose up -d redis knowledge-service agent-app
docker compose ps
```

设置随机长密钥后，检查服务状态：

```powershell
curl.exe http://127.0.0.1:8001/api/v1/health/live
curl.exe http://127.0.0.1:8001/api/v1/health/ready
```

`live` 只说明 HTTP 进程存活；`ready` 会真实打开 Chroma 并统计 Chunk，因此能发现
HNSW 加载失败。首次启动会在后台预热，并从现有 Chroma 生成
`data/knowledge/chunks.sqlite3`。引入中文分词后，28,771 Chunk 的本地独立实测为：
Chunk Store 读取约 1.8 秒、BM25 全量构建约 56.5 秒；旧的空格分词 4–5 秒数据
不再具有参考性。Smoke 集纯 BM25 Top-10 的平均查询耗时约 425ms。
这些阶段部分并行，实际总时长以 `/api/v1/runtime/warmup` 返回的 timings 为准。
Knowledge Service 默认设置 `HF_HUB_OFFLINE=true` 和
`TRANSFORMERS_OFFLINE=true`，只使用挂载的模型缓存，避免服务重启时因 Hugging
Face 网络波动导致预热失败。首次部署且缓存为空时，可在 `.env` 中临时设为
`false` 完成模型下载，确认缓存完整后恢复 `true`。

Redis 使用固定的官方服务端镜像，不包含 RedisInsight，也不承担向量检索或
Agent Checkpoint。Redis 和 Streamlit 端口均只绑定 `127.0.0.1`，适合本机开发、
录屏和面试演示；缓存不做磁盘持久化，容器重建后会自动重建。将来对外部署时
应通过应用入口和 HTTPS 暴露，不能直接开放 Redis。

默认从官方 PyPI 安装依赖；需要使用国内镜像时可显式覆盖构建参数：

```powershell
docker compose build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple knowledge-service agent-app
```

Streamlit 地址：

```text
http://localhost:8501
```

### 7.4 按需启动 Docling 离线解析

Docling 只用于解析和建库，不参与日常在线检索。先在 `.env` 中启用自动路由：

```env
DOCLING_ENABLED=true
DOCLING_PARSER_MODE=auto
DOCLING_BASE_URL=http://127.0.0.1:5001
```

需要建库时启动 CPU 单 Worker 容器：

```powershell
docker compose --profile ingestion up -d docling
docker compose --profile ingestion ps
curl.exe http://127.0.0.1:5001/ready
```

由 Knowledge Service 发起增量建库；路径只能相对于
`KNOWLEDGE_SERVICE_INGESTION_ROOT`，不能传 Windows 或容器绝对路径：

```powershell
curl.exe -X POST http://127.0.0.1:8001/api/v1/ingestions `
  -H "Content-Type: application/json" `
  -H "X-Knowledge-Service-Key: <your-key>" `
  -d '{"relative_path":"."}'
```

`auto` 模式会将扫描件、图像密集、表格密集或多栏 PDF 交给 Docling，普通文本 PDF 继续使用本地解析器。本地结果还会检查页码覆盖、文本量、乱码率和溯源完整性，不达标时再转交 Docling；Docling 失败时仅在 `auto` 模式回退到可用的本地结果。超过 `DOCLING_MAX_PDF_PAGES`（默认 200）页的 PDF 会在哈希和解析前直接排除。其余大文件由 RAG 适配器使用异步任务接口轮询，不受 10 分钟同步等待上限影响。

建库完成后可停止服务：

```powershell
docker compose --profile ingestion stop docling
```

### 7.5 在 Conda 环境启动 Streamlit

```bash
cd src
streamlit run tests/test_agent.py
```

### 7.6 在 Conda 环境启动 FastAPI

```bash
cd src
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger 文档：

```text
http://localhost:8000/docs
```

---

## 8. 测试与评测

### 8.1 API 回归测试

```bash
pytest tests/api -q
```

测试设计要点：

- 使用 FakeAgent 替代真实 LLM；
- 使用 fakeredis 替代真实 Redis；
- 覆盖鉴权、限流、错误信封、SSE 断连、Prometheus、会话 CRUD、IDOR 防护；
- CI 中不调用真实模型，不消耗 API 费用。

### 8.2 RAG 检索与 RAGAS 评测

评测实现按职责分为两个子包：`eval/dataset/` 负责数据模型、Chunk 数据源、
分层抽样、证据校验、问答生成和数据集 Bundle；`eval/pipeline/` 负责检索、
回答生成、回答行为裁判、RAGAS、聚合与报告。`eval.run_eval` 仅保留 CLI、
运行配置解析和各评测阶段的顺序编排。数据集生成和审核分别使用
`python -m eval.dataset` 与 `python -m eval.dataset.audit`。

```bash
# 从当前知识库分层采样并生成 full/smoke/regression 候选集。
# 此步骤会将抽样 chunk 发送给配置的外部 LLM，并产生模型费用。
conda run -n new_agent python -m eval.dataset \
  --output eval/results/eval_dataset_docling_v1.candidate.jsonl \
  --max_chunks 150 --n_per_chunk 1 \
  --multi_chunk_ratio 0.25 --no_answer_count 20

# 审核候选集：核对问题、答案、引文；通过项改为 review_status=approved。
# 将定稿 full 集复制到 eval/dataset 后再严格审计。
conda run -n new_agent python -m eval.dataset.audit \
  --dataset eval/dataset/eval_dataset_docling_v1.jsonl --strict

# 日常检索回归：Hit@K、Precision@K、Recall@K、MRR、nDCG@K，
# 不调用 LLM，并默认绕过语义查询缓存
conda run -n new_agent python -m eval.run_eval --deterministic_only --n 150

# 用相同数据分别测候选源（两种模式都会自动绕过混合查询缓存）
conda run -n new_agent python -m eval.run_eval --deterministic_only --retrieval_mode bm25 --n 150
conda run -n new_agent python -m eval.run_eval --deterministic_only --retrieval_mode vector --n 150

# 仅检索 RAGAS：调用裁判 LLM 计算 Context Precision/Recall
conda run -n new_agent python -m eval.run_eval --retrieval_only --n 150

# 端到端：生成答案；正样本计算 RAGAS，全部样本执行回答/拒答裁判
conda run -n new_agent python -m eval.run_eval --n 150
```

正式基线建议使用独立的一次性评测容器。它不挂载 Chroma、Chunk Store、模型缓存
或私有研报，只能通过 Knowledge Service API 检索和分页读取 chunks；输入数据集
只读挂载，报告写入宿主机 `eval/results/`：

```powershell
# 开发阶段合并 dev 配置，eval-runner 直接只读挂载最新源码，无需重建镜像
$compose = @('-f', 'docker-compose.yml', '-f', 'docker-compose.dev.yml')

# 从 Knowledge Service API 读取当前 chunks，生成候选 QA 数据集。
# 同时导出 full、smoke、regression 和 manifest；无答案项必须人工确认。
docker compose @compose --profile eval run --rm eval-runner `
  python -m eval.dataset `
  --output /app/eval-results/eval_dataset_docling_v1.candidate.jsonl `
  --max_chunks 150 --n_per_chunk 1 `
  --multi_chunk_ratio 0.25 --no_answer_count 20

# 审核后将定稿文件放入宿主机 eval/dataset/，再严格审计
docker compose @compose --profile eval run --rm eval-runner `
  python -m eval.dataset.audit `
  --dataset /app/eval/dataset/eval_dataset_docling_v1.jsonl --strict

# 无 LLM、禁用查询缓存的正式混合检索基线
docker compose @compose --profile eval run --rm eval-runner `
  python -m eval.run_eval `
  --dataset /app/eval/dataset/eval_dataset_docling_v1.jsonl `
  --deterministic_only --retrieval_mode hybrid --n 150
```

`eval.run_eval` 在未传 `--allow_query_cache` 时默认调用上述专用管道，
并把阶段追踪写入明细 CSV。只有显式允许查询缓存时才回退到普通检索端点，
此时不生成阶段追踪，不应用于 BM25/Vector/Hybrid 的正式 A/B。

将最后一条命令的 `retrieval_mode` 分别改成 `bm25`、`vector` 即可完成
三路消融评测。`run_eval` 默认读取 `EVAL_RESULTS_DIR`，也可通过
`--output-dir` 指定其他可写目录。只有去掉 `--deterministic_only` 时才会加载
生成和 RAGAS 裁判模型并可能产生外部模型费用。端到端模式默认增加结构化
回答行为裁判；临时排障时可用 `--skip_answerability_judge` 跳过，但正式评测
不建议关闭。

新生成的数据集包含 `case_id`、`gold_chunk_ids`、`gold_sources`、
`gold_evidence`、`evidence_validation`、`review_status`、`answerable` 和
`category`。可回答项会先通过引文、答案数字和多 chunk 覆盖的确定性校验；
无答案项仍需执行全库检索筛查并人工确认。评测器仍支持 `source_file + page` 的页码级标签，
但正式 A/B 应优先使用 `gold_chunk_ids`。解析器更换前的历史报告仅作归档，
不能作为当前 Docling 知识库的基线。只有在明确需要且真实模型配置齐备时
才运行 RAGAS 模式。

无答案样本不参与 Recall/MRR/nDCG 或 RAGAS 的正样本均值。检索层单独报告
`retrieval_abstention_rate`（精排后是否返回空上下文）；端到端回答层报告：

- `abstention_accuracy`：无答案问题中正确拒答的比例；
- `hallucination_rate`：无答案问题中给出无依据答案的比例；
- `false_refusal_rate`：可回答问题中错误拒答的比例；
- `negative_label_conflict_rate`：裁判发现检索上下文实际足以回答，提示负样本标签可能错误；
- `answerability_accuracy`：可回答/不可回答两组的综合行为准确率。
- `answerability_judge_coverage`：成功完成行为判定的样本占比；批次裁判遗漏的样本会自动单条重试一次。

检索延迟汇总优先使用专用评测管道的 `evaluation_total`，因此 P50/P95
包含召回、融合和 Reranker；普通检索端点没有该字段时回退到 `total`。

### 8.3 向量库建库性能

| 指标 | 串行版 | 多进程版 | 变化 |
|---|---:|---:|---:|
| 全量建库总耗时 | ~57 min | ~28 min | 约 2 倍提速 |
| T1+T2 解析 | ~56 min | ~27.6 min | 4 进程并行 |
| T3 向量化 | 45.7s | 62.47s | 批次结构不同，略增 |
| T4 写入 | 24.4s | 49.48s | `add → upsert` 换取幂等重跑 |
| 崩溃重跑 | 易 DuplicateIDError | upsert 幂等 | 稳定性提升 |

---

## 9. 工程亮点与关键取舍

### 9.1 PDF 解析：从“能读”到“能用于金融研报”

金融研报常见双栏排版、表格密集、页眉页脚干扰。简单 PDF loader 容易造成左右栏交错、表格断裂、页码丢失。本项目改用 PyMuPDF 读取坐标块，按版面重建阅读顺序；表格由 pdfplumber 抽取后转 Markdown 整块保留，避免字符分块器破坏行列结构。

### 9.2 检索：双路召回 + RRF + reranker

BM25 擅长股票代码、公司名、指标名等精确匹配；向量检索擅长语义近似。两路 Top-K 召回后用 RRF 融合，再交给 Cross-Encoder 精排。召回池从 10 扩到 20 后，减少相关文档在检索阶段被提前丢弃的问题。

### 9.3 语义缓存：只缓存“可信结果”

RRF 候选池最多保留 20 条；进入 Cross-Encoder 前还会按设备档位限流。
CPU 默认评分 12 条，CUDA 默认评分 20 条，在个人开发机上兼顾检索质量和等待时间。

早期实现中，低相关结果也会被缓存 1 小时，后续相似 query 命中缓存后会跳过新检索，造成错误固化。本项目让 reranker 返回 `top_score`，写入缓存前进行门控：

| top_score | 策略 |
|---|---|
| ≥ 0.5 | 正常缓存 |
| 0 ~ 0.5 | 不缓存，避免低质结果固化 |
| 空结果 | 短 TTL 缓存，允许后续重试 |

### 9.4 工具调用协议：把消息序列合法性作为强约束

DeepSeek / LangChain 工具调用中，复杂工具参数可能进入 `.invalid_tool_calls`，如果路由只看 `.tool_calls`，就会误判“没有工具调用”，导致悬空 tool_call 被写入 checkpoint，下一轮恢复历史时触发多轮 400 死锁。

本项目统一封装 `_ai_tool_call_ids()`，同时覆盖 `.tool_calls` 与 `.invalid_tool_calls`，并在路由、入口净化、异常兜底三层使用同一检测口径。这个问题的教训是：**协议检测必须与实际发送给模型的序列化路径保持一致**。

### 9.5 FastAPI serving：从 demo 接口到可运营服务

这轮服务化改造不仅是“加几个接口”，而是补齐了 LLM 应用上线前常见的工程护栏：

- API versioning；
- request_id 全链路追踪；
- problem+json 统一错误；
- API Key 鉴权；
- 匿名 IP bucket 限流；
- per-user token 预算；
- token 级 SSE；
- 断连取消上游 LLM；
- Prometheus 指标；
- 零真实依赖的回归测试。


### 9.6 MCP RAG 冷启动处理

MCP 改造中定位到的问题：首次 `query_financial_reports` 变慢的主要根因不是 BM25、Chroma、Redis 或检索/精排算法本身，而是 LangSmith/LangChain tracing 在首次底层 retriever 调用时采集 runtime metadata，触发 `git describe --tags --always --dirty` 并卡住 40 秒以上。

已落地的处理方式：

- `mcp_rag_server.py` 改为 stdio 薄启动入口，启动时不做重资源预热；
- MCP 进程启动后强制关闭 `LANGCHAIN_TRACING_V2`、`LANGSMITH_TRACING_V2`、`LANGCHAIN_TRACING`、`LANGSMITH_TRACING`；
- MCP 相关工具拆分到 `react_agent/mcp_server/` 目录，按职责模块化注册；
- 预热统一通过 `RagAdminService.warmup()` 加载 Retriever 与 Reranker 进程内单例；
- `start_rag_singleton_warmup` 可手动后台触发单例预热，立即返回；
- `get_rag_singleton_warmup_status` 用于查询后台单例预热状态；
- RAG 查询工具保留语义缓存命中路径，减少重复查询成本。

当前预热会在加载 Retriever 与 Reranker 单例后，分别执行一次真实 embedding
和 reranker 前向推理。在线 RAG 根据 `RAG_DEVICE=auto|cpu|cuda` 选择运行档位：
CPU 默认单精排、12 个候选和 120 秒工具预算；CUDA 默认单精排、20 个候选和
45 秒预算。召回、缓存、精排排队和推理耗时会进入日志或工具元数据。

模型计算超时不会自动重试。同步模型推理进入 Python 工作线程后不能由协程安全
终止，立即重试只会重复占用计算资源；进程级精排闸门用于阻止多个重任务同时争抢
CPU/GPU。

推荐使用流程：

```text
1. List Tools
2. query_financial_reports
```

如需管理员提前预热，可手动执行：

```text
1. MCP_EXPOSE_ADMIN_TOOLS=1
2. start_rag_singleton_warmup
3. get_rag_singleton_warmup_status，直到 warmup_status.state=done
4. query_financial_reports
```

当前已确认的问题边界：预热可以消除模型首次前向计算的额外开销，但不能替代在线
查询的并发控制、候选限流和合理的设备超时预算。

---

## 10. 目录结构

```text
react-agent-main/
└── src/
    ├── api/                         # FastAPI 服务层
    │   ├── main.py                  # app 入口、lifespan、路由挂载
    │   ├── settings.py              # 配置校验
    │   ├── errors.py                # problem+json 错误信封
    │   ├── middleware.py            # request_id 日志与 Prometheus 采集
    │   ├── security.py              # API Key 鉴权
    │   ├── ratelimit.py             # Redis 限流与 token 预算
    │   ├── metrics.py               # Prometheus 指标
    │   └── routes/
    │       └── v1/
    │           ├── chat.py          # v1 token 级 SSE / invoke
    │           └── sessions.py      # 会话历史与删除
    ├── react_agent/
    │   ├── agent/                   # Agent 图、节点、路由、状态、提示词和对话用例
    │   ├── conversations/           # 会话契约、Repository Port、管理用例和持久化 Adapter
    │   ├── runtime/                 # LLM、Tools、Agent、Conversations 与生命周期的应用级 Composition Root
    │   ├── configuration/           # 全局配置加载、校验与非敏感 YAML 默认值
    │   │   ├── settings.py          # 环境变量、路径、配置模型与聚合入口
    │   │   ├── config.yaml          # 工具层非敏感默认配置
    │   │   └── deepseek_pricing.yaml # DeepSeek 价格卡
    │   ├── rag/                     # 独立 RAG 业务边界
    │   │   ├── contracts.py         # 来源、元数据、解析、查询、建库与健康状态契约
    │   │   ├── ports.py             # 解析、缓存、召回、精排、建库、知识库只读端口
    │   │   ├── query/               # RetrievalService：唯一在线检索编排
    │   │   ├── ingestion/           # 文档解析入口、PDF 路由、建库与缓存一致性
    │   │   ├── admin/               # RagAdminService：健康、人工失效与离线语料读取
    │   │   ├── operations/          # 预热状态机、幂等、等待、失败与取消策略
    │   │   ├── infrastructure/      # 技术实现与端口适配器
    │   │   │   ├── parsing/         # 本地 PDF、Docling、普通文档与表格解析
    │   │   │   ├── retrieval/       # Chroma/BM25 候选源、Redis 索引仓库、RRF 与精排
    │   │   │   ├── cache/           # Redis 精确与语义查询缓存
    │   │   │   ├── models/          # Runtime 持有的懒加载 Embedding Provider
    │   │   │   └── storage/         # Chroma 建库、更新、诊断和只读访问
    │   │   └── runtime/             # 实例级生产对象图、设备档位与资源生命周期
    │   │       ├── offline.py       # 显式知识库路径的离线管理服务工厂
    │   │       └── testing.py       # 服务容器与预热状态的测试重置入口
    │   ├── tools/                   # RAG / Search / Excel / Word / Markdown / SQL
    │   ├── tooling/                 # ToolResult 公共契约与工具重试策略
    │   ├── models/                  # LLM Provider 工厂
    │   ├── metering/                # 模型用量归一化、人民币计价与轮次投影
    │   ├── infrastructure/          # Redis 等共享技术适配器
    │   ├── observability/           # 已计算用量的日志与界面展示
    │   ├── mcp_server/              # MCP 工具注册与 stdio Server 模块
    │   │   ├── app.py               # 创建 MCP Server；默认注册 info/rag，admin 模式注册 health/warmup
    │   │   ├── info_tools.py        # server_info：当前可用工具与能力边界
    │   │   ├── health_tools.py      # check_knowledge_base，可选 admin 工具
    │   │   ├── warmup_tools.py      # RAG 单例预热启动 / 状态查询，可选 admin 工具
    │   │   ├── rag_tools.py         # query_financial_reports
    │   │   └── responses.py         # mcp_ok / mcp_err 返回结构
    ├── eval/                        # RAGAS 评测脚本与数据生成
    ├── scripts/                     # PDF 检查、财务抽取、Redis 检查、历史管理
    ├── tests/                       # Streamlit 入口与 API 回归测试
    ├── mcp_rag_server.py            # MCP stdio 薄启动入口：关闭 tracing / 日志 / create_mcp_server
    ├── pyproject.toml
    ├── .env.example
    └── .gitignore
```

---

## 11. 当前边界与后续优化

| 问题 | 当前状态 | 后续方向 |
|---|---|---|
| 图表型研报财务抽取 | 文本 chunk 对图表主导型 PDF 支持有限 | 引入多模态模型逐页识别图表 |
| 评测集指代不明 | 自动生成 QA 会产生“该公司”这类缺实体问题 | 分块时注入公司名 / 行业名等全局元数据 |
| 固定窗口限流 | 窗口边界存在短时 2× 突发 | 生产高并发可升级为滑动窗口或令牌桶 |
| token 预算事后扣费 | 单条超长请求可能先超过预算 | 增加单请求 max_tokens 硬限制 |
| 历史工具链干扰 | 已用 Guard 和跨轮计数缓解 | 将上一轮 tool call chain 压缩为中性摘要 |
| MCP 首次 query 长尾 | 已定位为 LangSmith / LangChain tracing 触发 `git describe` 长尾；MCP 进程已强制关闭 tracing，warmup 降级为 retriever/reranker 单例加载 | 保持 tracing 关闭；如需排障可独立压测 `bm25.invoke()` / `vector.invoke()` / Chroma 查询 |

---

## 12. 更新日志

### 2026-09-07：RAG 核心元数据契约收敛

- 新增 `SourceReference` 与 `ChunkMetadata`，统一定义来源、页码、chunk ID、文档类型、行业和检索分数；
- `RagDocument` 与 `StoredChunk` 对外兼容既有字典输入，内部统一暴露类型化核心字段；
- Chroma、Redis、BM25、混合召回、精排和结果构建不再各自解释核心 metadata 裸键；
- 旧 Chroma/Redis 数据仍按原平铺格式读取，历史 `chunk_id` 来源推导集中为单点兼容逻辑。
- RAG 全局服务 getter 替换为实例级 `RagRuntime`，应用 Runtime、MCP、评测和脚本显式持有各自对象图；
- 预热状态机迁入 `rag/operations/`，Runtime 只负责组装、触发与关闭；
- Agent 与 MCP RAG Adapter 改为显式服务提供者注入，不再执行期定位全局容器；
- Tavily Client、重试参数、搜索数量及 Artifact 输出目录均由应用 Runtime 传入工具工厂；
- Embedding Provider 由 Runtime 创建，查询、语义缓存与建库 Writer 共享同一实例；
- RAG Redis 连接池、Embedding/Reranker 引用、Chroma Wrapper 与 BM25 后台执行器纳入实例生命周期并在关闭时释放。

### 2026-09-06：RAG 内部类型与召回组件边界收敛

- 新增 `RagDocument`，缓存、召回、精排与向量写入端口不再传递 `list[Any]` 或 LangChain `Document`；
- 文档解析 worker 显式注入 `DocumentParsingService`，移除 ingestion 对 runtime 容器的反向调用；
- 来源标识规则提升到 RAG 共享层，解析基础设施不再依赖 ingestion 包；
- Reranker 的设备和推理并发由 Runtime 注入，基础设施不再依赖 runtime；
- 混合召回拆为 Chroma 候选源、BM25 适配器、Redis 索引仓库、纯 RRF 策略和组合适配器。

### 2026-09-03：Redis 与 PostgreSQL 职责收敛

- Redis Stack 改为固定版本的官方 Redis 服务端镜像，移除 RedisInsight，端口仅绑定本机；
- Redis 只承担 RAG/BM25 缓存、API 限流和 token 预算，Agent 会话统一由 PostgreSQL 承担；
- 语义缓存由 `pickle` 改为受限 JSON 与原始 `float32` 向量字节，并使用 `rag:v2:*` 键空间隔离旧缓存；
- 限流与预算的 Redis key 改用不可逆摘要，不再暴露 API Key 前缀；
- PostgreSQL Checkpointer 补齐依赖、显式连接池生命周期、`dict_row` 和安全反序列化设置；
- `data_sql/financials.db` 暂作为旧实验数据源保留，不与会话迁移混在一起。

### 2026-09-02：Docling 按需解析与依赖单一来源

- 新增 Docling Serve 异步解析适配器、本地 PDF 解析适配器和 `auto/local/docling` 路由；
- 解析结果统一为 RAG 公共契约，保留 `source_file`/`source_page`/`source_pages`/`chunk_id` 溯源信息；
- 自动路由增加图片面积、表格倾向、多栏倾向和本地解析质量门槛，超过 200 页的 PDF 直接排除；
- 新增 `DocumentParsingService` 唯一入口，普通文档、结构化表格、PDF 路由各自独立；OCR 策略明确为 `auto/force/disabled`；
- 删除旧 `loaders.py`/`chunker.py` 分派链，建库代码只调用统一解析服务，跨端口文档统一为 `RagDocument`；
- 将 PDF、混合召回、精排、语义缓存和 Chroma 建库实现从 `rag/` 根目录迁至 `rag/infrastructure/`，根目录只保留公共入口与契约；
- 将 `infrastructure/` 内部脚本继续按 `parsing/retrieval/cache/storage` 分包，应用层只依赖 Port，具体实现仅由 Runtime 容器显式组装；
- Compose 新增只在 `ingestion` profile 下启动的 CPU 单 Worker Docling 服务；
- `pyproject.toml` 成为唯一依赖源，删除旧 `requirements.txt` 和过期锁文件，Docker 构建也直接安装项目元数据。

### 2026-08-28：RAG 模块边界与统一服务

- 新增 RAG 公共契约、端口、在线查询服务、建库服务、基础设施适配器和运行时容器；
- `query_internal_knowledge` 与 `query_financial_reports` 改为薄适配器，共用唯一 `RetrievalService`；
- 建库成功后的 BM25/语义缓存失效改由 `IngestionService` 统一调度；
- 将预热状态机从 MCP 层收回 RAG 模块（现位于 `react_agent/rag/operations/warmup.py`），删除 MCP 专属旧链；
- `react_agent` 包入口改为懒加载 graph，导入独立 RAG 契约不再隐式加载 LangGraph。
- 评测、Streamlit、MCP 健康检查、财务抽取和数据集生成均改走 RAG 公共服务；Chroma 原生客户端收口到基础设施适配器；
- 根目录 `check_tables.py` 缩减为 `IngestionService` 兼容入口，移除重复建库与直接缓存失效链路。

### 2026-07-03：MCP Server warmup 精简与 tracing 关闭

- 将 MCP 工具拆分到 `react_agent/mcp_server/`，包含 `app.py`、`responses.py`、`info_tools.py`、`health_tools.py`、`warmup_tools.py`、`rag_tools.py`；预热运行时现已迁移到 RAG 模块；
- 改造 `mcp_rag_server.py` 为 stdio 薄启动入口：路径锚定、Windows UTF-8 流修复、`.env` 加载、关闭 tracing、stderr + file 日志，不在握手阶段预热重资源；
- 定位首次慢 query 的关键根因为 LangSmith / LangChain tracing 触发 runtime metadata 采集和 `git describe` 长尾；
- 删除完整 RAG pipeline warmup，保留 `start_rag_singleton_warmup` / `get_rag_singleton_warmup_status`，现由 `RagAdminService` 统一加载两个进程内模型单例；
- 默认只注册 `server_info` 与 `query_financial_reports`，`check_knowledge_base`、`start_rag_singleton_warmup`、`get_rag_singleton_warmup_status` 仅在 `MCP_EXPOSE_ADMIN_TOOLS=1` 时暴露；
- `server_info` 默认只返回普通业务能力，不暴露 hidden admin tools，降低 Claude Code 等 Agent 的信息负担。

### 2026-06-28：工具调用协议健壮性

- 新增 `_ai_tool_call_ids()`，统一检测 `.tool_calls` 与 `.invalid_tool_calls`；
- 修复 invalid tool call 悬空导致的多轮 400 死锁；
- 入口净化、路由、异常兜底统一使用同一检测口径；
- 新增连续 HumanMessage 合并逻辑，减少重试堆积造成的角色交替问题。

### 2026-06-26：文档生成层

- 新增 `docx_tool`，支持 Word 研报生成；
- 新增 `md_tool`，支持 Markdown 输出；
- 新增 `_doc_common.py`，复用文档工具公共逻辑；
- `pyproject.toml` 新增 `python-docx`。

### 2026-06：FastAPI serving 生产化改造

- 新增 `/api/v1` 路由；
- 实现 token 级 SSE、断连取消、统一错误、request_id、鉴权、限流、预算、Prometheus；
- 新增 API 回归测试，使用 FakeAgent + fakeredis 避免真实依赖。

---

## 13. License

MIT License
