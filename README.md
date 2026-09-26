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
| **Skill 工作流** | 内置行业/市场研究 Skill；按用户轮确定性选择、按需加载并瞬态注入模型上下文，不扩大 Runtime 工具权限 |
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
  → 选择并加载匹配的内置 Skill（未匹配则保持普通 ReAct）
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
| `react_agent/agent/service.py` | 对外 Agent 对话用例；校验 `thread_id`，封装 invoke / stream |
| `react_agent/agent/workflow/` | LangGraph 拓扑、条件路由与节点适配层；`nodes/` 按生命周期、模型和工具节点拆分 |
| `react_agent/agent/contracts/` | 图状态与运行时依赖契约，不包含模型或工具实现 |
| `react_agent/agent/config.py` | 管理图内行为的 `AgentContext`，以及环境变量覆盖与类型转换 |
| `react_agent/agent/context_management/` | 从完整 State 构造单次模型上下文；负责轮次分段、Token 预算、协议修复、证据账本与旧轮次摘要 |
| `react_agent/agent/policies.py` | 模型轮次、工具批次、重试和递归安全预算的纯策略判断 |
| `react_agent/agent/model_execution.py` | 模型绑定、调用、响应与用量状态更新；不再决定历史和证据如何进入上下文 |
| `react_agent/agent/prompts.py` | 默认系统提示词，以及失败恢复/主动收口的瞬态控制指令 |
| `react_agent/agent/tool_flow/` | Agent 内部工具调用解析、执行、结果归一化、限幅与预算统计 |
| `react_agent/skills/` | 内置专业工作流的契约、缓存加载、确定性选择与 `SKILL.md` 包资源 |
| `react_agent/agent/time.py` | Agent 使用的时区时间表示规则 |
| `react_agent/agent/__init__.py` | 通过懒加载导出稳定公共 API；不提供旧模块兼容门面 |
| `react_agent/configuration/` | 应用级配置模型、环境/YAML 加载及非敏感工具默认值、价格卡；不负责 Agent 图内行为 |
| `react_agent/tooling/` | Agent 与工具适配器共享的 ToolResult 信封和重试执行契约 |
| `react_agent/models/` | LLM Provider 解析、创建与缓存 |
| `react_agent/metering/` | 统一模型 token 归一化、人民币计价及累计状态的轮次用量差值；不决定 Agent 对话边界 |
| `react_agent/infrastructure/` | Redis 等跨用例共享的技术资源适配器 |
| `react_agent/observability/` | 已计算用量的结构化日志、会话汇总与界面展示；不负责重新计量或计价 |
| `react_agent/conversations/contracts.py` | 类型化消息视图、删除结果及持久化配置／错误契约 |
| `react_agent/conversations/ports.py` | `ConversationRepositoryPort` 出站端口 |
| `react_agent/conversations/service.py` | 历史读取与会话删除用例 |
| `react_agent/conversations/infrastructure/` | LangGraph Checkpointer Adapter 及 PostgreSQL / SQLite / Memory 工厂 |
| `react_agent/runtime/container.py` | 选择 LLM Adapter 与 Agent 工具集，创建共享 Checkpointer，完成依赖注入并管理实例生命周期 |

全局 Agent 配置入口是 `react_agent/configuration/settings.py`。工具默认值和价格卡随
`react_agent.configuration` 一起打包；`.env` / `.env.example` 留在应用根目录，
供本地运行和 Docker Compose 注入环境变量。`pyproject.toml`、Compose 文件与
`pytest.ini` 保留在根目录，供相应构建、部署和测试工具发现。
Knowledge Server 的 HTTP 设置和本地 Runtime 配置分别由
`knowledge/server/settings.py` 和 `knowledge/server/runtime.py` 从环境变量组装，
不依赖 Agent 配置对象。Runtime 数字、布尔值、枚举和取值范围会在服务启动时
统一校验：环境变量缺失时使用默认值，显式配置但格式错误或越界时直接启动失败。
远程 Knowledge Service 的请求超时由 `KNOWLEDGE_SERVICE_TIMEOUT` 控制。

`AgentService` 不提供历史读取或删除接口；FastAPI Chat 路由只注入
`AgentService`，Sessions 路由只注入 `ConversationService`。两者不互相依赖，
由 Composition Root 共享同一个 Checkpointer。`AgentContext` 只管理提示词、
能力开关、业务预算、上下文控制和图安全熔断。模型选择/推理参数由
`LLMConfig` 管理，具体工具参数由各工具配置管理；模型提供者、计价策略与工具
集合通过 `AgentDependencies` 显式传入，Agent 节点不再自行导入模型工厂或
全局工具表。会话标识由调用方提供；`ConversationPersistenceConfig` 携带
后端、数据库路径和 PostgreSQL 连接池参数，Checkpointer 工厂不再读取全局
PostgreSQL 设置。FastAPI 和 Streamlit 都通过
`create_application_services(...)` 完成组装，并把返回的服务实例传给
`close_application_services(...)` 精确释放本实例资源。健康检查展示的是实际
生效后端，因此 SQLite/PostgreSQL 降级到 Memory 时不会继续误报原配置值。
FastAPI 与 Streamlit 均通过 `load_conversation_persistence_config()` 读取
`CHECKPOINT_BACKEND`、`CHECKPOINT_DB_PATH` 和 PostgreSQL 参数；未配置 backend
时默认 SQLite，未知 backend 在启动时明确报错。
当显式配置 `CHECKPOINT_BACKEND=postgres` 时，依赖缺失、连接串缺失、连接超时
或建表失败都会终止应用启动，不会再静默降级到易失的 MemorySaver。SQLite
初始化失败时仍保留面向本地开发的 MemorySaver 降级能力。
`/api/v1/health/ready` 比较请求与实际启用的后端：例如请求 SQLite 却降级
MemorySaver 时返回 503；显式 Memory 模式则正常就绪。原有 `/api/v1/health`
和 `/health` 保留存活检查语义，不因后端不一致而返回 503。
会话 history 返回的是**最新 Checkpoint 中的消息列表**，不是历次 Checkpoint
版本的审计日志；仓储将 LangChain 消息投影为类型化的 `ConversationMessage`，
API 再负责分页和响应格式。删除只使用 Checkpointer 的公开 `adelete_thread()`；
后端不支持时返回 501，不再直接操作存储内部结构。

Agent 包内部由 `service.py` 调用 `workflow/`；节点使用 `policies.py`、
`model_execution.py`、`context_management/` 和 `tool_flow/`，共享契约位于
`contracts/`，图内行为配置位于 `config.py`。`model_execution.py`
只消费 `context_management` 构造完成的 `ModelContext`；历史裁剪、工具协议修复
和证据索引均不再从模型调用层或 `tool_flow` 暴露兼容入口。模型、工具和
持久化的具体 Adapter 只能由 `react_agent/runtime/` 注入，禁止反向导入到
Agent 包。外部入口统一从 `react_agent.agent` 导入公共对象。LangGraph 显式
节点名称保持不变。历史 Token 数使用离线、确定性的 UTF-8 近似估算；
一次上下文构造只估算每条消息一次，不会在调用模型前下载 tokenizer 资源。
`MAX_INPUT_TOKENS` 是单次调用的本地输入估算上限，包含最终系统提示词、
瞬态指令、绑定工具 Schema、当前轮、证据和已选历史；`MAX_HISTORY_TOKENS`
只为会话消息分配子预算，不限制固定提示词和工具 Schema；当前轮即使超过该
子预算也保留给总预算闸门判断。若配置模型实际窗口
`LLM_CONTEXT_WINDOW_TOKENS`，还会从中
扣除 `LLM_MAX_TOKENS` 输出预留和 `CONTEXT_SAFETY_MARGIN_TOKENS` 安全余量，
取两种输入上限的较小值。未配置模型窗口时只执行本地策略上限，不宣称与
Provider 窗口一致。每次调用生成不进入 Checkpoint 的 `BudgetReport`；
默认历史/总输入预算分别为 80,000/96,000 估算 Token。
模型输出限制按 Provider 映射：OpenAI 使用 `max_completion_tokens`，DeepSeek
通过兼容接口发送 `max_tokens`，Anthropic 使用原生 `max_tokens`。主调用会记录
配置上限、实际输入/输出、推理 Token 和结束原因；若 Provider 返回 `length`
或 `max_tokens`，Agent 不会执行可能不完整的工具调用，并显式提示结果被截断。
`MAX_HISTORY_TOKENS` 按已完成的完整用户轮次裁剪，近期历史保持连续；
当前轮不会被历史子预算拆开。总预算超限时，先尝试投影旧工具正文以保留轮次，
仍超限才整轮移除；只剩当前轮时，依次缩短证据索引摘录、工具正文及索引条数。
投影只作用于本次模型输入，不覆盖 Checkpoint；RAG 可见结果保留尽可能多的
`source`、`page`、`chunk_id`，省略处显式标记，`BudgetReport` 记录原因和规模。
调用前按 `tool_call_id` 校验并行工具调用与结果的配对，正文投影不删除
ToolMessage 协议外壳。固定内容、当前问题或当前轮仍超限时在调用模型前报错。
估算并非 Provider 精确 Token 计数。启用 `ENABLE_HISTORY_COMPACTION` 时，
新用户轮开始且可见历史达到子预算约 80% 后，至多用一次额外模型调用总结
游标之后的较早完整轮次；近期三轮及当前轮不压缩。模型只写定性概览，
数字、否定约束、更正原文及 RAG 来源位置由程序按句段确定性附加；
调用前先排除无法达到最小节省量的批次。不合格或不节省空间的摘要不推进
游标，并在候选来源新增 `HISTORY_COMPACTION_RETRY_NEW_TOKENS` 之前不再付费
重试。摘要输出受 `HISTORY_COMPACTION_MAX_OUTPUT_TOKENS` 限制，日志区分超长、
不安全概览、无 Token 节省等原因。原始 `State.messages` 保留在 Checkpoint，摘要以独立
字段存储，只在单次模型输入中替代其覆盖的旧轮次；摘要调用计入模型用量。
API 非流式配额也包含摘要调用；流式接口只向用户推送主回答的文本 Token，
摘要调用仅产生用量事件。

Agent 的正常终止由 `MAX_MODEL_ROUNDS`、`MAX_TOOL_BATCHES` 和
`MAX_TOOL_RETRIES` 控制；`RECURSION_LIMIT` 只作为图异常循环的最后熔断器，
并在 `AgentContext` 初始化时校验其足以覆盖所配置的业务预算。若模型在预算
耗尽时已经生成工具调用，图会先写入 `TOOL_BUDGET_EXHAUSTED` ToolMessage
闭合调用协议，再进入不绑定工具的 `finalize_model`，避免持久化悬空调用。
`RAG_CALL_LIMIT` 限制当前用户轮中被 Agent 接受执行的 RAG 工具调用：
成功、未命中和执行失败均占用一次，在执行前因配额不足被拦截的调用
不占用。同一模型批次生成多个 RAG 调用时，工具执行层只放行剩余
配额，并为其余调用写入 `RAG_CALL_BUDGET_EXHAUSTED` ToolMessage 后主动收口。
图运行期间还会读取 LangGraph 注入的 `RemainingSteps`：若剩余步骤不足以完成
工具执行、结果处理和最终总结，主模型在当前节点禁用工具，依据已有证据回答或说明不足；工具
结果处理后若不足以继续循环，则提前进入 `finalize_model`。意外耗尽图步骤时，
API 返回明确的执行步数错误；流式接口发出 `error` 后仍按约定发出 `done`。
RAG 工具结果按整个 JSON 输出预算限幅：优先保留各片段的来源、页码与
`chunk_id`，再分配可见正文；无法容纳任何证据时返回明确的预算错误，
不伪装成检索未命中，并基于此前可见证据主动收口。Agent 在本轮内按
`chunk_id` 合并模型可见片段的有界证据索引，最终总结会收到这份索引。
索引只保留短的原文摘录和溯源，
不代表片段已经核验，也不能保证截断部分没有关键事实。每次成功 RAG 还会
独立维护不含正文的跨轮来源索引，保存查询、来源文件、页码和 `chunk_id`；
该索引不依赖历史摘要是否被接受，因此原工具轮退出消息窗口后仍可用于来源
回顾。旧 Checkpoint 会在下一轮增量补建该索引。工具调用轨迹只附带有界
`sources` 列表，不复制检索正文。

Streamlit 登录后从共享 Checkpoint 恢复用户消息和最终助手消息，过滤内部
工具调用规划与 ToolMessage。回答期间通过单次 `astream_events`
执行获取真实图进度，只向 `st.status` 投影“分析问题、调用工具、
分析结果、整理回答”等脱敏状态；不展示提示词、工具参数或检索正文。
后台事件循环通过线程安全队列通知 Streamlit 主线程，不从后台线程调用
UI API。新回答在展示动画开始前先写入页面会话状态，回答期间
输入框保持禁用；展示采用快速分块而非逐字符延迟，因此中途 rerun
不会吞掉已经生成的完整回答。

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
  → 公共 KnowledgeDocument 边界转换
  → BM25 Top-10 + Vector Top-10
  → RRF 融合到最多 20 个候选
  → Cross-Encoder 精排
  → 按配置返回 Top-N（默认 5）给 Agent
```

RAG 采用模块自治的端口与适配器分层：`knowledge/contracts.py` 和
`knowledge/runtime_ports.py` 只保存跨边界公共数据与对外能力接口；RAG 内部请求、
候选轨迹和依赖端口归 `knowledge/rag/contracts.py`、`knowledge/rag/ports.py` 管理。
`knowledge/rag/query/RetrievalService` 是缓存、召回、精排和来源归一化的在线编排入口，
`knowledge/rag/retrieval/HybridRetrievalService` 负责 BM25/向量并行、RRF 融合、过滤和
失败降级。管理服务归 `knowledge/rag/admin/`，Redis 生命周期、语义缓存、BM25、
Chroma Retriever、只读管理适配器和 Reranker 等具体实现均位于
`knowledge/rag/infrastructure/`。`knowledge/ingestion/IngestionService` 负责增量建库后
发布索引变更通知，不直接依赖 RAG；两者只在实例级 `KnowledgeRuntime` 中汇合。
Agent 与 MCP 的 RAG Adapter 都在注册时接收 `RetrievalService` 提供者，只负责协议转换，
执行过程中不再访问 RAG Service Locator。

建库模块同样自治：解析请求、解析结果与 OCR 策略位于
`knowledge/ingestion/contracts.py`，解析器、清单、页数检查、预检、索引写入与变更通知
端口位于 `knowledge/ingestion/ports.py`；PDF、Docling、Office/文本、表格解析器和
Chroma/SQLite 写入协调实现均收归 `knowledge/ingestion/infrastructure/`，建库观测归
`knowledge/ingestion/observability.py`，来源路径归一化归 `knowledge/ingestion/source.py`。
跨建库与检索边界只传递中立的
`KnowledgeDocument`，共享设施不得引用两侧的内部契约或实现。

本地对象图也按模块隔离：`knowledge/rag/runtime.py` 只组装查询、评测、管理、预热和
RAG 私有资源，`knowledge/ingestion/runtime.py` 只组装解析与建库依赖；
`knowledge/runtime/resources.py` 惰性持有双方确实共用的 Embedding、Chunk Store 和
写锁。顶层 `knowledge/runtime/container.py` 仅连接两个模块、桥接建库提交后的 RAG
缓存失效事件并管理关闭顺序。Knowledge Server 路由分别接收互不暴露能力的 RAG 和
Ingestion Runtime 视图。顶层 `KnowledgeRuntime` 本身不再转发查询、管理、预热或建库
方法，`KnowledgeServerRuntimePort` 也只公开两个模块视图与统一 `close()`；Server 不再
接受缺少独立视图的混合 Runtime 兼容对象。

配置边界与对象图边界保持一致：`knowledge/rag/config.py` 独占检索预算、Redis 和
Reranker 配置，`knowledge/ingestion/config.py` 独占 Docling、批处理策略和指纹清单路径；
`knowledge/runtime/config.py` 只定义共享 Embedding/设备、共享存储位置，并在 Server
组合根聚合两侧配置。模块 Runtime 不接收完整 `KnowledgeRuntimeConfig`，Chroma 路径、
共享设备和资源提供者均由顶层按最小参数注入，因此任一模块都无法读取另一侧的配置细节。

本地评测数据集需要直接读取指定 Chroma 时，使用 `knowledge/rag/offline.py` 的最小
Chunk 读取入口。该入口只创建 RAG 自己的只读适配器，不再创建完整
`KnowledgeRuntime`，因此不会附带 Redis、Embedding、建库对象图或无法关闭的隐藏资源。

模块依赖方向由 `tests/architecture/import_graph.py` 使用 Python AST 提取真实导入，
并由 `tests/architecture/test_knowledge_dependencies.py` 校验。守卫覆盖 RAG、Ingestion、
Foundation、Transport、Client、Runtime 与 Server 的禁止依赖，以及 Runtime/Server
组合根允许接触的模块边界；注释和字符串不会被误判，检查仅在测试阶段运行。

HTTP 边界使用同一套中立传输契约：`knowledge/transport/schemas.py` 定义请求与响应
Schema，`knowledge/transport/codecs.py` 负责领域对象与 JSON Payload 的双向转换。
Server 不再手工 `asdict`，Client 也不再逐字段重建 `RetrievedChunk`、评测轨迹、
健康状态、分页 Chunk 或建库报告；新增字段、默认值和可空值由同一份 Schema 约束。
传输层不依赖 FastAPI、httpx 或 RAG/Ingestion 的内部实现。

`knowledge/client/` 是独立的远程访问边界，包含 HTTP 协议客户端、
互不包含的 `RemoteRagRuntime` / `RemoteIngestionRuntime`、显式
`KnowledgeClientConfig` 和环境配置适配器。Agent、MCP 和评测只依赖在线 RAG
Runtime，建库脚本只依赖 Ingestion Runtime。旧 Agent 内嵌 RAG 模块及兼容
导入路径已删除，不再提供隐式本地 Runtime 或远程客户端入口。

`knowledge/server/` 是唯一允许直接打开 `CHROMA_DB_PATH` 的在线进程边界，提供
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
`ParseResult`，应用端口统一传递中立的 `KnowledgeDocument`；LangChain `Document`
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

本项目提供 stdio 模式 MCP Server，用于将金融研报 RAG 能力接入 Claude Desktop、Cursor、Claude Code、MCP Inspector 等支持 MCP 的客户端。

| 文件 | 职责 |
|---|---|
| `mcp_rag_server.py` | 无副作用兼容入口，仅调用 `mcp_service.main:main` |
| `mcp_service/main.py` | Windows UTF-8、`.env`、tracing、日志和 stdio 进程初始化；支持 `MCP_LOG_PATH` |
| `mcp_service/bootstrap.py` | 显式持有 FastMCP 与远程 RAG Runtime，保证组装失败或 stdio 退出后释放连接 |
| `mcp_service/app.py` | MCP Server 创建与工具注册入口；默认注册 info + rag，`MCP_EXPOSE_ADMIN_TOOLS=1` 时注册 health + warmup |
| `mcp_service/info_tools.py` | 注册 `server_info`，返回当前真实可用工具与能力边界；默认不暴露 admin tools |
| `mcp_service/health_tools.py` | 注册 `check_knowledge_base`，轻量检查知识库与向量库可用性，可选 admin 工具 |
| `knowledge/rag/operations/warmup.py` | 实例级 RAG 预热状态机，拥有顺序、幂等、失败、等待与取消规则 |
| `mcp_service/warmup_tools.py` | 注册 `start_rag_singleton_warmup` / `get_rag_singleton_warmup_status`，可选 admin 工具 |
| `mcp_service/rag_tools.py` | 注册 `query_financial_reports`，对外暴露金融研报 RAG 查询 |
| `mcp_service/responses.py` | 统一 MCP 工具返回结构，如 `mcp_ok` / `mcp_err` |
| `mcp_service/observability.py` | 为每次工具调用记录 request_id、阶段、状态、耗时和错误类型 |

MCP 启动方式：

```bash
conda run -n new_agent python mcp_rag_server.py
# 安装项目后也可以直接运行：financial-rag-mcp
```

使用 MCP Inspector 测试：

```bash
npx -y @modelcontextprotocol/inspector -- conda run -n new_agent python mcp_rag_server.py
```

安装项目后，可将正式命令注册到 Codex：

```bash
codex mcp add financial-rag --env MCP_EXPOSE_ADMIN_TOOLS=0 -- financial-rag-mcp
```

MCP 启动入口遵循“薄启动”原则：启动阶段不预热 embedding、reranker、Chroma、Redis 等重资源，避免 stdio 握手阶段阻塞。普通业务查询只需调用 `query_financial_reports`；首次查询会自动触发 retriever / reranker 单例预热并进行有限等待。诊断或显式预热场景可设置 `MCP_EXPOSE_ADMIN_TOOLS=1`，再调用 `start_rag_singleton_warmup` / `get_rag_singleton_warmup_status`。
源码检出中的 MCP 入口仍以项目 `src` 为运行根目录；普通安装后的
`financial-rag-mcp` 以当前工作目录为根，避免从 `site-packages` 读取 `.env`
或写入日志。可通过启动环境中的 `MCP_ENV_FILE` 和 `MCP_LOG_PATH` 指定路径；
日志文件不可写时会降级为仅写入 `stderr`，不会阻断 stdio 握手。
MCP 启动时通过 `knowledge.client` 根据 `RAG_RUNTIME_MODE` 创建 Runtime；默认 `remote`，此时必须配置
`KNOWLEDGE_SERVICE_URL`，不会隐式打开本地 Chroma。注册函数只接收显式的 Query、
Admin 和 Operations 提供者；关闭 stdio Server 后由 `McpServiceApplication` 在
`finally` 中释放客户端连接，不依赖模块级 RAG 服务单例。每次工具调用的返回
`meta` 和结构化日志都包含 `request_id`、`stage`、`status` 与 `elapsed_ms`；失败时
另含 `error_type`，便于关联客户端报错和服务端日志。

当前 MCP 适配器使用 Python SDK v1 的 `mcp.server.fastmcp.FastMCP` API，因此依赖
明确限制为 `mcp>=1.28,<2`。SDK v2 是破坏性升级，后续迁移应单独调整 Server、
ClientSession 与传输层 API，并重新执行 stdio 子进程握手测试，不能直接解除上限。

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
非流式请求若意外触发图步数硬熔断，会从当前提问的 Checkpoint 补记已完成的
模型调用；只有匹配本次用户消息 ID 的状态才会计入，避免重复计算上一轮。
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

连接串只保存在本地 `.env`，不要提交 GitHub。`data_sql/financials.db` 是可选
Text2SQL 工具的数据源，与 Agent 会话存储相互独立。

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
`http://knowledge-service:8001` 访问知识库。当前使用宿主 bind mount；启动容器前
必须停止其他会直接打开同一 Chroma 目录的本地进程或容器，避免多进程争用。
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
HNSW 加载失败。首次启动会在后台预热，并在需要时从现有 Chroma 生成
`data/knowledge/chunks.sqlite3`。预热阶段部分并行，实际状态和耗时以
`/api/v1/runtime/warmup` 返回值为准。
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
conda run -n new_agent python -m streamlit run tests/test_agent.py
```

### 7.6 在 Conda 环境启动 FastAPI

```bash
conda run -n new_agent python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
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

reranker 返回 `top_score`，语义缓存按置信度决定是否写入，避免低相关结果固化：

| top_score | 策略 |
|---|---|
| ≥ 0.5 | 正常缓存 |
| 0 ~ 0.5 | 不缓存，避免低质结果固化 |
| 空结果 | 短 TTL 缓存，允许后续重试 |

### 9.4 工具调用协议：把消息序列合法性作为强约束

DeepSeek / LangChain 工具调用中，复杂工具参数可能进入 `.invalid_tool_calls`，如果路由只看 `.tool_calls`，就会误判“没有工具调用”，导致悬空 tool_call 被写入 checkpoint，下一轮恢复历史时触发多轮 400 死锁。

本项目统一封装 `_ai_tool_call_ids()`，同时覆盖 `.tool_calls` 与 `.invalid_tool_calls`，并在路由、入口净化、异常兜底三层使用同一检测口径。这个问题的教训是：**协议检测必须与实际发送给模型的序列化路径保持一致**。

### 9.5 FastAPI serving：从 demo 接口到可运营服务

FastAPI 服务包含以下工程护栏：

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


### 9.6 MCP stdio 与资源生命周期

`mcp_rag_server.py` 和安装命令 `financial-rag-mcp` 都只负责 stdio 初始化与
`mcp_service` 组装，启动阶段不会加载 embedding、reranker、Chroma 或 Redis。
MCP 默认使用远程 Knowledge Service；查询时由服务端按需完成 RAG 预热，连接在
stdio Server 退出后统一关闭。

普通客户端只需调用 `query_financial_reports`。诊断场景可设置
`MCP_EXPOSE_ADMIN_TOOLS=1`，使用 `check_knowledge_base`、
`start_rag_singleton_warmup` 和 `get_rag_singleton_warmup_status`。stdout 仅承载
MCP 协议消息，诊断日志写入 stderr 或 `MCP_LOG_PATH` 指定的文件。

---

## 10. 目录结构

```text
src/
├── api/                         # FastAPI 服务、鉴权、限流、指标和 v1 路由
├── knowledge/                   # 统一 Knowledge 命名空间
│   ├── contracts.py             # 跨模块文档、来源、查询、建库与健康状态契约
│   ├── runtime_ports.py         # 对 Agent、Server 与建库入口暴露的能力视图
│   ├── foundation/              # Embedding、Chunk Store 等跨模块公共设施
│   ├── rag/                     # 在线检索的配置、契约、端口、实现与本地对象图
│   ├── ingestion/               # 建库配置、契约、端口、解析/写入实现与本地对象图
│   ├── transport/               # HTTP Schema 与领域对象双向 Codec
│   ├── runtime/                 # 顶层组合根、共享资源/配置与设备策略
│   ├── client/                  # Knowledge Service HTTP 客户端与远程 Runtime
│   └── server/                  # ASGI 入口、服务端配置和环境组合根
├── react_agent/
│   ├── agent/                   # Agent 状态、工作流、策略、上下文和工具流
│   ├── skills/                  # 内置专业工作流及选择、加载契约
│   ├── conversations/           # 会话用例、端口和 Checkpointer 适配器
│   ├── runtime/                 # 应用级 Composition Root 与生命周期
│   ├── configuration/           # 应用配置、默认值和价格卡
│   ├── tools/                   # RAG、Search、Excel、Word、Markdown 和 SQL 工具
│   ├── tooling/                 # ToolResult 与重试契约
│   ├── models/                  # LLM Provider 工厂
│   ├── metering/                # Token、成本和轮次用量计算
│   ├── infrastructure/          # Agent 侧共享技术适配器
│   └── observability/           # 用量日志与界面展示
├── mcp_service/                 # 独立 MCP 协议适配器
│   ├── main.py                  # 环境、日志、stdio 初始化
│   ├── bootstrap.py             # MCP 应用组装与资源释放
│   ├── app.py                   # FastMCP 创建和工具注册
│   ├── rag_tools.py             # query_financial_reports
│   ├── health_tools.py          # 可选健康检查工具
│   ├── warmup_tools.py          # 可选预热工具
│   ├── info_tools.py            # server_info
│   ├── responses.py             # 统一工具响应结构
│   └── observability.py         # MCP 工具调用日志
├── eval/                        # RAGAS 评测和数据集脚本
├── scripts/                     # 数据检查、财务抽取和调试脚本
├── tests/                       # 离线测试与 Streamlit 应用入口
├── mcp_rag_server.py            # MCP stdio 兼容入口
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── .env.example
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

---

## 12. License

MIT License
