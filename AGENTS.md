# AGENTS.md

本文件适用于当前目录及其所有子目录，用于约束在本项目中工作的自动化编码 Agent。

## 项目概览

本项目是面向金融研报场景的 ReAct-RAG Agent，主要技术栈为 Python、LangGraph、LangChain、FastAPI、Streamlit、Chroma 和 Redis。系统支持私有知识库检索、来源页码追溯、联网搜索、文档/表格生成、持久化会话以及 MCP stdio 接入。

主要目录与入口：

- `react_agent/agent/`：Agent 状态、节点、路由、工具策略、用量计算、图编译、执行上下文与对话用例。
- `react_agent/agent/context_management/`：从完整 Agent State 构造单次模型输入，管理轮次分段、Token 预算、工具协议修复、证据索引与后续压缩契约。
- `react_agent/tooling/`：Agent 与工具适配器共享的 ToolResult 信封和重试执行契约。
- `react_agent/models/`：LLM Provider 解析、创建与缓存；由 Runtime、评测和显式脚本消费。
- `react_agent/infrastructure/`：Redis 等跨用例共享的技术资源适配器，不得反向依赖 Agent。
- `react_agent/observability/`：应用会话用量记录与展示；知识库建库计时归 `knowledge/ingestion/observability.py`。
- `react_agent/conversations/`：会话契约、管理用例、Repository Port 与 Checkpointer 基础设施。
- `react_agent/runtime/`：选择并注入 LLM、Agent Tools、Conversation 与共享 Checkpointer，管理应用实例生命周期。
- `react_agent/configuration/`：应用级配置模型、环境/YAML 加载与非敏感默认配置文件；Agent 编排代码不得放入此目录。
- `knowledge/`：统一知识库命名空间；根层只保留跨模块公共契约、对外能力接口、公共设施和显式配置。在线 RAG 与建库通过独立 Runtime Port 暴露，仅在 Server 组合根汇合。
- `knowledge/foundation/`：只容纳 RAG 与建库确实共用、且不依赖两侧内部类型的公共设施；不得放入单侧模块专属的端口或适配器。
- `knowledge/rag/`：自治管理在线查询、候选召回、管理、离线读取、评测、预热、模块配置、内部契约/端口、对象图，以及 Redis、语义缓存、BM25、Chroma Retriever/只读管理、Reranker 等具体实现；不得依赖 `knowledge.ingestion` 或顶层业务基础设施实现。
- `knowledge/ingestion/`：文档解析、PDF 路由和增量建库用例，并自治管理模块配置、来源标识、内部契约、端口、观测、对象图与解析/写入适配器；不得依赖 `knowledge.rag`。
- `knowledge/runtime/`：Knowledge Server 顶层组合根、共享资源/配置生命周期和跨模块事件连接；顶层 Runtime 只暴露独立的 `rag_runtime`、`ingestion_runtime` 视图和统一关闭能力，不得转发模块业务方法、直接组装内部组件或把完整顶层配置传给任一模块 Runtime。
- `knowledge/client/`：独立 Knowledge Service HTTP 客户端，以及互不暴露能力的远程 RAG/Ingestion Runtime；只依赖公共契约，不得反向依赖 Agent、Server 或 MCP。
- `knowledge/transport/`：Knowledge HTTP 请求/响应 Schema 与领域对象 Codec 的中立唯一实现；不得依赖 FastAPI、httpx、Client、Server 或 RAG/Ingestion 内部类型。
- `knowledge/server/`：Knowledge Service ASGI 入口、服务端配置和环境组合根；请求/响应模型统一使用 `knowledge.transport`，且该进程是唯一允许打开本地知识库资源的在线进程，不得依赖 `react_agent`。
- `react_agent/tools/`：RAG、搜索、Excel、Word、Markdown 和 SQL 协议适配器；Agent 实际工具集合由 `react_agent/runtime/container.py` 组装。
- `mcp_service/`：独立 MCP Server 协议适配器、进程生命周期和工具注册，只依赖 Knowledge 公共契约/客户端；不得依赖 `react_agent`。`mcp_service.main:main` 是正式命令入口，根目录 `mcp_rag_server.py` 仅作兼容转发。
- `api/`：FastAPI 服务、鉴权、限流、指标、错误处理和版本化路由。
- `tests/api/`：使用 FakeAgent 与 fakeredis 的离线 API 回归测试。
- `tests/architecture/`：使用 Python AST 校验 Knowledge 模块的静态导入方向；新增或调整跨模块依赖时必须同步审查规则，不得通过删除规则或扩大通配允许范围掩盖越界依赖。
- `scripts/streamlit_app.py`：Streamlit 应用入口，由 `scripts/` 目录统一管理，不是单元测试。
- `eval/`：RAGAS 数据集生成与评测，可能访问真实模型、知识库和外部服务。
- `scripts/`：Streamlit 应用入口、PDF 质检和财务数据抽取脚本。
- `react_agent/configuration/config.yaml`、`react_agent/configuration/deepseek_pricing.yaml`：非敏感工具默认值与价格卡；根目录 `.env`：本地运行环境配置。

## 环境与命令约定

- 始终使用 Conda 环境 `new_agent`，不要使用系统 Python 或自行创建其他虚拟环境。
- 在 PowerShell 中优先使用 `conda run -n new_agent ...`，这样命令不依赖当前 shell 是否已执行 `conda activate`。
- 所有命令默认从本文件所在目录执行。
- Docker 镜像使用 Python 3.10；本地环境可能更高，因此新增代码必须保持 Python 3.10+ 兼容，不使用仅在 3.11+ 可用的语法或标准库 API。
- 首次安装或依赖变化后执行：

  ```powershell
  conda run -n new_agent python -m pip install -e . --group dev
  ```

- 不要在未经请求时升级、重新锁定或批量整理依赖；`pyproject.toml` 是当前运行和测试依赖的唯一依据。

## 配置与敏感信息

- 从 `.env.example` 复制本地 `.env`，按需配置 DeepSeek/OpenAI、Tavily、Redis、PostgreSQL 等服务。
- 不读取、输出、记录或提交 `.env` 中的真实密钥；日志、异常消息、测试夹具和示例中也不得泄露凭据。
- Agent 配置入口是 `react_agent/configuration/settings.py`；工具配置优先级为：Conda/系统环境变量 > `.env` > `react_agent/configuration/config.yaml` > 代码默认值。
- Knowledge Server 的 HTTP 边界配置位于 `knowledge/server/settings.py`，本地 Runtime 环境配置投影位于 `knowledge/server/runtime.py`；不得借用 Agent settings。
- RAG 专属配置定义在 `knowledge/rag/config.py`，建库专属配置定义在 `knowledge/ingestion/config.py`；`knowledge/runtime/config.py` 只保留共享资源配置和顶层聚合，跨模块路径或资源应由组合根以最小参数注入。
- 新增配置时同步更新所属边界的配置模型和 `.env.example`；只有 Agent 工具层的非敏感默认值适合放入 `react_agent/configuration/config.yaml`。
- 测试应使用假密钥和 mock/fake 依赖。不要为普通回归测试调用真实 LLM、Tavily、Redis、PostgreSQL 或外网服务。

## 实现约定

- 保持改动聚焦；先阅读受影响模块、调用方和相关测试，不顺手重构无关代码。
- 延续现有风格：4 空格缩进、类型注解、简洁 docstring、模块级 `logging`。不要用 `print` 代替服务日志；配置诊断脚本除外。
- 异步调用链中不得直接执行耗时同步 I/O 或模型加载；使用已有异步 API，必要时通过 `asyncio.to_thread` 隔离阻塞工作。
- `AgentService` 的调用必须传入明确的 `thread_id`。API 层需继续保持按 API Key/用户命名空间隔离，避免共享默认线程和 IDOR。
- 修改 Agent 工具时，同时检查工具 schema、Runtime 工具注入、提示词/路由、错误返回和测试；不要仅新增实现文件而忘记组装。
- RAG 结果必须尽量保留 `source_file`、`source_page`、`chunk_id` 等溯源元数据。调整检索、分块或缓存逻辑时，不得静默破坏引用链。
- 不要在导入模块时加载大型 embedding/reranker、扫描语料或建立真实网络连接；重资源保持懒加载和进程内单例。
- MCP 使用 stdio 协议：`stdout` 只能承载协议消息，日志必须写入 `stderr` 或日志文件。不要在 `mcp_rag_server.py` 启动阶段预热模型或执行真实查询。
- FastAPI 变更需保持现有契约：`/api/v1/*` 版本化路由、problem+json 错误体、`request_id`、鉴权豁免、限流/预算、SSE 事件顺序和断连取消行为。
- 公共 API 或行为变化应同步更新 `README.md`、`.env.example` 或相应工程说明。

## 测试与验证

优先运行与改动最相关的最小测试，再运行离线 API 回归测试：

```powershell
# 单文件或单用例
conda run -n new_agent python -m pytest tests/api/test_stream.py -q
conda run -n new_agent python -m pytest tests/api/test_stream.py::test_stream_event_sequence -q

# 完整离线 API 回归
conda run -n new_agent python -m pytest tests/api -q
```

注意事项：

- 如果环境提示缺少 `pytest` 或其他依赖，先说明并按 `pyproject.toml` 安装，不要改用系统 Python 绕过。
- `scripts/streamlit_app.py` 是 Streamlit UI 入口，不应由 pytest 导入或收集。
- `eval/run_eval.py`、真实 RAG 查询、MCP Inspector、Streamlit 和服务启动都属于集成/手工验证，可能耗时、访问外部服务或产生费用；仅在任务需要且配置齐备时运行，并在结果中明确说明。
- 修改 Python 代码后，至少对改动文件做语法检查；可使用：

  ```powershell
  conda run -n new_agent python -m compileall <受影响的包或文件>
  ```

- 项目依赖包含 Ruff；安装依赖后可对受影响文件执行：

  ```powershell
  conda run -n new_agent python -m ruff check <受影响的文件>
  ```

常用手工启动命令：

```powershell
# FastAPI
conda run -n new_agent python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Streamlit
conda run -n new_agent python -m streamlit run scripts/streamlit_app.py

# MCP stdio Server
conda run -n new_agent python mcp_rag_server.py

# RAGAS 评测（仅在明确需要真实评测时）
conda run -n new_agent python eval/run_eval.py
```

## 本地数据与生成物

以下内容属于本地运行状态、私有数据或生成物，除非任务明确要求，否则不要编辑、清理、提交或用测试覆盖：

- `.env`、`logs/`、`*.log`、`ingestion_metrics.jsonl`
- `chroma_db/`、`redis-data/`、`data_sql/`
- `*.sqlite3*`、`*.db`
- `outputs/`、`eval/results/`、`eval/dataset/`
- `__pycache__/`、`.pytest_cache/`、IDE 配置

测试和脚本产生的新临时文件应写入 pytest 临时目录或系统临时目录；不要污染上述持久化目录。涉及数据库重建、向量库重建、会话删除或覆盖输出文件时，先确认准确目标和影响范围。

## 交付要求

- 最终说明应包含：修改了什么、涉及哪些文件、执行了哪些验证及其结果。
- 若验证未运行或失败，明确给出原因，不得声称通过。
- 不要为了让测试通过而弱化鉴权、隔离、限流、来源追溯、错误处理或 mock 边界。
