# =============================================================================
# react_agent/core/config.py
# 项目唯一配置入口
#
# 职责：
#   1. 定位项目根目录，加载 .env（不覆盖 Conda / 系统已有的环境变量）
#   2. 用 Pydantic 校验并暴露所有环境变量（密钥 + 运行时参数）
#   3. 加载 config.yaml，用 Pydantic 校验工具层 / 核心层参数
#   4. 对外暴露唯一实例 `settings`，任何脚本只需 `from react_agent.core.config import settings`
#
# 优先级（高 → 低）：
#   Conda / 系统环境变量 > .env 文件 > config.yaml > 代码默认值
# =============================================================================

from __future__ import annotations

import os
import sys
import yaml
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, model_validator


# ---------------------------------------------------------------------------
# 0. 定位根目录并加载 .env
#    config.py 位于 src/react_agent/core/config.py
#    所以根目录 = __file__ 向上 4 级
# ---------------------------------------------------------------------------

_CONFIG_FILE = Path(__file__).resolve()               # .../src/react_agent/core/config.py
_CORE_DIR = _CONFIG_FILE.parent                    # .../src/react_agent/core/
_REACT_DIR = _CORE_DIR.parent                       # .../src/react_agent/
_SRC_DIR = _REACT_DIR.parent                      # .../src/
_PROJECT_ROOT = _SRC_DIR.parent                        # .../react-agent-main/

_ENV_PATH = _SRC_DIR / ".env"
_YAML_PATH = _SRC_DIR / "config.yaml"  # 与当前 config.yaml 位置一致

# override=False：若 Conda / 系统中已有该变量，.env 不会覆盖它
load_dotenv(dotenv_path=_ENV_PATH, override=False)


def resolve_app_path(value: str | Path) -> str:
    """将运行时相对路径稳定地解析到源码/容器应用根目录。"""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _SRC_DIR / path
    return str(path.resolve())


def _path_env_payload(
    values: Any,
    mapping: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    """以环境变量覆盖 YAML/默认路径，同时保留 Pydantic 校验。"""
    payload = dict(values or {})
    for field_name, env_names in mapping.items():
        for env_name in env_names:
            env_value = os.getenv(env_name, "").strip()
            if env_value:
                payload[field_name] = env_value
                break
    return payload


# ---------------------------------------------------------------------------
# 1. 启动期校验助手
# ---------------------------------------------------------------------------

def _require(key: str, hint: str = "") -> str:
    """
    启动时立即检测必须的环境变量。
    缺失则打印明确提示后退出，而不是运行到一半才报 OpenAIError。
    """
    val = os.getenv(key, "").strip()
    if not val:
        print(f"\n[config] ❌  缺少必要环境变量: {key}", file=sys.stderr)
        if hint:
            print(f"[config]    {hint}", file=sys.stderr)
        print(f"[config]    请在 .env 文件 或 conda env config vars set {key}=... 中设置\n",
              file=sys.stderr)
        sys.exit(1)
    return val


# ---------------------------------------------------------------------------
# 2. 环境变量层 Pydantic 模型
#    所有字段默认值 = "空字符串 / 合理默认值"，由 load_dotenv 或 Conda 覆盖
# ---------------------------------------------------------------------------

class SecretsConfig(BaseModel):
    """密钥类：不应出现在任何日志 / 提交记录中"""

    DEEPSEEK_API_KEY: str = Field(default="")
    DEEPSEEK_BASE_URL: str = Field(default="https://api.deepseek.com/v1")

    OPENAI_API_KEY: str = Field(default="")

    ANTHROPIC_API_KEY: str = Field(default="")

    TAVILY_API_KEY: str = Field(default="")

    DOCLING_API_KEY: str = Field(default="")

    LOCAL_OPENAI_BASE_URL: str = Field(default="http://127.0.0.1:8000/v1")
    LOCAL_OPENAI_API_KEY: str = Field(default="local-key")

    @model_validator(mode="after")
    def _load_from_env(self) -> "SecretsConfig":
        """从 os.environ 填充（load_dotenv 已经写入 os.environ）"""
        for field_name in self.__class__.model_fields:
            env_val = os.getenv(field_name, "").strip()
            if env_val:
                object.__setattr__(self, field_name, env_val)
        return self

    def __repr__(self) -> str:
        """防止密钥出现在日志里"""
        masked = {k: "***" if "KEY" in k else v
                  for k, v in self.model_dump().items()}
        return f"SecretsConfig({masked})"


class LLMConfig(BaseModel):
    """LLM 推理参数：可通过 .env 或 config.yaml 覆盖"""

    model: str = Field(default="deepseek/deepseek-v4-flash")
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, gt=0)
    llm_timeout: int = Field(default=60, gt=0)
    llm_retries: int = Field(default=2, ge=0)


class RuntimeConfig(BaseModel):
    """Agent 运行行为开关：可通过 .env 覆盖"""

    DEBUG: bool = Field(default=False)
    ENABLE_TOOLS: bool = Field(default=True)
    ENABLE_WEB_SEARCH: bool = Field(default=True)
    TIMEZONE: str = Field(default="Asia/Shanghai")

    RECURSION_LIMIT: int = Field(default=20, gt=0)
    MAX_HISTORY_TOKENS: int = Field(default=6000, gt=0)

    CHECKPOINT_BACKEND: str = Field(default="sqlite")
    CHECKPOINT_DB_PATH: str = Field(
        default="./data/agent-state/agent_checkpoints.sqlite3"
    )
    DEFAULT_THREAD_ID: str = Field(default="default")

    @model_validator(mode="after")
    def _load_from_env(self) -> "RuntimeConfig":
        bool_fields = {"DEBUG", "ENABLE_TOOLS", "ENABLE_WEB_SEARCH"}
        int_fields = {"MAX_SEARCH_RESULTS", "RECURSION_LIMIT", "MAX_HISTORY_TOKENS"}

        for field_name in self.__class__.model_fields:
            val = os.getenv(field_name, "").strip()
            if not val:
                continue
            if field_name in bool_fields:
                object.__setattr__(self, field_name, val.lower() == "true")
            elif field_name in int_fields:
                try:
                    object.__setattr__(self, field_name, int(val))
                except ValueError:
                    pass
            else:
                object.__setattr__(self, field_name, val)
        object.__setattr__(
            self,
            "CHECKPOINT_DB_PATH",
            resolve_app_path(self.CHECKPOINT_DB_PATH),
        )
        return self


class RedisConfig(BaseModel):
    """Redis 连接配置"""
    REDIS_URL: str = Field(default="redis://localhost:6379")
    REDIS_MAX_CONNECTIONS: int = Field(default=20)           # 连接池大小
    SEMANTIC_CACHE_TTL: int = Field(default=3600)            # 语义缓存 TTL，秒
    SEMANTIC_CACHE_THRESHOLD: float = Field(default=0.95)    # 相似度阈值
    BM25_INDEX_TTL: int = Field(default=86400)               # BM25 索引缓存，24小时
    BM25_HMAC_SECRET: str = Field(default="")

    @model_validator(mode="after")
    def _load_from_env(self) -> "RedisConfig":
        for field_name in self.__class__.model_fields:
            val = os.getenv(field_name, "").strip()
            if not val:
                continue
            tp = type(getattr(self, field_name))
            try:
                if tp is int:
                    object.__setattr__(self, field_name, int(val))
                elif tp is float:
                    object.__setattr__(self, field_name, float(val))
                else:
                    object.__setattr__(self, field_name, val)
            except Exception:
                pass
        return self


class PostgresConfig(BaseModel):
    """PostgreSQL Checkpointer 连接与连接池配置。"""

    POSTGRES_DB_URL: SecretStr = Field(default_factory=lambda: SecretStr(""))
    POSTGRES_POOL_MIN_SIZE: int = Field(default=1, ge=1)
    POSTGRES_POOL_MAX_SIZE: int = Field(default=10, ge=1)
    POSTGRES_CONNECT_TIMEOUT_SECONDS: float = Field(default=5.0, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, values: Any) -> dict[str, Any]:
        payload = dict(values or {})
        for field_name in cls.model_fields:
            value = os.getenv(field_name, "").strip()
            if value:
                payload[field_name] = value
        return payload

    @model_validator(mode="after")
    def _validate_pool_sizes(self) -> "PostgresConfig":
        if self.POSTGRES_POOL_MIN_SIZE > self.POSTGRES_POOL_MAX_SIZE:
            raise ValueError(
                "POSTGRES_POOL_MIN_SIZE 不能大于 POSTGRES_POOL_MAX_SIZE"
            )
        return self


class DoclingConfig(BaseModel):
    """独立 Docling Serve 的解析与自动路由配置。"""

    DOCLING_ENABLED: bool = Field(default=False)
    DOCLING_STRICT_MODE: bool = Field(default=False)
    DOCLING_PARSER_MODE: Literal["auto", "local", "docling"] = "auto"
    DOCLING_BASE_URL: str = Field(default="http://127.0.0.1:5001")
    DOCLING_REQUEST_TIMEOUT: float = Field(default=120.0, gt=0)
    DOCLING_TASK_TIMEOUT: float = Field(default=3600.0, gt=0)
    DOCLING_POLL_INTERVAL: float = Field(default=2.0, gt=0)
    DOCLING_MAX_PAGES_PER_TASK: int = Field(default=3, gt=0)
    DOCLING_SEGMENT_RETRIES: int = Field(default=1, ge=0)
    DOCLING_AUTO_IMAGE_BYTES_PER_PAGE: int = Field(default=307_200, gt=0)
    DOCLING_AUTO_MIN_TEXT_CHARS_PER_PAGE: int = Field(default=100, ge=0)
    DOCLING_AUTO_MIN_IMAGE_AREA_RATIO: float = Field(default=0.15, ge=0, le=1)
    DOCLING_AUTO_MIN_TABLE_LIKE_PAGES_RATIO: float = Field(default=0.6, ge=0, le=1)
    DOCLING_AUTO_MIN_MULTI_COLUMN_PAGES_RATIO: float = Field(default=0.4, ge=0, le=1)
    DOCLING_AUTO_SAMPLE_PAGES: int = Field(default=5, gt=0)
    DOCLING_MAX_PDF_PAGES: int = Field(default=200, gt=0)
    DOCLING_LOCAL_MIN_PAGE_COVERAGE: float = Field(default=0.65, ge=0, le=1)
    DOCLING_LOCAL_MIN_CHARS_PER_PAGE: int = Field(default=80, ge=0)
    DOCLING_LOCAL_MAX_REPLACEMENT_RATIO: float = Field(default=0.005, ge=0, le=1)
    DOCLING_LOCAL_MIN_TRACEABLE_RATIO: float = Field(default=0.95, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, values: Any) -> dict[str, Any]:
        """将环境变量交回 Pydantic 统一转换和校验，避免绕过比例边界。"""
        payload = dict(values or {})
        for field_name in cls.model_fields:
            value = os.getenv(field_name, "").strip()
            if value:
                payload[field_name] = value
        return payload


class RagIngestionConfig(BaseModel):
    """离线建库的资源边界与失败策略。"""

    RAG_INGESTION_WORKERS: int = Field(default=1, ge=1, le=16)
    RAG_INGESTION_BATCH_SIZE: int = Field(default=1_000, ge=1, le=20_000)
    RAG_INGESTION_FAIL_FAST: bool = Field(default=True)

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, values: Any) -> dict[str, Any]:
        payload = dict(values or {})
        for field_name in cls.model_fields:
            value = os.getenv(field_name, "").strip()
            if value:
                payload[field_name] = value
        return payload


# ---------------------------------------------------------------------------
# 3. YAML 层 Pydantic 模型（沿用你原有的结构，补充 absolute path 修正）
# ---------------------------------------------------------------------------
class RagProfile(BaseModel):
    """根据实际计算设备解析出的在线 RAG 运行参数。"""

    device: Literal["cpu", "cuda"]
    timeout: int = Field(gt=0)
    rerank_candidates: int = Field(ge=1, le=100)
    rerank_top_n: int = Field(ge=1, le=20)
    reranker_concurrency: int = Field(ge=1, le=8)


class RagConfig(BaseModel):
    """RAG 工具配置；支持 CPU/GPU 两套默认档位及环境变量覆盖。"""

    max_retries: int = Field(default=2, ge=0)
    max_content_chars: int = Field(default=800, ge=200, le=5000)
    client_timeout: float = Field(default=150.0, gt=0)

    # 兼容旧配置；设置后同时覆盖 CPU/GPU 的 timeout。
    timeout: int | None = Field(default=None, gt=0)
    cpu_timeout: int = Field(default=120, gt=0)
    cuda_timeout: int = Field(default=45, gt=0)

    # 全局字段用于临时强制覆盖；未设置时按设备选择对应档位。
    rerank_candidates: int | None = Field(default=None, ge=1, le=100)
    cpu_rerank_candidates: int = Field(default=12, ge=1, le=100)
    cuda_rerank_candidates: int = Field(default=20, ge=1, le=100)
    rerank_top_n: int = Field(default=5, ge=1, le=20)
    reranker_concurrency: int | None = Field(default=None, ge=1, le=8)
    cpu_reranker_concurrency: int = Field(default=1, ge=1, le=8)
    cuda_reranker_concurrency: int = Field(default=1, ge=1, le=8)

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, values: Any) -> dict[str, Any]:
        payload = dict(values or {})
        env_mapping = {
            "client_timeout": "KNOWLEDGE_SERVICE_TIMEOUT",
            "timeout": "RAG_TIMEOUT",
            "cpu_timeout": "RAG_CPU_TIMEOUT",
            "cuda_timeout": "RAG_CUDA_TIMEOUT",
            "rerank_candidates": "RAG_RERANK_CANDIDATES",
            "cpu_rerank_candidates": "RAG_CPU_RERANK_CANDIDATES",
            "cuda_rerank_candidates": "RAG_CUDA_RERANK_CANDIDATES",
            "rerank_top_n": "RAG_RERANK_TOP_N",
            "reranker_concurrency": "RAG_RERANKER_CONCURRENCY",
            "cpu_reranker_concurrency": "RAG_CPU_RERANKER_CONCURRENCY",
            "cuda_reranker_concurrency": "RAG_CUDA_RERANKER_CONCURRENCY",
        }
        for field_name, env_name in env_mapping.items():
            value = os.getenv(env_name, "").strip()
            if value:
                payload[field_name] = value
        return payload

    def profile(self, device: Literal["cpu", "cuda"]) -> RagProfile:
        """返回指定设备的最终配置，显式全局覆盖优先于设备档位。"""
        is_cuda = device == "cuda"
        return RagProfile(
            device=device,
            timeout=self.timeout or (self.cuda_timeout if is_cuda else self.cpu_timeout),
            rerank_candidates=self.rerank_candidates
            or (
                self.cuda_rerank_candidates
                if is_cuda
                else self.cpu_rerank_candidates
            ),
            rerank_top_n=self.rerank_top_n,
            reranker_concurrency=self.reranker_concurrency
            or (
                self.cuda_reranker_concurrency
                if is_cuda
                else self.cpu_reranker_concurrency
            ),
        )


class ExcelConfig(BaseModel):
    mode: Literal["timestamp", "overwrite", "append"] = "timestamp"
    keep_backup: bool = False


class SearchConfig(BaseModel):
    max_retries: int = 2
    timeout: int = 15
    max_search_results: int = 10


class DatabaseConfig(BaseModel):
    data_dir: str = str(_SRC_DIR / "data")
    # Chroma 数据库所在目录
    CHROMA_DB_PATH: str = str(_SRC_DIR / "chroma_db")

    # Hash 记录文件路径
    HASH_RECORD_PATH: str = str(_SRC_DIR / "chroma_db" / "file_hashes.json")
    # 与 Chroma/HNSW 解耦的规范化 Chunk 语料快照。
    CHUNK_STORE_PATH: str = str(_SRC_DIR / "data" / "knowledge" / "chunks.sqlite3")

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, values: Any) -> dict[str, Any]:
        return _path_env_payload(
            values,
            {
                "data_dir": ("DATA_DIR",),
                "CHROMA_DB_PATH": ("CHROMA_DB_PATH",),
                "HASH_RECORD_PATH": ("HASH_RECORD_PATH",),
                "CHUNK_STORE_PATH": ("CHUNK_STORE_PATH",),
            },
        )

    @model_validator(mode="after")
    def _resolve_paths(self) -> "DatabaseConfig":
        for field_name in (
            "data_dir",
            "CHROMA_DB_PATH",
            "HASH_RECORD_PATH",
            "CHUNK_STORE_PATH",
        ):
            object.__setattr__(
                self,
                field_name,
                resolve_app_path(getattr(self, field_name)),
            )
        return self


class SqlDataConfig(BaseModel):
    DB_PATH: str = str(_SRC_DIR / "data_sql" / "financials.db")

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, values: Any) -> dict[str, Any]:
        return _path_env_payload(values, {"DB_PATH": ("SQL_DB_PATH", "DB_PATH")})

    @model_validator(mode="after")
    def _resolve_path(self) -> "SqlDataConfig":
        object.__setattr__(self, "DB_PATH", resolve_app_path(self.DB_PATH))
        return self


class FileStorageConfig(BaseModel):
    """Agent 生成文件的持久化目录。"""

    OUTPUT_DIR: str = str(_SRC_DIR / "outputs")

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, values: Any) -> dict[str, Any]:
        return _path_env_payload(values, {"OUTPUT_DIR": ("OUTPUT_DIR",)})

    @model_validator(mode="after")
    def _resolve_path(self) -> "FileStorageConfig":
        object.__setattr__(self, "OUTPUT_DIR", resolve_app_path(self.OUTPUT_DIR))
        return self


class ToolsConfig(BaseModel):
    rag:          RagConfig = RagConfig()
    excel:        ExcelConfig = ExcelConfig()
    search:       SearchConfig = SearchConfig()
    vector_store: DatabaseConfig = DatabaseConfig()
    sql_store:    SqlDataConfig = SqlDataConfig()


class AgentCoreConfig(BaseModel):
    MAX_HISTORY_TOKENS: int = 6000


class CoreConfig(BaseModel):
    agent: AgentCoreConfig = AgentCoreConfig()


class YamlConfig(BaseModel):
    tools: ToolsConfig = ToolsConfig()
    core:  CoreConfig = CoreConfig()


def _load_yaml() -> YamlConfig:
    if _YAML_PATH.exists():
        with open(_YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return YamlConfig(**data)
    else:
        print(f"[config] ⚠️  未找到 config.yaml（{_YAML_PATH}），使用默认参数")
        return YamlConfig()


# ---------------------------------------------------------------------------
# 4. 顶层聚合：Settings —— 对外唯一暴露的对象
# ---------------------------------------------------------------------------

class Settings(BaseModel):
    """
    全局设置聚合器。

    用法（任意脚本）：
        from react_agent.core.config import settings
        settings.secrets.DEEPSEEK_API_KEY
        settings.llm.LLM_TEMPERATURE
        settings.runtime.RECURSION_LIMIT
        settings.yaml.tools.rag.max_retries
    """
    secrets: SecretsConfig = Field(default_factory=SecretsConfig)
    llm:     LLMConfig = Field(default_factory=LLMConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    yaml:    YamlConfig = Field(default_factory=_load_yaml)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    docling: DoclingConfig = Field(default_factory=DoclingConfig)
    ingestion: RagIngestionConfig = Field(default_factory=RagIngestionConfig)
    storage: FileStorageConfig = Field(default_factory=FileStorageConfig)

    # 便捷属性：让 llm.py 改动最小
    @property
    def model(self) -> str:
        return self.llm.model

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def app_root(self) -> Path:
        """包含 config.yaml、运行数据目录和 Python 包的应用根目录。"""
        return _SRC_DIR

    @property
    def tools(self) -> ToolsConfig:
        return self.yaml.tools  # 转发给 yaml 层

    @property
    def core(self) -> CoreConfig:
        return self.yaml.core  # 转发给 yaml 层


# ---------------------------------------------------------------------------
# 5. 全局单例 —— 模块加载时初始化一次
# ---------------------------------------------------------------------------
settings = Settings()

# ---------------------------------------------------------------------------
# 6. 启动期密钥校验（仅在直接运行此文件时触发完整校验，import 时不强制退出）
#    真正需要密钥的地方（llm.py / eval_rag.py）会在用到时自行校验
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 运行 python config.py 可以快速验证配置是否完整
    print("=" * 60)
    print(f"项目根目录 : {_PROJECT_ROOT}")
    print(f".env 路径  : {_ENV_PATH}  ({'✅ 存在' if _ENV_PATH.exists() else '❌ 不存在'})")
    print(f"YAML 路径  : {_YAML_PATH} ({'✅ 存在' if _YAML_PATH.exists() else '❌ 不存在'})")
    print()
    print(f"[LLM]     MODEL = {settings.llm.model}")
    print(f"[LLM]     TEMPERATURE = {settings.llm.llm_temperature}")
    print(f"[LLM]     MAX_TOKENS = {settings.llm.llm_max_tokens}")
    print()
    print(f"[密钥]    DEEPSEEK_API_KEY= {'✅ 已设置' if settings.secrets.DEEPSEEK_API_KEY else '❌ 未设置'}")
    print(f"[密钥]    OPENAI_API_KEY = {'✅ 已设置' if settings.secrets.OPENAI_API_KEY  else '⚠️  未设置（可选）'}")
    print(f"[密钥]    TAVILY_API_KEY = {'✅ 已设置' if settings.secrets.TAVILY_API_KEY  else '⚠️  未设置（可选）'}")
    print()
    print(f"[运行时]  RECURSION_LIMIT = {settings.runtime.RECURSION_LIMIT}")
    print(f"[运行时]  MAX_HISTORY = {settings.runtime.MAX_HISTORY_MESSAGES}")
    print()
    print(f"[YAML]    data_dir = {settings.yaml.tools.vector_store.data_dir}")
    print(f"[YAML]    excel.mode = {settings.yaml.tools.excel.mode}")
    print("=" * 60)
    print("✅ 配置加载完成")
