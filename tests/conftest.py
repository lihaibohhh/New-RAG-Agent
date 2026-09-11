"""
Root conftest — 必须在任何 api.* import 之前设置假环境变量，
否则 api.main 的 validate_llm_key() 在 CI（无 secret）环境里会 sys.exit(1)。
"""
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-fake")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake")
# 关闭 LangSmith/LangChain tracing，避免外网调用和 git metadata 采集。
for _tracing_env in (
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGSMITH_TRACING",
):
    os.environ[_tracing_env] = "false"
# 禁用 .env 文件加载对测试环境的干扰（覆盖在 import 前已生效）
