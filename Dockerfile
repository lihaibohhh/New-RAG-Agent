# Python 3.10 与项目当前运行基线保持一致；固定 Debian 系列，避免发行版漂移。
FROM python:3.10-slim-bookworm AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 默认使用官方 PyPI；国内环境可在构建时通过 --build-arg 覆盖。
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PYTORCH_VERSION=2.12.0

WORKDIR /app

# 使用固定 UID/GID 的非 root 用户运行服务。
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --no-log-init app

# pyproject.toml 是唯一依赖来源。先复制安装所需文件，尽量复用依赖层；
# CPU Torch 先只下载 wheel，再用 PyPI 安装其普通依赖，避免将 PyTorch
# 专用索引误当成 typing-extensions、flit-core 等通用包的唯一来源。
# BuildKit 缓存只加速下载，不会进入最终镜像。
COPY --chown=app:app pyproject.toml ./
COPY --chown=app:app react_agent ./react_agent
COPY --chown=app:app knowledge ./knowledge
COPY --chown=app:app mcp_service ./mcp_service
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p /tmp/torch-cpu \
    && python -m pip download --no-deps --dest /tmp/torch-cpu \
       --index-url "${PYTORCH_INDEX_URL}" "torch==${PYTORCH_VERSION}" \
    && python -m pip install --index-url "${PIP_INDEX_URL}" /tmp/torch-cpu/torch-*.whl \
    && rm -rf /tmp/torch-cpu \
    && python -m pip install --index-url "${PIP_INDEX_URL}" .

# 其余运行时代码后复制，避免 README、日志和本地数据污染镜像。
COPY --chown=app:app . .

RUN mkdir -p /app/data /app/chroma_db /app/outputs /app/logs \
    && chown -R app:app /app

EXPOSE 8001 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

# 独立评测任务镜像。完整 RAGAS 依赖不进入在线服务镜像。
FROM runtime-base AS eval-runner

USER root
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade "pip>=25.1,<26" \
    && python -m pip install --group eval . \
    && mkdir -p /app/eval-results \
    && chown -R app:app /app/eval-results

USER app
HEALTHCHECK NONE
CMD ["python", "-m", "eval.run_eval", "--help"]

# 保持默认构建目标为在线应用，避免评测依赖扩大生产镜像。
FROM runtime-base AS app

USER app

CMD ["python", "-m", "streamlit", "run", "scripts/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
