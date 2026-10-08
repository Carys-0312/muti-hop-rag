# ============================================================
#  多跳检索问答 Agent —— 应用镜像
#
#  分层构建的关键：先装依赖，再拷代码。
#  这样只改业务代码时，下面那个耗时最久的 pip install 层能命中缓存。
# ============================================================
FROM python:3.12-slim

WORKDIR /app

# ── ⭐ 关键：给容器内的 pip 设置国内镜像源 ──
# 不设的话，构建时会去连 pypi.org，在国内会卡死。
# 想换源可以在构建时覆盖：docker build --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip config set global.index-url "$PIP_INDEX_URL" \
    && pip config set global.timeout 120 \
    && pip config set global.retries 10

# ── ⭐ HuggingFace 镜像 + 禁用 Xet ──
# HF_ENDPOINT：模型走镜像站下载（否则国内直连很慢）
# HF_HUB_DISABLE_XET：禁用 HF 的 Xet 存储后端。
#     因为 Xet 会 302 重定向到 cas-server.xethub.hf.co，
#     而镜像站不代理那个后端 → 会 401 失败。
ENV HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_DISABLE_XET=1

# 说明：这里【不需要】build-essential。
# embedding 用的是 fastembed(ONNX Runtime)，所有依赖都有预编译 wheel。
# 如果将来 pip 报"编译失败"，再把下面这行放开：
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
#     && rm -rf /var/lib/apt/lists/*

# ① 先装依赖（利用层缓存：只改代码时这层不会重跑）
COPY pyproject.toml README.md ./
RUN mkdir -p src/multihop_rag \
    && touch src/multihop_rag/__init__.py \
    && pip install --no-cache-dir -e . \
    && rm -rf src

# ② 再拷真实代码
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
RUN pip install --no-cache-dir -e .

# config.py 靠这个找到 configs/ 目录（见 src/multihop_rag/config.py）
#
# ⭐ 编码相关的环境变量（必须）：
#    python:*-slim 镜像默认没有 locale，LANG 为空 → locale 退化成 POSIX/C，
#    Python 于是把 sys.stdout / sys.stderr 当成 ASCII 编码，
#    一写中文（日志、traceback、print）就抛
#        UnicodeEncodeError: 'ascii' codec can't encode characters in position ...
#    API 的流式接口会把这个异常推给前端，所以浏览器上会直接看到报错。
ENV MHRAG_ROOT=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUTF8=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

EXPOSE 8000

# ⚠️ 必须是【前台运行】的命令，否则容器会立刻退出
CMD ["uvicorn", "multihop_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
