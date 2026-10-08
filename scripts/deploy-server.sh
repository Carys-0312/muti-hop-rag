#!/bin/bash
# ============================================================
#  多跳检索问答 Agent —— 服务器部署脚本（不依赖 docker compose）
#
#  为什么不用 compose：
#    这台服务器是 CentOS 7，yum 源已失效，docker compose 插件
#     在阿里云镜像站上也没有，只能从 GitHub 下（国内不稳）。
#     本脚本改用 docker run + 自定义网络，功能完全等价。
#
#  用法（在服务器上执行）：
#     bash scripts/deploy-server.sh deps     # 起 Redis + Qdrant
#     bash scripts/deploy-server.sh build    # 构建 API 镜像
#     bash scripts/deploy-server.sh index    # 建向量索引（最慢）
#     bash scripts/deploy-server.sh api      # 启动 API
#     bash scripts/deploy-server.sh status   # 查看状态
#     bash scripts/deploy-server.sh logs     # 看 API 日志
#     bash scripts/deploy-server.sh test     # 验证（health + demo）
#     bash scripts/deploy-server.sh stop     # 停掉全部
#     bash scripts/deploy-server.sh restart  # 重启 API
# ============================================================

set -e

# ── 配置 ──────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NET="mhrag-net"
IMAGE="mhrag-api:latest"
REDIS_NAME="mhrag-redis"
QDRANT_NAME="mhrag-qdrant"
API_NAME="mhrag-api"

# 内部服务地址（容器名 + 网络别名，脚本里已给容器加 network-alias）
REDIS_URL="redis://redis:6379/0"
QDRANT_URL="http://qdrant:6333"

# ⭐ 把宿主机源码挂进容器：
#    镜像是 `pip install -e .`（editable）安装，/app/src 是一个软链接目标。
#    挂载宿主机的 src/configs/scripts 覆盖容器内的同名目录后，
#    改代码【不需要重建镜像】——重启容器即可生效。
#    这对低配服务器很重要：重建一次镜像要 15~40 分钟。
SRC_MOUNTS=(
    -v "$PROJECT_DIR/src:/app/src"
    -v "$PROJECT_DIR/configs:/app/configs"
    -v "$PROJECT_DIR/scripts:/app/scripts"
)

# ⭐ 编码环境变量（必须！）
#    python:*-slim 镜像默认没有 locale（LANG 为空），locale 退化成 POSIX/C，
#    于是 Python 把 sys.stdout / sys.stderr 当成 ASCII 编码。
#    一旦写中文（日志、异常 traceback、print），就抛：
#        UnicodeEncodeError: 'ascii' codec can't encode characters in position ...
#    而 API 的流式接口会把这个异常原样推给前端，所以浏览器上直接看到报错。
#    显式设成 UTF-8 即可解决。
ENC_ENVS=(
    -e PYTHONIOENCODING=utf-8
    -e PYTHONUTF8=1
    -e LANG=C.UTF-8
    -e LC_ALL=C.UTF-8
)

# 国内镜像（HuggingFace 模型下载，不用会非常慢）
HF_ENDPOINT_VAL="https://hf-mirror.com"

cd "$PROJECT_DIR"

# ── 工具函数 ──────────────────────────────────────────────
info()  { echo -e "\n\033[36m▶ $*\033[0m"; }
ok()    { echo -e "\033[32m✅ $*\033[0m"; }
warn()  { echo -e "\033[33m⚠️  $*\033[0m"; }
die()   { echo -e "\033[31m❌ $*\033[0m" >&2; exit 1; }

need_docker() {
    command -v docker >/dev/null 2>&1 || die "没找到 docker 命令"
    docker info >/dev/null 2>&1 || die "Docker 守护进程没在运行：systemctl start docker"
}

ensure_network() {
    if ! docker network inspect "$NET" >/dev/null 2>&1; then
        info "创建网络 $NET"
        docker network create "$NET" >/dev/null
    fi
}

wait_healthy() {
    # wait_healthy <容器名> <探测命令...>
    local name="$1"; shift
    local i=0
    printf "  等待 %s 就绪" "$name"
    while [ $i -lt 40 ]; do
        if docker exec "$name" "$@" >/dev/null 2>&1; then
            printf " ✅\n"; return 0
        fi
        printf "."
        sleep 3
        i=$((i + 1))
    done
    printf "\n"
    warn "$name 等待超时，继续（可能是启动慢，后面用 status 检查）"
    return 0
}

# ══════════════════════════════════════════════════════════
#  deps —— 起 Redis 和 Qdrant
# ══════════════════════════════════════════════════════════
cmd_deps() {
    need_docker
    ensure_network

    info "启动 Redis"
    docker rm -f "$REDIS_NAME" >/dev/null 2>&1 || true
    docker run -d \
        --name "$REDIS_NAME" \
        --network "$NET" \
        --network-alias redis \
        --restart unless-stopped \
        -v mhrag-redis-data:/data \
        redis:7-alpine redis-server --appendonly yes >/dev/null
    wait_healthy "$REDIS_NAME" redis-cli ping
    ok "Redis 已启动（容器名 $REDIS_NAME，网络别名 redis）"

    info "启动 Qdrant"
    docker rm -f "$QDRANT_NAME" >/dev/null 2>&1 || true
    docker run -d \
        --name "$QDRANT_NAME" \
        --network "$NET" \
        --network-alias qdrant \
        --restart unless-stopped \
        -p 127.0.0.1:6333:6333 \
        -v mhrag-qdrant-data:/qdrant/storage \
        qdrant/qdrant:latest >/dev/null
    ok "Qdrant 已启动（容器名 $QDRANT_NAME，网络别名 qdrant）"

    echo
    docker ps --filter "name=mhrag-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

# ══════════════════════════════════════════════════════════
#  build —— 构建 API 镜像
# ══════════════════════════════════════════════════════════
cmd_build() {
    need_docker

    # 检查内存，给出预期
    local mem_mb
    mem_mb=$(awk '/MemTotal/ {printf "%d", $2/1024}' /proc/meminfo)
    local swap_mb
    swap_mb=$(awk '/SwapTotal/ {printf "%d", $2/1024}' /proc/meminfo)
    echo "  内存: ${mem_mb}MB，Swap: ${swap_mb}MB"
    if [ "$mem_mb" -lt 3500 ] && [ "$swap_mb" -lt 2000 ]; then
        warn "内存和 Swap 都偏小，构建可能很慢或失败"
        warn "建议先加 Swap：见 DEPLOY.md 或对话记录"
    fi

    info "构建镜像 $IMAGE（首次约 15~40 分钟，请勿中断）"
    docker build -t "$IMAGE" .
    ok "镜像构建完成"

    echo
    docker images "$IMAGE" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
}

# ══════════════════════════════════════════════════════════
#  index —— 建向量索引（最慢的一步）
# ══════════════════════════════════════════════════════════
cmd_index() {
    need_docker

    ensure_network
    docker ps --filter "name=$REDIS_NAME" --format '{{.Names}}' | grep -q . \
        || die "Redis 没在运行。先执行：bash scripts/deploy-server.sh deps"
    docker ps --filter "name=$QDRANT_NAME" --format '{{.Names}}' | grep -q . \
        || die "Qdrant 没在运行。先执行：bash scripts/deploy-server.sh deps"

    mkdir -p data/raw data/processed data/eval results

    # 文档库已存在就不用重新下载（省 5~10 分钟）
    local dl_flag="--download"
    if [ -f "$PROJECT_DIR/data/processed/corpus.jsonl" ]; then
        dl_flag=""
        info "检测到已有文档库（data/processed/corpus.jsonl），跳过下载与抽取"
    fi

    info "建索引（向量化并写入 Qdrant，约 30~60 分钟）"
    echo "  ⚠️ 这一步最慢。有进度条在走就是正常，别中断。"
    echo "  💡 这一步【不需要】DeepSeek API Key，只做本地向量化。"
    echo "  💡 中断了不要紧：直接重跑同一条命令，会自动【断点续传】。"
    echo

    docker run --rm \
        --network "$NET" \
        -e REDIS_URL="$REDIS_URL" \
        -e QDRANT_URL="$QDRANT_URL" \
        -e HF_ENDPOINT="$HF_ENDPOINT_VAL" \
        -e HF_HUB_DISABLE_XET=1 \
        -v "$PROJECT_DIR/data:/app/data" \
        -v mhrag-hf-cache:/root/.cache/huggingface \
        "${SRC_MOUNTS[@]}" \
        "$IMAGE" \
        python scripts/build_index.py $dl_flag

    ok "索引完成"
}

# ══════════════════════════════════════════════════════════
#  evalset —— 生成评测集（跑 test / eval 前必须先做）
# ══════════════════════════════════════════════════════════
cmd_evalset() {
    need_docker
    ensure_network
    mkdir -p data/eval

    info "生成评测集（200 条分层抽样）"
    echo "  ⚠️ 必须和建索引用同一批问题（脚本读同一个 config，seed 一致）"
    echo

    docker run --rm \
        --network "$NET" \
        -e HF_ENDPOINT="$HF_ENDPOINT_VAL" \
        -e HF_HUB_DISABLE_XET=1 \
        -v "$PROJECT_DIR/data:/app/data" \
        -v mhrag-hf-cache:/root/.cache/huggingface \
        "${SRC_MOUNTS[@]}" \
        "$IMAGE" \
        python scripts/build_eval_set.py --force

    ok "评测集已生成"
    echo
    echo "  验证："
    docker run --rm \
        -v "$PROJECT_DIR/data:/app/data" \
        "${SRC_MOUNTS[@]}" \
        "$IMAGE" python -c "import json;from multihop_rag.config import settings;rows=[json.loads(l) for l in open(settings.eval_set_path)];print(f'共 {len(rows)} 条');[print(' ', r['qid'],'|',r['type'],'/',r['level'],'|',r['question'][:70]) for r in rows[:5]]"
}

# ══════════════════════════════════════════════════════════
#  api —— 启动 API
# ══════════════════════════════════════════════════════════
cmd_api() {
    need_docker
    [ -f .env ] || die ".env 不存在。先执行：cp .env.example .env && vim .env"

    ensure_network

    info "启动 API 容器"
    docker rm -f "$API_NAME" >/dev/null 2>&1 || true

    docker run -d \
        --name "$API_NAME" \
        --network "$NET" \
        --restart unless-stopped \
        -p 8000:8000 \
        --env-file .env \
        "${ENC_ENVS[@]}" \
        -e REDIS_URL="$REDIS_URL" \
        -e QDRANT_URL="$QDRANT_URL" \
        -e MHRAG_ROOT=/app \
        -e HF_ENDPOINT="$HF_ENDPOINT_VAL" \
        -e HF_HUB_DISABLE_XET=1 \
        -v "$PROJECT_DIR/data:/app/data" \
        -v "$PROJECT_DIR/results:/app/results" \
        -v mhrag-hf-cache:/root/.cache/huggingface \
        "${SRC_MOUNTS[@]}" \
        "$IMAGE" >/dev/null

    sleep 5
    echo
    docker ps --filter "name=mhrag-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo
    ok "API 已启动"
    echo "  本机验证:  curl -s http://localhost:8000/health"
    echo "  外部访问:  http://<服务器公网IP>:8000   （需在阿里云安全组放行 8000）"
    echo "  看日志:    bash scripts/deploy-server.sh logs"
}

# ══════════════════════════════════════════════════════════
#  status / logs / test / stop / restart
# ══════════════════════════════════════════════════════════
cmd_status() {
    need_docker
    echo "=== 容器状态 ==="
    docker ps -a --filter "name=mhrag-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo
    echo "=== 资源占用 ==="
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" \
        2>/dev/null | grep -E "NAME|mhrag" || true
    echo
    echo "=== 宿主机内存 ==="
    free -h
}

cmd_logs() {
    need_docker
    echo "（Ctrl+C 退出查看，不会停服务）"
    docker logs -f --tail 100 "$API_NAME"
}

cmd_test() {
    need_docker
    echo "=== 1. 健康检查 ==="
    curl -s http://localhost:8000/health || die "API 没响应"
    echo
    echo
    echo "=== 2. 检索冒烟测试（用评测集里的问题，需要 API Key）==="
    docker run --rm \
        --network "$NET" \
        --env-file .env \
        "${ENC_ENVS[@]}" \
        -e REDIS_URL="$REDIS_URL" \
        -e QDRANT_URL="$QDRANT_URL" \
        -v "$PROJECT_DIR/data:/app/data" \
        -v mhrag-hf-cache:/root/.cache/huggingface \
        "${SRC_MOUNTS[@]}" \
        "$IMAGE" python scripts/demo.py --n 2
}

cmd_stop() {
    need_docker
    info "停止并删除容器（数据卷保留）"
    docker rm -f "$API_NAME" "$QDRANT_NAME" "$REDIS_NAME" >/dev/null 2>&1 || true
    ok "已停止。数据在卷里：mhrag-redis-data / mhrag-qdrant-data / mhrag-hf-cache"
    echo "  重新启动: bash scripts/deploy-server.sh deps && bash scripts/deploy-server.sh api"
}

cmd_restart() {
    need_docker
    info "重启 API"
    docker restart "$API_NAME"
    sleep 5
    docker ps --filter "name=$API_NAME" --format "table {{.Names}}\t{{.Status}}"
}

# ══════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════
case "${1:-}" in
    deps)    cmd_deps ;;
    build)   cmd_build ;;
    evalset) cmd_evalset ;;
    index)   cmd_index ;;
    api)     cmd_api ;;
    status)  cmd_status ;;
    logs)    cmd_logs ;;
    test)    cmd_test ;;
    stop)    cmd_stop ;;
    restart) cmd_restart ;;
    all)
        cmd_deps
        cmd_build
        cmd_evalset
        cmd_index
        cmd_api
        cmd_test
        ;;
    *)
        cat <<'USAGE'
多跳检索问答 Agent —— 服务器部署脚本

用法：
  bash scripts/deploy-server.sh deps     起 Redis + Qdrant
  bash scripts/deploy-server.sh build    构建 API 镜像（15~40 分钟）
  bash scripts/deploy-server.sh evalset  生成评测集（200 条，跑 test 前需要）
  bash scripts/deploy-server.sh index    建向量索引（30~60 分钟，最慢）
  bash scripts/deploy-server.sh api      启动 API
  bash scripts/deploy-server.sh status   查看容器状态 + 资源占用
  bash scripts/deploy-server.sh logs     看 API 日志
  bash scripts/deploy-server.sh test     验证（health + demo）
  bash scripts/deploy-server.sh stop     停掉全部（数据卷保留）
  bash scripts/deploy-server.sh restart  重启 API
  bash scripts/deploy-server.sh all      按顺序跑完 deps→build→evalset→index→api→test

推荐顺序：
  deps → build → evalset → index → api → test

⚠️ evalset 和 index 必须用同一批问题（都读 config 的 data.corpus_questions，
   seed 一致），否则评测集里的金标准文档不在索引里，指标会全错。
USAGE
        exit 1
        ;;
esac
