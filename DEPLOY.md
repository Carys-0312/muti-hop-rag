# 部署到服务器

本文档是从零把项目部署到一台 Linux 服务器的完整步骤。

---

## 0. 服务器要求

| 项 | 最低 | 推荐 |
|---|---|---|
| CPU | 2 核 | 4 核 |
| 内存 | **4 GB** | 8 GB |
| 磁盘 | 20 GB | 40 GB |
| 系统 | Ubuntu 22.04 / Debian 12 | 同 |

**为什么至少 4GB 内存**：embedding 模型（ONNX）常驻约 500MB，Qdrant 约 300MB，
Redis 约 50MB，Python 进程约 1GB。2GB 的机器会在建索引时 OOM。

> ⚠️ **成本提醒**：服务器只是运行环境，**每次问答都会调用 DeepSeek API 产生费用**。
> 上线前先估算日均请求量，并给 API 账号设消费上限。

---

## 1. 服务器装 Docker

```bash
# Ubuntu / Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# 重新登录让用户组生效，然后验证：
docker --version && docker compose version
```

**国内服务器还需要配置镜像加速器**（否则拉镜像极慢甚至失败）：

```bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<'EOF'
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io"
  ]
}
EOF
sudo systemctl restart docker
```

---

## 2. 把代码传到服务器

### 方式 A：Git（推荐）

```bash
# 服务器上
git clone <你的仓库地址> multi-hop-rag
cd multi-hop-rag
```

> **前提**：你已经把项目推到 GitHub/Gitee，且 `.gitignore` 正确排除了 `.env` 和 `data/`。

### 方式 B：本地打包上传（适合内网服务器）

```powershell
# 本地（排除数据和密钥）
cd "C:\Users\27542\Documents\deepseek-harness\default-workspace"
tar -czf multi-hop-rag.tar.gz --exclude=multi-hop-rag/data/raw --exclude=multi-hop-rag/data/processed --exclude=multi-hop-rag/.venv --exclude=multi-hop-rag/.env --exclude=multi-hop-rag/.git multi-hop-rag

scp multi-hop-rag.tar.gz user@服务器IP:/home/user/
```

```bash
# 服务器
tar -xzf multi-hop-rag.tar.gz && cd multi-hop-rag
```

---

## 3. 配置环境变量（不要用本地 .env！）

```bash
cp .env.example .env
vim .env
```

**必须修改的内容：**

```ini
# ⭐ 生产环境的 API Key（不要复用开发用的）
DEEPSEEK_API_KEY=sk-生产环境的key

# ⭐ 保持镜像配置
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_DISABLE_XET=1

# ⚠️ 注意：下面两个在 compose 里已经被覆盖成服务名，
#    写 localhost 也没关系，但别改成公网地址
REDIS_URL=redis://redis:6379/0
QDRANT_URL=http://qdrant:6333
```

**安全加固：**

```bash
chmod 600 .env      # 只有属主可读
```

---

## 4. 构建镜像

```bash
# ⭐ 注意用 -f 显式指定，避免加载开发用的 override 文件
docker compose -f docker-compose.yml -f docker-compose.prod.yml build
```

> 构建时会从 PyPI 装依赖（Dockerfile 里已配清华源），约 **5~15 分钟**。

---

## 5. 启动依赖服务

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d redis qdrant
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
```

**验收**：两个容器都是 `running`，redis 显示 `(healthy)`。

---

## 6. ⭐ 建向量索引（**必须做，否则检索不到任何东西**）

镜像里**不包含索引数据**（`.dockerignore` 排除了 `data/`），必须在服务器上建一次。

### 方式 A：服务器上重新下载并建索引（简单，需要能访问 HuggingFace 镜像）

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml run --rm api \
  python scripts/build_index.py --download
```

约 **10~15 分钟**（取决于服务器 CPU）。

### 方式 B：从本地导出 Qdrant 快照（推荐，快且不依赖服务器网络）

```powershell
# ① 本地：创建快照
curl.exe -X POST "http://localhost:6333/collections/hotpotqa/snapshots"

# ② 本地：查看快照名
curl.exe "http://localhost:6333/collections/hotpotqa/snapshots"

# ③ 本地：下载（把 <name> 换成上一步返回的文件名）
curl.exe -o hotpotqa.snapshot "http://localhost:6333/collections/hotpotqa/snapshots/<name>"

# ④ 传到服务器
scp hotpotqa.snapshot user@服务器IP:/tmp/
```

```bash
# ⑤ 服务器：上传快照恢复（会自动创建集合并导入数据）
curl -X POST "http://127.0.0.1:6333/collections/hotpotqa/snapshots/upload?priority=snapshot" \
  -F "snapshot=@/tmp/hotpotqa.snapshot"

# ⑥ 验证条数
curl -s http://127.0.0.1:6333/collections/hotpotqa | head -c 300
```

> **方式 B 的前提**：服务器上先启动 qdrant（第 5 步已完成）。

### 评测集（可选，只在服务器上跑评测才需要）

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml run --rm api \
  python scripts/build_eval_set.py --force
```

---

## 7. 启动 API

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d api
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f api
```

**验收：**

```bash
# 健康检查（应返回 count > 0）
curl -s http://localhost:8000/health

# 问一个问题
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Were Pavel Urysohn and Leonid Levin born in the same country?","strategy":"adaptive"}'
```

> ⚠️ 上面那个问题来自 HotpotQA 的 **dev** split，而我们的索引是用 **train** split 建的，
> 所以答案可能不理想。**想验证效果，请用评测集里的问题**：
> ```bash
> docker compose ... exec api python scripts/demo.py --n 2
> ```

浏览器访问 `http://服务器IP:8000` 可以看到演示页面。

---

## 8. 开放端口 & 反向代理（推荐）

### 只放行 8000（最简单的做法）

```bash
# Ubuntu 的 ufw
sudo ufw allow 22
sudo ufw allow 8000
sudo ufw enable
```

> **Redis(6379) 和 Qdrant(6333) 已经只绑定 127.0.0.1**，公网访问不到，
> 所以防火墙不放行它们是正确的。

### 用 Nginx + 域名 + HTTPS（更正式）

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # ⭐ SSE 流式接口必须关掉缓冲，否则 /ask/stream 不会实时推送
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
}
```

```bash
# HTTPS 证书（Let's Encrypt）
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

> ⭐ **`proxy_buffering off` 是关键** —— 不关的话 SSE 流式输出会被 Nginx 缓存，
> 前端就看不到"逐跳实时输出"的效果了，你的演示亮点会失效。

---

## 9. 日常运维

```bash
# 看状态
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps

# 看日志（实时）
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f api

# 更新代码后重新部署
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# 停掉（数据卷保留）
docker compose -f docker-compose.yml -f docker-compose.prod.yml down

# 备份 Qdrant 数据
docker run --rm -v multi-hop-rag_qdrant-data:/data -v $(pwd):/backup alpine \
  tar czf /backup/qdrant-backup-$(date +%F).tar.gz -C /data .
```

**设置开机自启**：`restart: unless-stopped` 已经配好了，服务器重启后容器会自动起来。

---

## 10. 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `api` 容器反复重启 | `.env` 没配 / 索引是空的 | `logs api` 看报错；确认 `/health` 里 count > 0 |
| 检索结果为空 | 索引没建 | 重做第 6 步 |
| `Cannot connect to the Docker daemon` | Docker 没启动 | `sudo systemctl start docker` |
| 建索引时容器被杀 | 内存不足（OOM） | 升内存，或把 `scripts/build_index.py --batch-size` 调小 |
| 拉镜像超时 | 没配镜像加速器 | 做第 1 步的 daemon.json 配置 |
| SSE 不实时输出 | Nginx 缓冲没关 | 加 `proxy_buffering off` |
| API 报 402 | DeepSeek 余额不足 | 充值 |

---

## 11. 和开发环境的差异速查

| | 本地开发 | 服务器生产 |
|---|---|---|
| 启动命令 | `docker compose up -d` | `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d` |
| 代码来源 | 挂载本机 `./src` | 镜像里 bake 好的 |
| 热重载 | ✅ `--reload` | ❌ 无 |
| uvicorn worker | 1 | 2 |
| 日志级别 | info | warning |
| Redis/Qdrant 端口 | 127.0.0.1 | 127.0.0.1 |
| 内存限制 | 无 | 3GB |
| 日志轮转 | 无 | 10MB × 3 |
