"""Qdrant 向量库封装。

为什么用 Qdrant 而不是 FAISS：
- 它可以通过 Docker 起成独立服务，和 Redis 一起用 compose 编排
- 有 Web UI 可以直接看数据（http://localhost:6333/dashboard）
- 更接近真实生产环境

⚠️ 关于 qdrant-client 的版本兼容：
    新版（1.x 后期）把 `client.search()` 换成了 `client.query_points()`，
    并移除了旧方法。这里对两种 API 都做了适配，
    避免因为 client 版本不同就挂掉。
    —— 这也是"依赖版本别写太松"的真实教训。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, collection: str | None = None, dim: int | None = None):
        from qdrant_client import QdrantClient

        self.collection = collection or settings.retrieval.collection
        self.dim = dim

        # ⭐ 超时必须显式设置。
        #    qdrant-client 默认只有 5 秒 —— 在低配服务器上（1 核 2G），
        #    一批 64 条的 upsert 经常超过 5 秒，会抛
        #        ResponseHandlingException: timed out
        #    导致整个建索引流程中断。这里默认给 120 秒。
        self.timeout = int(settings.retrieval.get("qdrant_timeout", 120))

        self.client = QdrantClient(url=settings.qdrant_url, timeout=self.timeout)
        logger.info(
            "Qdrant 客户端已连接: %s（API: %s，timeout: %ds）",
            settings.qdrant_url,
            "query_points" if hasattr(self.client, "query_points") else "search",
            self.timeout,
        )

    # ── 集合管理 ──
    def collection_exists(self) -> bool:
        """新旧版本都兼容"""
        if hasattr(self.client, "collection_exists"):
            try:
                return bool(self.client.collection_exists(self.collection))
            except Exception as e:  # noqa: BLE001
                logger.warning("检查集合是否存在失败: %s", e)
                return False
        try:
            self.client.get_collection(self.collection)
            return True
        except Exception:  # noqa: BLE001
            return False

    # 保留旧名，避免其他地方引用不到
    _collection_exists = collection_exists

    def recreate(self, dim: int) -> None:
        """删掉重建集合（重建索引时用）"""
        from qdrant_client.models import Distance, VectorParams

        self.dim = dim
        if self.collection_exists():
            logger.info("删除已有集合: %s", self.collection)
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("创建集合: %s (dim=%d)", self.collection, dim)

    # ── 写入 ──
    def upsert_batch(
        self,
        ids: list[str],
        vectors,
        payloads: list[dict[str, Any]],
        max_retries: int = 3,
    ) -> None:
        """写入一批点，失败自动重试。

        ⭐ 为什么要重试：
            低配服务器上 Qdrant 偶尔会响应慢（内存吃紧、磁盘 IO 抖动），
            一次超时就放弃会导致跑到一半的索引全废。
            重试 3 次 + 递增等待，能扛过绝大多数瞬时抖动。
        """
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=_to_uuid(i), vector=v.tolist(), payload=p)
            for i, v, p in zip(ids, vectors, payloads)
        ]

        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                self.client.upsert(collection_name=self.collection, points=points)
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                wait = 2 * attempt  # 2s, 4s, 6s
                if attempt < max_retries:
                    logger.warning(
                        "写入 Qdrant 失败（第 %d/%d 次）: %s —— %d 秒后重试",
                        attempt, max_retries, e, wait,
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"写入 Qdrant 连续失败 {max_retries} 次（timeout={self.timeout}s）。"
            f"建议：调小 retrieval.batch_size，或调大 retrieval.qdrant_timeout"
        ) from last_err

    # ── 检索 ⭐ 这里是版本兼容的关键 ──
    def search(self, vector, limit: int = 5) -> list[dict[str, Any]]:
        """返回 [{chunk_id, title, text, score}, ...]"""
        raw = vector.tolist()

        if hasattr(self.client, "query_points"):
            # 新版 API（qdrant-client 1.x 后期）
            resp = self.client.query_points(
                collection_name=self.collection,
                query=raw,
                limit=limit,
                with_payload=True,
            )
            hits = resp.points
        else:
            # 旧版 API
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=raw,
                limit=limit,
                with_payload=True,
            )

        return [
            {
                "chunk_id": (h.payload or {}).get("chunk_id", str(h.id)),
                "title": (h.payload or {}).get("title", ""),
                "text": (h.payload or {}).get("text", ""),
                "score": float(h.score),
            }
            for h in hits
        ]

    def count(self, max_retries: int = 3) -> int:
        """统计集合里的点数（带重试）"""
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                return int(self.client.count(self.collection, exact=True).count)
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < max_retries:
                    logger.warning("统计点数失败（第 %d/%d 次）: %s", attempt, max_retries, e)
                    time.sleep(2 * attempt)
        raise RuntimeError("统计 Qdrant 点数失败") from last_err

    def count_safe(self) -> int:
        """统计点数，失败返回 0（用于"续传时探测进度"，不该因为探测失败就中断）"""
        try:
            return self.count(max_retries=2)
        except Exception as e:  # noqa: BLE001
            logger.warning("探测已有点数失败，按 0 处理: %s", e)
            return 0


def _to_uuid(s: str) -> str:
    """Qdrant 只接受 UUID 或整数作为 id，这里把任意字符串转成稳定的 UUID"""
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, s))
