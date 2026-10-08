"""检索结果缓存（Redis）。

在多跳场景下的价值：
- 多跳会反复检索相似/相同的查询 → 缓存能省下重复的向量检索
- 缓存命中率、检索次数下降，都是可以量化的指标

设计要点：
1. key 用【归一化后的 query】做 hash，避免大小写/空格差异导致缓存穿透
2. Redis 连不上时优雅降级（不阻塞主流程）
3. 提供 stats() 用于统计命中率
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)


class RetrievalCache:
    def __init__(self, url: str | None = None, ttl: int | None = None, enabled: bool | None = None):
        self.enabled = settings.cache_enabled if enabled is None else enabled
        self.ttl = ttl or settings.cache.ttl
        self.namespace = settings.cache.namespace
        self._client = None

        # 本地计数（不依赖 Redis 的全局统计，方便单次实验）
        self.local_hits = 0
        self.local_misses = 0

        if self.enabled:
            try:
                import redis

                self._client = redis.from_url(settings.redis_url if url is None else url,
                                              decode_responses=True,
                                              socket_connect_timeout=2)
                self._client.ping()
                logger.info("Redis 缓存已连接: %s", settings.redis_url)
            except Exception as e:  # noqa: BLE001
                logger.warning("Redis 连接失败，缓存自动禁用: %s", e)
                self.enabled = False

    @staticmethod
    def _normalize(query: str) -> str:
        return " ".join(query.lower().split())

    def _key(self, query: str) -> str:
        h = hashlib.sha1(self._normalize(query).encode()).hexdigest()[:16]
        return f"{self.namespace}:retrieval:{h}"

    def get(self, query: str) -> list[dict[str, Any]] | None:
        if not self.enabled:
            return None
        raw = self._client.get(self._key(query))
        if raw is None:
            self.local_misses += 1
            return None
        self.local_hits += 1
        return json.loads(raw)

    def set(self, query: str, hits: list[dict[str, Any]]) -> None:
        if not self.enabled:
            return
        self._client.setex(
            self._key(query), self.ttl, json.dumps(hits, ensure_ascii=False)
        )

    def clear(self) -> None:
        """清掉本项目所有缓存 key（跑对比实验前清一次，保证公平）"""
        if not self.enabled:
            return
        for key in self._client.scan_iter(f"{self.namespace}:retrieval:*"):
            self._client.delete(key)
        self.local_hits = self.local_misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.local_hits + self.local_misses
        return self.local_hits / total if total else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hits": self.local_hits,
            "misses": self.local_misses,
            "hit_rate": round(self.hit_rate, 4),
        }
