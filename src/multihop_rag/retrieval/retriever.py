"""检索器：问题 → top-k 段落。

【三种策略共用同一个检索器】——这是保证对比公平的关键：
S1 单跳 / S2 固定两跳 / S3 自适应多跳，只用同一个 retriever，
区别只在"编排方式"（见 agents/strategies.py）。
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import settings
from .embedder import get_embedder
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self):
        self.embedder = get_embedder()
        self.store = VectorStore(dim=self.embedder.dim)
        self.top_k = settings.retrieval.top_k

    def build_index(self, chunks: list[dict[str, Any]], resume: bool = True) -> None:
        """建立索引。

        ⭐ resume=True 时支持断点续传：
            低配服务器上建 5881 条的索引可能要 30~60 分钟，
            中途一旦出错（网络抖动、Qdrant 超时），
            如果每次都 recreate 从零开始，代价太大。
            chunk_id 是确定性 UUID，upsert 是幂等的，
            所以可以根据"已有条数"跳过已完成的批次，直接接着跑。
        """
        from tqdm import tqdm

        bs = settings.retrieval.batch_size
        start = 0

        if resume and self.store.collection_exists():
            existing = self.store.count_safe()
            if existing > 0:
                # 按批次对齐（upsert 幂等，重做最后一批无害）
                start = min((existing // bs) * bs, len(chunks))
                logger.info(
                    "检测到已有 %d 条索引，从第 %d 条继续（共 %d 条）",
                    existing, start, len(chunks),
                )

        if start == 0:
            self.store.recreate(dim=self.embedder.dim)

        for i in tqdm(range(start, len(chunks), bs), desc="建立索引"):
            batch = chunks[i : i + bs]
            vectors = self.embedder.encode_documents([c["text"] for c in batch])
            self.store.upsert_batch(
                ids=[c["chunk_id"] for c in batch],
                vectors=vectors,
                payloads=batch,
            )
        logger.info("索引完成，共 %d 条", self.store.count_safe())

    def search(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        vec = self.embedder.encode_query(query)
        return self.store.search(vec, limit=k or self.top_k)
