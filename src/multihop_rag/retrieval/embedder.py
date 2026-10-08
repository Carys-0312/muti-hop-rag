"""Embedding 模型封装 —— 用 FastEmbed（ONNX Runtime）。

为什么不用 sentence-transformers：
    它会连带安装 PyTorch，Windows 上首次安装约 2.7 GB，非常慢且容易失败。
    FastEmbed 用 ONNX Runtime 跑【同一批模型】（包括 BAAI/bge-small-en-v1.5），
    依赖体积小一个数量级（约 50 MB），CPU 上反而更快。

BGE 系列的用法要点（容易被忽略的细节）：
    检索任务中，query 侧要加指令前缀（"Represent this sentence for ..."），
    文档侧不加。
    → fastembed 的 query_embed() 会按模型内置配置自动处理这个前缀，
      所以【文档用 embed()、查询用 query_embed()】，不要混用。
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2 归一化 → 之后内积就等于余弦相似度"""
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


class Embedder:
    def __init__(self, model_name: str | None = None):
        from fastembed import TextEmbedding

        self.model_name = model_name or settings.retrieval.embedding_model
        logger.info("加载 embedding 模型（fastembed / ONNX）: %s", self.model_name)

        # 首次使用会自动从 HuggingFace 下载 ONNX 模型（约 130 MB）
        self.model = TextEmbedding(model_name=self.model_name)
        self.dim = self._probe_dim()
        logger.info("模型就绪，向量维度 = %d", self.dim)

    def _probe_dim(self) -> int:
        """用一个探针文本问出向量维度"""
        return len(next(iter(self.model.embed(["dimension probe"]))))

    # ── 文档侧：用 embed()，不加指令前缀 ──
    def encode_documents(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        vecs = list(
            self.model.embed(
                texts,
                batch_size=batch_size or settings.retrieval.batch_size,
            )
        )
        arr = np.asarray(vecs, dtype=np.float32)
        return np.stack([_normalize(v) for v in arr])

    # ── Query 侧：用 query_embed()，模型会自动加指令前缀 ──
    def encode_query(self, query: str) -> np.ndarray:
        vec = next(iter(self.model.query_embed([query])))
        return _normalize(np.asarray(vec, dtype=np.float32))


@lru_cache(maxsize=2)
def get_embedder(model_name: str | None = None) -> Embedder:
    return Embedder(model_name)
