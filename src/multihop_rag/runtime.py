"""运行时依赖（懒加载单例）。

为什么要单独一个文件：
避免 nodes.py / strategies.py / scripts 之间循环导入，
同时保证 embedding 模型、Redis 连接、LLM 客户端只初始化一次。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from .config import settings

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

    from .retrieval.cache import RetrievalCache
    from .retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_retriever() -> "Retriever":
    from .retrieval.retriever import Retriever

    return Retriever()


@lru_cache(maxsize=1)
def get_cache() -> "RetrievalCache":
    from .retrieval.cache import RetrievalCache

    return RetrievalCache()


@lru_cache(maxsize=1)
def get_llm() -> "ChatOpenAI":
    from langchain_openai import ChatOpenAI

    logger.info("初始化 LLM: %s @ %s", settings.llm_model, settings.llm_base_url)
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=settings.agent.temperature,  # ⭐ 判断任务用 0，保证可复现
        timeout=60,
        max_retries=2,
    )


def reset_dependencies() -> None:
    """清空单例（换配置 / 跑对比实验时用）"""
    get_retriever.cache_clear()
    get_cache.cache_clear()
    get_llm.cache_clear()
