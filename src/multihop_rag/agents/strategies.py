"""三种检索策略 —— 本项目的对比实验主体。

⭐ 核心原则：控制变量。
   三种策略【共用同一个检索器、同一个生成器、同一份索引、同一个 top_k】，
   唯一不同的是"检索的编排方式"。
   这样数字对比才有意义。

   S1 单跳        → 检索一次就生成（基线）
   S2 固定两跳    → 强制检索两次（"傻"的对照，防止自我欺骗）
   S3 自适应多跳  → LangGraph 判断够不够，动态决定跳数（本项目亮点）
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ..config import settings
from ..generation import generator
from ..runtime import get_cache, get_retriever
from .graph import get_graph
from .state import Evidence, initial_state
from .utils import dedup_evidences, extract_titles, format_evidences, normalize_query

logger = logging.getLogger(__name__)


def _to_evidence(hit: dict, query: str, hop: int) -> Evidence:
    return Evidence(
        chunk_id=hit["chunk_id"],
        title=hit["title"],
        text=hit["text"],
        score=hit["score"],
        from_query=query,
        hop=hop,
    )


def _cached_search(query: str) -> tuple[list[dict], int]:
    """带缓存的检索，返回 (hits, 是否命中缓存)"""
    cache = get_cache()
    cached = cache.get(query)
    if cached is not None:
        return cached, 1
    hits = get_retriever().search(query)
    cache.set(query, hits)
    return hits, 0


def _result(
    question: str,
    answer_raw: str,
    evidences: list[Evidence],
    queries: list[str],
    hop: int,
    stop_reason: str,
    cache_hits: int,
    retrieval_calls: int,
    **extra: Any,
) -> dict:
    """统一的返回结构，评测层直接消费。"""
    evidences = dedup_evidences(evidences)
    return {
        "question": question,
        "answer_raw": answer_raw,
        "answer": generator.parse_answer(answer_raw),
        "retrieved_titles": extract_titles(evidences),
        "evidences": [e.model_dump() for e in evidences],
        "queries": queries,
        "hop": hop,
        "stop_reason": stop_reason,
        "cache_hits": cache_hits,
        "retrieval_calls": retrieval_calls,
        **extra,
    }


# ══════════════════════════════════════════════════════════
#  S1：单跳（基线）
# ══════════════════════════════════════════════════════════
def run_single_hop(question: str) -> dict:
    hits, cache_hit = _cached_search(question)
    evidences = [_to_evidence(h, question, hop=1) for h in hits]

    answer = generator.generate(
        question=question,
        evidences_text=format_evidences(evidences),
    )
    return _result(
        question, answer, evidences, [question], hop=1, stop_reason="single_hop",
        cache_hits=cache_hit, retrieval_calls=1 - cache_hit,
    )


# ══════════════════════════════════════════════════════════
#  S2：固定两跳（不做判断，强制搜两次）
# ══════════════════════════════════════════════════════════
def run_fixed_2hop(question: str) -> dict:
    cache_hits = retrieval_calls = 0

    # 第 1 跳
    hits1, hit1 = _cached_search(question)
    cache_hits += hit1
    retrieval_calls += 1 - hit1
    ev1 = [_to_evidence(h, question, hop=1) for h in hits1]

    # 改写（复用同一个改写 prompt，但【不做判断】）
    q2 = generator.simple_rewrite(
        question=question,
        evidences_text=format_evidences(ev1, max_chars=1500),
    ).strip()

    queries = [question]
    evidences = list(ev1)

    # 第 2 跳
    if q2 and normalize_query(q2) != normalize_query(question):
        queries.append(q2)
        hits2, hit2 = _cached_search(q2)
        cache_hits += hit2
        retrieval_calls += 1 - hit2
        evidences += [_to_evidence(h, q2, hop=2) for h in hits2]

    answer = generator.generate(
        question=question,
        evidences_text=format_evidences(evidences),
    )
    return _result(
        question, answer, evidences, queries, hop=len(queries), stop_reason="fixed_2hop",
        cache_hits=cache_hits, retrieval_calls=retrieval_calls,
    )


# ══════════════════════════════════════════════════════════
#  S3：自适应多跳（LangGraph）
# ══════════════════════════════════════════════════════════
def run_adaptive(question: str) -> dict:
    graph = get_graph()
    final = graph.invoke(
        initial_state(question),
        config={"recursion_limit": 20},   # 保险：防止图设计有 bug 时死循环
    )
    return _result(
        question,
        final["answer"],
        final["evidences"],
        final["queries"],
        hop=final["hop"],
        stop_reason=final["stop_reason"] or "finished",
        cache_hits=final["cache_hits"],
        retrieval_calls=final["retrieval_calls"],
        judge_sufficient=(
            final["judge_result"].sufficient if final["judge_result"] else None
        ),
        judge_reasoning=(
            final["judge_result"].reasoning if final["judge_result"] else ""
        ),
        new_evidence_counts=final["new_evidence_counts"],
        used_entities=final["used_entities"],
    )


# ══════════════════════════════════════════════════════════
#  统一入口
# ══════════════════════════════════════════════════════════
STRATEGIES: dict[str, Callable[[str], dict]] = {
    "single": run_single_hop,
    "fixed_2hop": run_fixed_2hop,
    "adaptive": run_adaptive,
}

STRATEGY_LABELS = {
    "single": "S1 单跳",
    "fixed_2hop": "S2 固定两跳",
    "adaptive": "S3 自适应多跳",
}


def run(question: str, strategy: str) -> dict:
    if strategy not in STRATEGIES:
        raise ValueError(f"未知策略: {strategy}（可选: {list(STRATEGIES)}）")
    return STRATEGIES[strategy](question)
