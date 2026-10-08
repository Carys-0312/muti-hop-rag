"""LangGraph 的四个节点。

⭐ 每个节点都做三件事：读 state → 干活 → 返回【增量更新】字典。
⭐ 所有节点都埋了统计点，跑完就能算指标，不用额外写统计代码。
"""

from __future__ import annotations

import logging

from ..config import settings
from ..generation import generator
from ..retrieval.retriever import Retriever
from ..runtime import get_cache, get_retriever
from .state import Evidence, JudgeResult, MultiHopState
from .utils import format_evidences, normalize_query

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  节点 1：retrieve —— 用最新查询检索，去重后加入证据池
# ══════════════════════════════════════════════════════════
def retrieve_node(state: MultiHopState) -> dict:
    query = state["queries"][-1]        # 取最新查询（第一跳时就是原问题）
    hop = state["hop"] + 1

    # ── Redis 缓存 ──
    cache = get_cache()
    cached = cache.get(query)
    if cached is not None:
        hits, cache_hit = cached, 1
    else:
        hits = get_retriever().search(query)
        cache.set(query, hits)
        cache_hit = 0

    # ── ⭐ 去重：排除已经收集过的 chunk，否则证据池会被重复内容塞满 ──
    seen = {e.chunk_id for e in state["evidences"]}
    new_hits = [
        Evidence(
            chunk_id=h["chunk_id"],
            title=h["title"],
            text=h["text"],
            score=h["score"],
            from_query=query,
            hop=hop,
        )
        for h in hits
        if h["chunk_id"] not in seen
    ]

    return {
        "evidences": new_hits,
        "new_evidence_counts": [len(new_hits)],
        "hop": hop,
        "cache_hits": state["cache_hits"] + cache_hit,
        "retrieval_calls": state["retrieval_calls"] + (1 - cache_hit),
    }


# ══════════════════════════════════════════════════════════
#  节点 2：judge —— 判断证据是否充分（本项目的技术核心）
# ══════════════════════════════════════════════════════════
def judge_node(state: MultiHopState) -> dict:
    max_hop = settings.agent.max_hop

    # ── 硬性停止条件优先，省一次 LLM 调用 ──
    if state["hop"] >= max_hop:
        return {
            "judge_result": JudgeResult(
                sufficient=True, missing="", reasoning=f"已达最大跳数 {max_hop}"
            ),
            "stop_reason": "max_hop",
        }

    # 上一跳一条新证据都没搜到 → 再跳也没意义
    if state["new_evidence_counts"] and state["new_evidence_counts"][-1] == 0:
        return {
            "judge_result": JudgeResult(
                sufficient=True, missing="", reasoning="上一跳未检索到新证据"
            ),
            "stop_reason": "no_new_evidence",
        }

    # ── 让 LLM 判断 ──
    result = generator.judge(
        question=state["question"],
        evidences_text=format_evidences(state["evidences"]),
    )
    logger.debug("judge: sufficient=%s reason=%s", result.sufficient, result.reasoning)

    return {
        "judge_result": result,
        "stop_reason": "sufficient" if result.sufficient else "",
    }


# ══════════════════════════════════════════════════════════
#  节点 3：rewrite —— 生成新的检索查询
# ══════════════════════════════════════════════════════════
def rewrite_node(state: MultiHopState) -> dict:
    jr = state["judge_result"]

    # ⭐ 当 min_hop 强制继续、但 judge 已说"充分"时，missing 会是空的。
    #    这时给改写器一个兜底指令，让它去找"补充证据"而不是卡住。
    missing = (jr.missing if jr else "") or ""
    if not missing.strip():
        missing = "（证据被判定为已充分，但策略要求继续补充更多相关背景，以便更准确地回答）"

    result = generator.rewrite(
        question=state["question"],
        evidences_text=format_evidences(state["evidences"], max_chars=2000),
        missing=missing,
        used_queries=state["queries"],
    )

    new_query = (result.query or "").strip()
    used_norm = {normalize_query(q) for q in state["queries"]}

    # ⭐ 停止条件三：改写不出新查询 → 说明已经搜到头了
    if not new_query or normalize_query(new_query) in used_norm:
        logger.info("改写重复，停止检索: %r", new_query)
        return {
            "queries": [],
            "used_entities": result.used_entities or [],
            "judge_result": (
                state["judge_result"].model_copy(
                    update={"sufficient": True, "reasoning": "改写查询与历史重复"}
                )
                if state["judge_result"]
                else None
            ),
            "stop_reason": "query_duplicated",
        }

    logger.info("改写查询: %r → %r", state["queries"][-1], new_query)
    return {"queries": [new_query], "used_entities": result.used_entities or []}


# ══════════════════════════════════════════════════════════
#  节点 4：generate —— 根据证据生成最终答案
# ══════════════════════════════════════════════════════════
def generate_node(state: MultiHopState) -> dict:
    raw = generator.generate(
        question=state["question"],
        evidences_text=format_evidences(state["evidences"], max_chars=3000),
    )
    return {"answer": raw}
