"""图的组装：四个节点 + 一条条件边 + 一条回边。

这就是 LangGraph 存在的意义：
如果流程是直线（检索→生成），用 LangChain 就够；
只有当流程【带循环和条件分支】时，才需要 LangGraph 的状态机。
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..config import settings
from .nodes import generate_node, judge_node, retrieve_node, rewrite_node
from .state import MultiHopState


def route_after_judge(state: MultiHopState) -> str:
    """⭐ 条件边：决定"继续搜"还是"生成答案"。

    判断顺序（重要）：
      ① 硬性停止原因优先（no_new_evidence / query_duplicated / max_hop）
         —— 这些是"再搜也没用"的情况，必须尊重
      ② 未达到 min_hop → 强制继续搜
         —— 用于验证"judge 是否过早说够了"
      ③ judge 判断充分 → generate
      ④ 达到 max_hop → generate
      ⑤ 否则 → rewrite
    """
    # ① 硬性停止（judge_node 里已经判定的"再搜也没用"）
    if state.get("stop_reason") in ("no_new_evidence", "query_duplicated", "max_hop"):
        return "generate"

    # ② 最小跳数：强制至少搜够这么多次
    min_hop = settings.agent.get("min_hop", 1)
    if state["hop"] < min_hop:
        return "rewrite"

    # ③ judge 说够了
    jr = state.get("judge_result")
    if jr is not None and jr.sufficient:
        return "generate"

    # ④ 撞上限
    if state["hop"] >= settings.agent.max_hop:
        return "generate"

    return "rewrite"


def build_graph():
    builder = StateGraph(MultiHopState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("judge", judge_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "judge")

    # ⭐ 条件边：LangGraph 的灵魂
    builder.add_conditional_edges(
        "judge",
        route_after_judge,
        {"rewrite": "rewrite", "generate": "generate"},
    )

    # ⭐ 回边：形成循环（多跳的本质就是循环）
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", END)

    return builder.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH
