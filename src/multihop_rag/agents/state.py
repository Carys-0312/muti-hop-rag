"""LangGraph 的状态定义。

⭐ 设计原则：所有"错误分析需要的数据"都要在这里留下字段。
   否则跑完实验才发现没有数据可分析，只能重跑。
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """一条检索到的证据"""

    chunk_id: str
    title: str
    text: str
    score: float
    # ── 下面两个字段是"埋点"，专门为错误分析准备 ──
    from_query: str = Field(default="", description="这条证据是哪个查询搜出来的")
    hop: int = Field(default=0, description="第几跳搜到的")


class JudgeResult(BaseModel):
    """judge 节点的结构化输出。

    用 Pydantic 约束而不是让 LLM 输出自由文本，好处：
    1. 直接拿到 bool，不用写正则解析
    2. 解析失败可以被捕获并计入错误统计
    """

    sufficient: bool = Field(description="现有证据是否足以回答问题")
    missing: str = Field(description="还缺少什么信息；若已充分则为空字符串")
    reasoning: str = Field(description="判断理由，一句话")


class RewriteResult(BaseModel):
    """rewrite 节点的结构化输出"""

    query: str = Field(description="新的检索查询，简短的关键词组合")
    used_entities: list[str] = Field(
        default_factory=list,
        description="查询中使用的、来自已有证据的实体名",
    )


class MultiHopState(TypedDict):
    # ── 输入 ──
    question: str

    # ── 累积型：Annotated[..., operator.add] 表示节点返回时【追加】而不是覆盖 ──
    queries: Annotated[list[str], operator.add]
    evidences: Annotated[list[Evidence], operator.add]
    # 每一跳新搜到几条证据（用来判断"再搜也没用" + 统计）
    new_evidence_counts: Annotated[list[int], operator.add]

    # ── 覆盖型：节点返回就整体覆盖 ──
    hop: int
    judge_result: JudgeResult | None
    answer: str
    stop_reason: str          # ⭐ 为什么停止检索（错误分析要用）

    # ── 统计埋点（用于量化缓存收益）──
    cache_hits: int
    retrieval_calls: int
    used_entities: Annotated[list[str], operator.add]


def initial_state(question: str) -> MultiHopState:
    """构造初始状态。注意 queries 先放原问题——第一跳就用它搜。"""
    return MultiHopState(
        question=question,
        queries=[question],
        evidences=[],
        new_evidence_counts=[],
        hop=0,
        judge_result=None,
        answer="",
        stop_reason="",
        cache_hits=0,
        retrieval_calls=0,
        used_entities=[],
    )
