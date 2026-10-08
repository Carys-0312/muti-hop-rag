"""API 的请求 / 响应模型。

用 Pydantic 定义的好处：
- 自动校验（比如 strategy 只能是那三个值）
- 自动生成 OpenAPI 文档（访问 /docs 就能看到交互式文档）
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

StrategyName = Literal["single", "fixed_2hop", "adaptive"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    strategy: StrategyName = Field(default="adaptive", description="检索策略")


class AskResponse(BaseModel):
    question: str
    answer: str
    hop: int = Field(description="实际跳数")
    stop_reason: str = Field(description="为什么停止检索")
    queries: list[str] = Field(description="每一跳用的查询")
    retrieved_titles: list[str] = Field(description="检索到的文档标题（按分数排序）")
    cache_hits: int
    retrieval_calls: int


class HealthResponse(BaseModel):
    status: str
    version: str
    redis: dict[str, Any]
    vector_store: dict[str, Any]
    # ⭐ 启动期配置自检的结果（见 startup_check.py）
    #    配置错误会让 status = "degraded"，并在这里列出具体原因
    config_errors: list[str] = Field(default_factory=list)
    config_warnings: list[str] = Field(default_factory=list)


class ExampleQuestion(BaseModel):
    qid: str
    question: str
    type: str
    level: str
    answer: str


class StatsResponse(BaseModel):
    redis: dict[str, Any]
    strategies: list[str]
    config: dict[str, Any]
