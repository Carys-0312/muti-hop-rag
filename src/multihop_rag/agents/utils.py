"""Agent 层的小工具函数。"""

from __future__ import annotations

from ..config import settings
from .state import Evidence


def format_evidences(
    evidences: list[Evidence],
    max_chars: int | None = None,
    snippet_chars: int | None = None,
    numbered: bool = True,
) -> str:
    """把证据列表格式化成塞进 prompt 的文本。

    为什么要截断：多跳会累积很多证据，不截断会爆 token。
    """
    if not evidences:
        return "（暂无证据）"

    max_chars = max_chars or settings.agent.max_evidence_chars
    snippet_chars = snippet_chars or settings.agent.evidence_snippet_chars

    lines: list[str] = []
    used = 0
    for i, ev in enumerate(evidences, start=1):
        text = ev.text[:snippet_chars].replace("\n", " ")
        prefix = f"[{i}] " if numbered else ""
        block = f"{prefix}标题：{ev.title}\n内容：{text}"
        if used + len(block) > max_chars:
            lines.append(f"...（还有 {len(evidences) - i + 1} 条证据被省略）")
            break
        lines.append(block)
        used += len(block)
    return "\n\n".join(lines)


def normalize_query(q: str) -> str:
    """归一化查询，用于判断"是否重复"和缓存 key"""
    return " ".join(q.lower().strip().split())


def dedup_evidences(evidences: list[Evidence]) -> list[Evidence]:
    """按 chunk_id 去重，保留首次出现的（即最早跳数搜到的）"""
    seen: set[str] = set()
    out: list[Evidence] = []
    for ev in evidences:
        if ev.chunk_id in seen:
            continue
        seen.add(ev.chunk_id)
        out.append(ev)
    return out


def extract_titles(evidences: list[Evidence], top_n: int | None = None) -> list[str]:
    """按分数排序取出标题列表（算召回率时用）"""
    ranked = sorted(evidences, key=lambda e: e.score, reverse=True)
    titles = [e.title for e in ranked]
    return titles[:top_n] if top_n else titles
