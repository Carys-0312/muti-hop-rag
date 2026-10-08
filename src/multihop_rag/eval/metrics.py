"""评测指标。

分两类，分开测才能定位问题：
  检索层 → 有没有找到正确文档？（Evidence Recall / Full Recall / MRR）
  生成层 → 答得对不对？（EM / F1）

⭐ 关键区别：
  Evidence Recall = 找到了多少比例的金标准文档（部分命中也算）
  Full Recall     = 金标准文档【全部】找到才算命中
  → 多跳的价值恰恰体现在 Full Recall 上，因为多跳就是为了"凑齐"
"""

from __future__ import annotations

import re
import string
from collections import Counter


# ══════════════════════════════════════════════════════════
#  生成层指标
# ══════════════════════════════════════════════════════════
def normalize_answer(s: str) -> str:
    """HotpotQA 官方风格的归一化。

    不做归一化的话，EM 会虚低（"the United States" vs "United States"）。
    """
    if not s:
        return ""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text: str) -> str:
        return text.lower()

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ══════════════════════════════════════════════════════════
#  检索层指标
# ══════════════════════════════════════════════════════════
def evidence_recall(retrieved: list[str], gold: list[str], k: int = 5) -> float:
    """top-k 里命中了多少比例的金标准文档（部分命中给部分分）"""
    if not gold:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & set(gold)) / len(gold)


def full_recall(retrieved: list[str], gold: list[str], k: int = 5) -> float:
    """金标准文档是否【全部】出现在 top-k 里（0 或 1）"""
    if not gold:
        return 0.0
    return float(set(gold) <= set(retrieved[:k]))


def mrr(retrieved: list[str], gold: list[str]) -> float:
    """第一个正确文档的排名倒数"""
    gold_set = set(gold)
    for rank, title in enumerate(retrieved, start=1):
        if title in gold_set:
            return 1.0 / rank
    return 0.0


# ══════════════════════════════════════════════════════════
#  汇总
# ══════════════════════════════════════════════════════════
def compute_metrics(records: list[dict], k: int = 5) -> dict:
    """records: [{predicted, gold_answer, retrieved_titles, gold_titles, hop, ...}]"""
    n = len(records)
    if n == 0:
        return {}

    sums = {
        "evidence_recall": 0.0,
        "full_recall": 0.0,
        "mrr": 0.0,
        "em": 0.0,
        "f1": 0.0,
        "hop": 0.0,
    }

    for r in records:
        sums["evidence_recall"] += evidence_recall(r["retrieved_titles"], r["gold_titles"], k)
        sums["full_recall"] += full_recall(r["retrieved_titles"], r["gold_titles"], k)
        sums["mrr"] += mrr(r["retrieved_titles"], r["gold_titles"])
        sums["em"] += exact_match(r["predicted"], r["gold_answer"])
        sums["f1"] += f1_score(r["predicted"], r["gold_answer"])
        sums["hop"] += r.get("hop", 0)

    out = {f"{name}@{k}" if name in ("evidence_recall", "full_recall") else name: total / n
           for name, total in sums.items()}
    out["num_samples"] = n
    return {key: (round(v, 4) if isinstance(v, float) else v) for key, v in out.items()}


def compute_breakdown(records: list[dict], key: str, k: int = 5) -> dict[str, dict]:
    """按某个字段分组统计，比如按 type / level 分组。

    ⭐ 这是"分场景分析"的实现，面试时用它说明"哪类问题提升最大"。
    """
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r.get(key, "unknown"), []).append(r)
    return {g: compute_metrics(rs, k) for g, rs in sorted(groups.items())}
