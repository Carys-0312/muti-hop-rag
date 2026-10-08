"""冒烟测试：验证评测指标逻辑（纯标准库，不需要额外依赖）。

用法：
    python scripts/smoke_test_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multihop_rag.eval.metrics import (  # noqa: E402
    compute_breakdown,
    compute_metrics,
    evidence_recall,
    exact_match,
    f1_score,
    full_recall,
    mrr,
    normalize_answer,
)


def main() -> None:
    print("=== 1. 答案归一化 ===")
    cases = ["The United States!", "  A  yes ", "YES"]
    for c in cases:
        print(f"  {c!r:24} -> {normalize_answer(c)!r}")

    print("\n=== 2. EM / F1 ===")
    print(f"  EM 完全相同          : {exact_match('yes', 'yes')}")
    print(f"  EM 冠词+大小写+标点   : {exact_match('The Yes!', 'yes')}")
    print(f"  F1 部分匹配          : {f1_score('Leonid Levin', 'Leonid Anatolievich Levin'):.3f}")
    print(f"  F1 完全相同          : {f1_score('yes', 'yes'):.3f}")
    print(f"  F1 完全不匹配        : {f1_score('no', 'yes'):.3f}")

    print("\n=== 3. 检索指标 ===")
    gold = ["Pavel Urysohn", "Leonid Levin"]
    scenarios = {
        "两个都进 top5": ["Leonid Levin", "Foo", "Pavel Urysohn", "Bar", "Baz"],
        "只进一个": ["Leonid Levin", "Foo", "Bar", "Baz", "Qux"],
        "一个都没进": ["X", "Y", "Z", "W", "V"],
        "命中但排在后面": ["X", "Y", "Z", "W", "Leonid Levin"],
    }
    print(f"  金标准 = {gold}")
    for name, got in scenarios.items():
        er = evidence_recall(got, gold, k=5)
        fr = full_recall(got, gold, k=5)
        mr = mrr(got, gold)
        print(f"  {name:12} EvidenceRecall={er:.2f}  FullRecall={fr:.0f}  MRR={mr:.3f}")

    print("\n=== 4. 汇总指标 ===")
    records = [
        {
            "predicted": "yes",
            "gold_answer": "yes",
            "retrieved_titles": scenarios["两个都进 top5"],
            "gold_titles": gold,
            "hop": 2,
            "type": "comparison",
            "level": "hard",
        },
        {
            "predicted": "no",
            "gold_answer": "yes",
            "retrieved_titles": scenarios["只进一个"],
            "gold_titles": gold,
            "hop": 3,
            "type": "bridge",
            "level": "medium",
        },
    ]
    metrics = compute_metrics(records, k=5)
    for key, val in metrics.items():
        print(f"  {key:>20}: {val}")

    print("\n=== 5. 分层统计 ===")
    by_level = compute_breakdown(records, "level", k=5)
    by_type = compute_breakdown(records, "type", k=5)
    print(f"  按难度 FullRecall@5: { {k: v['full_recall@5'] for k, v in by_level.items()} }")
    print(f"  按类型 FullRecall@5: { {k: v['full_recall@5'] for k, v in by_type.items()} }")

    print("\n✅ 全部指标逻辑正常")


if __name__ == "__main__":
    main()
