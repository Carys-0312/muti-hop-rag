"""快速查看已有评测集的分布（不加载数据集，秒出）。

用法：
    python scripts/show_eval_set.py
    python scripts/show_eval_set.py --n 5     # 多看几条示例
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multihop_rag.config import settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="打印几条示例")
    args = parser.parse_args()

    path = settings.eval_set_path
    if not path.exists():
        print(f"❌ 评测集不存在: {path}")
        print("   先跑：python scripts/build_eval_set.py --force")
        sys.exit(1)

    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    print("=" * 72)
    print(f" 评测集：{path}")
    print("=" * 72)
    print(f"\n总数: {len(rows)}")
    print(f"类型: {dict(Counter(r['type'] for r in rows))}")
    print(f"难度: {dict(Counter(r['level'] for r in rows))}")

    both = Counter(f"{r['type']}/{r['level']}" for r in rows)
    print(f"\n细分（6 个组合）:")
    for key in sorted(both):
        n = both[key]
        bar = "█" * min(n, 40)
        flag = "✅" if n >= 10 else "⚠️"
        print(f"  {key:<24} {n:>4}  {flag} {bar}")

    thin = {k: v for k, v in both.items() if v < 10}
    if thin:
        print(f"\n⚠️ 样本偏少的组合: {thin}")
    else:
        print("\n✅ 所有组合都有 ≥10 条，可以做分场景分析")

    print(f"\n前 {args.n} 条示例:")
    for r in rows[: args.n]:
        print(f"  [{r['qid']}] ({r['type']}/{r['level']})")
        print(f"        Q: {r['question']}")
        print(f"        A: {r['answer']}   需检索: {r['gold_titles']}")


if __name__ == "__main__":
    main()
