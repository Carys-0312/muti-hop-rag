"""构建评测集（如果还没有）。

用法：
    python scripts/build_eval_set.py            # 默认 200 条
    python scripts/build_eval_set.py --size 20  # 先做个小样本跑通流程
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=None, help="评测集规模（默认取配置值 200）")
    parser.add_argument("--force", action="store_true", help="已存在也重建")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from multihop_rag.config import settings
    from multihop_rag.data.build_eval_set import build_eval_set, save_eval_set
    from multihop_rag.data.load_hotpotqa import load_raw

    if settings.eval_set_path.exists() and not args.force:
        print(f"评测集已存在: {settings.eval_set_path}")
        print("如需重建，加 --force")
        return

    ds = load_raw()   # split 从 config 读（默认 train）
    eval_set = build_eval_set(ds, size=args.size)
    save_eval_set(eval_set)

    from collections import Counter

    print(f"\n✅ 评测集规模: {len(eval_set)}")
    print(f"   类型: {dict(Counter(r['type'] for r in eval_set))}")
    print(f"   难度: {dict(Counter(r['level'] for r in eval_set))}")

    # ⭐ 细分到 (类型 × 难度) 的 6 个组合，确认每一类都有足够样本
    both = Counter(f"{r['type']}/{r['level']}" for r in eval_set)
    print(f"   细分: {dict(sorted(both.items()))}")
    thin = {k: v for k, v in both.items() if v < 10}
    if thin:
        print(f"   ⚠️ 以下组合样本偏少（<10）：{thin}")
        print(f"      → 可把 configs/default.yaml 的 data.corpus_questions 调大")

    print("\n前 3 条示例:")
    for r in eval_set[:3]:
        print(f"  [{r['qid']}] ({r['type']}/{r['level']}) {r['question'][:68]}")
        print(f"        答案={r['answer']}  需检索={r['gold_titles']}")

    print(f"\n   路径: {settings.eval_set_path}")


if __name__ == "__main__":
    main()
