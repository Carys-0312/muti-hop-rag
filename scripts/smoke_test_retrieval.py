"""检索冒烟测试 —— 不需要 LLM / API Key，只验证索引能正常工作。

用法：
    python scripts/smoke_test_retrieval.py
    python scripts/smoke_test_retrieval.py --k 5

它会：
  1. 检查 Qdrant 里的文档条数
  2. 拿几个真实问题去检索，打印命中的文档标题
  3. 报告「金标准是否被检索到」—— 这就是后面算召回率的雏形
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# 关掉噪音日志
import logging  # noqa: E402

for _n in ("httpx", "httpcore", "urllib3", "huggingface_hub", "datasets", "qdrant_client"):
    logging.getLogger(_n).setLevel(logging.WARNING)


DEMO_QUESTIONS = [
    {
        "question": "Were Pavel Urysohn and Leonid Levin born in the same country?",
        "gold_titles": ["Pavel Urysohn", "Leonid Levin"],
    },
    {
        "question": "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
        "gold_titles": ["Laleli Mosque", "Esma Sultan Mansion"],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="每次取几条")
    parser.add_argument("--file", default=None, help="改用评测集文件里的问题")
    args = parser.parse_args()

    from multihop_rag.config import settings
    from multihop_rag.runtime import get_cache, get_retriever

    print("=" * 72)
    print(" 检索冒烟测试（不需要 API Key）")
    print("=" * 72)

    # ── 1. 连接检查 ──
    print("\n【1】连接检查")
    cache = get_cache()
    print(f"  Redis 缓存 : {'✅ 已连接' if cache.enabled else '⚠️ 未启用'}")

    retriever = get_retriever()
    total = retriever.store.count()
    print(f"  Qdrant 集合: {settings.retrieval.collection}")
    print(f"  文档条数   : {total}")

    if total == 0:
        print("\n❌ 索引是空的！先跑：python scripts/build_index.py --download")
        sys.exit(1)

    # ── 2. 准备测试问题 ──
    if args.file:
        path = Path(args.file)
        questions = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()][:5]
        questions = [{"question": q["question"], "gold_titles": q["gold_titles"]} for q in questions]
        print(f"\n【2】从 {path.name} 取 {len(questions)} 个问题")
    else:
        questions = DEMO_QUESTIONS
        print(f"\n【2】使用 {len(questions)} 个内置演示问题")

    # ── 3. 检索 ──
    print(f"\n【3】检索（top-{args.k}）")
    hits_total = 0
    for i, item in enumerate(questions, 1):
        print(f"\n  ── 问题 {i} ──")
        print(f"  {item['question']}")

        hits = retriever.search(item["question"], k=args.k)
        got = [h["title"] for h in hits]
        gold = set(item["gold_titles"])
        hit = gold & set(got)

        if item["gold_titles"]:
            print(f"  金标准       : {item['gold_titles']}")
        for rank, h in enumerate(hits, 1):
            mark = "✅" if h["title"] in gold else "  "
            print(f"  {mark} [{rank}] {h['score']:.3f}  {h['title']}")

        if gold:
            rate = len(hit) / len(gold)
            full = "✅ 全中" if gold <= set(got) else f"❌ 缺 {sorted(gold - set(got))}"
            print(f"  召回率       : {rate:.2f}   {full}")
            hits_total += 1 if gold <= set(got) else 0

    # ── 4. 汇总 ──
    print("\n" + "=" * 72)
    if hits_total:
        print(f"  全召回（Full Recall@{args.k}）: {hits_total}/{len(questions)}"
              f" = {hits_total / len(questions):.2f}")
    print("  ✅ 检索链路正常")
    print("=" * 72)
    print("\n下一步：")
    print("  1. 配置 .env 里的 DEEPSEEK_API_KEY（如果要跑问答/实验）")
    print("  2. python scripts/build_eval_set.py        # 构建评测集")
    print("  3. docker compose exec api python -m multihop_rag.cli ask \"...\" --verbose")


if __name__ == "__main__":
    main()
