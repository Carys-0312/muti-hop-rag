"""端点演示：从【评测集】里抽几条问题跑一遍，并对比金标准。

⚠️ 为什么必须用评测集里的问题：
    索引是用「和评测集同一批的 600 个问题」的候选文档建的。
    如果随手用别的问题（比如 HotpotQA 官方 dev 集的示例），
    它的金标准文档可能根本不在索引里 → 永远检索不到 → 答案必然错。

用法：
    python scripts/demo.py                    # 跑 3 条（跨类型/难度）
    python scripts/demo.py --n 5
    python scripts/demo.py --strategy single  # 换策略
    python scripts/demo.py --qid 0007         # 指定某一条
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import logging  # noqa: E402

for _n in ("httpx", "httpcore", "urllib3", "huggingface_hub", "datasets",
           "qdrant_client", "httpx2"):
    logging.getLogger(_n).setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

from multihop_rag.agents import strategies  # noqa: E402
from multihop_rag.config import settings  # noqa: E402
from multihop_rag.eval.metrics import evidence_recall, exact_match, f1_score, full_recall  # noqa: E402


def pick_diverse(rows: list[dict], n: int) -> list[dict]:
    """按 (type, level) 尽量分散地挑 n 条"""
    buckets: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        buckets.setdefault((r["type"], r["level"]), []).append(r)

    keys = sorted(buckets, key=lambda k: -len(buckets[k]))
    picked: list[dict] = []
    i = 0
    while len(picked) < n and any(buckets[k] for k in keys):
        key = keys[i % len(keys)]
        if buckets[key]:
            picked.append(buckets[key].pop(0))
        i += 1
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--strategy", default="adaptive",
                        choices=["single", "fixed_2hop", "adaptive"])
    parser.add_argument("--qid", default=None, help="只跑指定 qid")
    parser.add_argument("--k", type=int, default=5, help="算 Recall@k 的 k")
    args = parser.parse_args()

    path = settings.eval_set_path
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    if args.qid:
        targets = [r for r in rows if r["qid"] == args.qid]
        if not targets:
            print(f"❌ 找不到 qid={args.qid}")
            sys.exit(1)
    else:
        targets = pick_diverse(rows, args.n)

    print("=" * 76)
    print(f" 端点演示  策略={strategies.STRATEGY_LABELS[args.strategy]}  "
          f"问题数={len(targets)}")
    print("=" * 76)

    em_sum = f1_sum = fr_sum = 0.0
    latencies = []

    for i, item in enumerate(targets, 1):
        gold = item["gold_titles"]
        print(f"\n{'─' * 76}")
        print(f"[{i}/{len(targets)}] qid={item['qid']}  类型={item['type']}  "
              f"难度={item['level']}")
        print(f"Q: {item['question']}")
        print(f"金标准答案: {item['answer']}")
        print(f"应检索到  : {gold}")

        import time
        t0 = time.time()
        try:
            res = strategies.run(item["question"], args.strategy)
        except Exception as e:  # noqa: BLE001
            print(f"❌ 执行失败: {type(e).__name__}: {e}")
            continue
        latencies.append(time.time() - t0)

        got = res["retrieved_titles"]
        print(f"\n  跳数={res['hop']}  停止原因={res['stop_reason']}  "
              f"检索次数={res['retrieval_calls']}  缓存命中={res['cache_hits']}")
        print("  检索查询:")
        for j, q in enumerate(res["queries"], 1):
            print(f"    第{j}跳: {q}")

        print("  检索结果（前 8 条）:")
        for rank, t in enumerate(got[:8], 1):
            mark = "✅" if t in gold else "  "
            print(f"    {mark} [{rank}] {t}")

        er = evidence_recall(got, gold, k=args.k)
        fr = full_recall(got, gold, k=args.k)
        em = exact_match(res["answer"], item["answer"])
        f1 = f1_score(res["answer"], item["answer"])

        print(f"\n  答案    : {res['answer']!r}")
        print(f"  标准答案: {item['answer']!r}")
        print(f"  指标    : EvidenceRecall@{args.k}={er:.2f}  "
              f"FullRecall@{args.k}={fr:.0f}  EM={em:.0f}  F1={f1:.2f}")

        em_sum += em
        f1_sum += f1
        fr_sum += fr

    n = len(latencies)
    print(f"\n{'=' * 76}")
    print(" 汇总")
    print(f"{'=' * 76}")
    if n:
        print(f"  样本数          : {n}")
        print(f"  Full Recall@{args.k}    : {fr_sum / n:.3f}")
        print(f"  EM              : {em_sum / n:.3f}")
        print(f"  F1              : {f1_sum / n:.3f}")
        print(f"  平均延迟        : {sum(latencies) / n:.1f} 秒")
    print(f"{'=' * 76}")
    print("\n提示：单条可能偶然，要看整体效果请跑：")
    print("  python scripts/run_experiments.py --limit 20")


if __name__ == "__main__":
    main()
