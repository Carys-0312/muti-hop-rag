"""构建评测集：从 HotpotQA 分层抽样，生成 eval_set.jsonl。

为什么分层抽样而不是随机抽 200 条：
    这样你才能分析"自适应多跳在哪类问题上提升最大"。
    比如："hard 问题提升 +38%，easy 问题只提升 +6%"
    —— 这种分场景分析是面试官最想听的。

⭐ 配额怎么定（重要设计决策）：
    不用硬编码的比例（比如"bridge 60% / comparison 40%"），
    而是【按取样池的实际分布按比例分配】，并保证每类至少 min_per_stratum 条。

    原因：HotpotQA 真实分布是 bridge 约 80% / comparison 约 20%
    （和我一开始猜的 60/40 差很远）。硬编码会导致
    "comparison 类名额永远凑不齐，连续跳过好几个类别"。

⚠️ 评测集一旦确定就不要改！否则不同策略的数字没法对比。
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from ..config import settings
from .load_hotpotqa import load_raw, select_items

logger = logging.getLogger(__name__)

# 每个 (type, level) 组合至少保留多少条，保证能做分场景分析
MIN_PER_STRATUM = 10


def build_eval_set(
    ds,
    size: int | None = None,
    seed: int = 42,
    pool_size: int | None = None,
    min_per_stratum: int = MIN_PER_STRATUM,
) -> list[dict]:
    """分层抽样构建评测集。

    ⚠️ pool_size 和 seed 必须和「建索引时用的参数」完全一致！
       否则会出现：评测集里有问题，但它的候选文档不在索引里 → 永远检索不到。
       两者都由 config 的 data.corpus_questions + seed=42 控制，自动一致。
    """
    size = size or settings.eval.sample_size
    pool_size = pool_size or settings.corpus_questions

    # ⭐ 从【整个数据集随机抽】pool_size 个问题（和 extract_corpus 用同一个 seed）
    #    注意：不能"取前 N 个"——HotpotQA dev 开头几乎全是 hard 题目，会导致严重偏斜
    items = select_items(ds, n=pool_size, seed=seed)
    logger.info("取样池：随机 %d 个问题（与索引范围一致）", len(items))

    rng = random.Random(seed)          # 固定种子，保证可复现

    # 过滤：必须有支撑文档，答案不能是空
    items = [i for i in items if i["gold_titles"] and i["answer"]]

    # 按 (type, level) 分组
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for it in items:
        buckets[(it["type"], it["level"])].append(it)

    total = len(items)
    if total == 0:
        raise RuntimeError("取样池是空的，检查数据加载是否正常")

    # ── ① 按池子的实际分布按比例分配 ──
    quota: dict[tuple[str, str], int] = {
        key: int(size * len(pool) / total) for key, pool in buckets.items()
    }
    # ── ② 每类至少 min_per_stratum（但不超过该类实际数量）──
    for key, pool in buckets.items():
        quota[key] = min(max(quota[key], min_per_stratum), len(pool))

    # ── ③ 把总数调回 size ──
    # 先补（优先补"还有余量且池子大"的类，这样分布更接近真实比例）
    while sum(quota.values()) < size:
        candidates = [k for k in buckets if quota[k] < len(buckets[k])]
        if not candidates:
            logger.warning(
                "取样池只有 %d 条可用，凑不够 %d 条评测集", total, size
            )
            break
        key = max(candidates, key=lambda k: (len(buckets[k]) - quota[k], len(buckets[k])))
        quota[key] += 1

    # 再削（优先削"配额最多且超过下限"的类）
    while sum(quota.values()) > size:
        candidates = [k for k in buckets if quota[k] > min_per_stratum]
        if not candidates:
            break
        key = max(candidates, key=lambda k: quota[k])
        quota[key] -= 1

    # ── ④ 按配额抽样 ──
    picked: list[dict] = []
    for key, n in sorted(quota.items()):
        if n <= 0:
            continue
        picked += rng.sample(buckets[key], n)

    picked = picked[:size]
    rng.shuffle(picked)

    if sum(quota.values()) < size:
        logger.warning(
            "⚠️ 实际只取到 %d 条（目标 %d）。\n"
            "   原因：取样池太小。\n"
            "   解决：把 configs/default.yaml 的 data.corpus_questions 调大"
            "（比如 600），再跑 build_index.py --download 重建索引。",
            len(picked), size,
        )

    # ── ⑤ 生成评测集（qid 重编，方便引用）──
    eval_set = [
        {
            "qid": f"{idx:04d}",
            "question": it["question"],
            "answer": it["answer"],
            "gold_titles": it["gold_titles"],
            "type": it["type"],
            "level": it["level"],
            "source_id": it["id"],
        }
        for idx, it in enumerate(picked)
    ]
    return eval_set


def save_eval_set(eval_set: list[dict], path: Path | None = None) -> Path:
    path = path or settings.eval_set_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in eval_set:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("评测集已保存: %s (%d 条)", path, len(eval_set))
    return path


def load_eval_set(path: Path | None = None) -> list[dict]:
    path = path or settings.eval_set_path
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(size: int | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ds = load_raw()   # split 从 config 读（默认 train）
    eval_set = build_eval_set(ds, size=size)
    save_eval_set(eval_set)

    # 打印分布，确认分层生效
    by_type: dict[str, int] = defaultdict(int)
    by_level: dict[str, int] = defaultdict(int)
    by_both: dict[str, int] = defaultdict(int)
    for r in eval_set:
        by_type[r["type"]] += 1
        by_level[r["level"]] += 1
        by_both[f"{r['type']}/{r['level']}"] += 1

    print(f"\n✅ 评测集规模: {len(eval_set)}")
    print(f"   类型: {dict(by_type)}")
    print(f"   难度: {dict(by_level)}")
    print(f"   细分: {dict(sorted(by_both.items()))}")
    print("\n前 3 条示例:")
    for r in eval_set[:3]:
        print(f"  [{r['qid']}] ({r['type']}/{r['level']}) {r['question'][:70]}")
        print(f"        答案={r['answer']}  需检索={r['gold_titles']}")
    print(f"\n   路径: {settings.eval_set_path}")


if __name__ == "__main__":
    main()
