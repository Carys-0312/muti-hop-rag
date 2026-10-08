"""纯逻辑测试：验证评测集的分层配额算法（不需要数据集、不需要网络）。

用 HotpotQA **train split 的真实分布**模拟（来自实测）：

    bridge/easy       14,466    16.0%
    bridge/medium     46,074    50.9%
    bridge/hard       12,451    13.8%
    comparison/easy    3,506     3.9%
    comparison/medium 10,740    11.9%
    comparison/hard    3,210     3.5%
    ────────────────────────────────
    合计              90,447

用法：
    python scripts/smoke_test_quota.py
"""

from __future__ import annotations

import random

# 实测的 train split 分布
REAL_DIST = {
    ("bridge", "easy"): 14466,
    ("bridge", "medium"): 46074,
    ("bridge", "hard"): 12451,
    ("comparison", "easy"): 3506,
    ("comparison", "medium"): 10740,
    ("comparison", "hard"): 3210,
}
TOTAL = sum(REAL_DIST.values())

MIN_PER_STRATUM = 10


def allocate_quota(buckets: dict[tuple[str, str], int], size: int,
                   min_per_stratum: int = MIN_PER_STRATUM) -> dict[tuple[str, str], int]:
    """复制 build_eval_set 里的配额算法（保持同步）"""
    total = sum(buckets.values())
    if total == 0:
        return {}
    quota = {key: int(size * n / total) for key, n in buckets.items()}
    for key, n in buckets.items():
        quota[key] = min(max(quota[key], min_per_stratum), n)

    while sum(quota.values()) < size:
        cand = [k for k in buckets if quota[k] < buckets[k]]
        if not cand:
            break
        key = max(cand, key=lambda k: (buckets[k] - quota[k], buckets[k]))
        quota[key] += 1

    while sum(quota.values()) > size:
        cand = [k for k in buckets if quota[k] > min_per_stratum]
        if not cand:
            break
        key = max(cand, key=lambda k: quota[k])
        quota[key] -= 1

    return quota


def sample_pool(n_questions: int, seed: int = 42) -> dict[tuple[str, str], int]:
    """按真实比例随机抽 n_questions 个问题，返回各组合的实际数量"""
    keys = list(REAL_DIST)
    weights = [REAL_DIST[k] / TOTAL for k in keys]
    rng = random.Random(seed)

    counts: dict[tuple[str, str], int] = {k: 0 for k in keys}
    for _ in range(n_questions):
        counts[rng.choices(keys, weights=weights, k=1)[0]] += 1
    return counts


def simulate(n_questions: int, size: int, seed: int = 42) -> bool:
    buckets = {k: v for k, v in sample_pool(n_questions, seed).items() if v > 0}
    quota = allocate_quota(buckets, size)

    print(f"\n{'=' * 72}")
    print(f" 取样池 {n_questions} 题 → 评测集目标 {size} 条")
    print(f"{'=' * 72}")
    print(f" {'类别':<22}{'池子里':>8}{'配额':>8}{'余量':>8}  状态")
    print(f" {'-' * 58}")

    all_ok = True
    for key in sorted(buckets):
        n_pool, n_quota = buckets[key], quota[key]
        margin = n_pool - n_quota
        ok = n_quota <= n_pool and n_quota >= MIN_PER_STRATUM
        all_ok = all_ok and ok
        flag = "✅" if ok else "❌"
        print(f" {key[0] + '/' + key[1]:<22}{n_pool:>8}{n_quota:>8}{margin:>8}  {flag}")

    got = sum(quota.values())
    ok_total = got == size
    print(f" {'-' * 58}")
    print(f" {'合计':<22}{sum(buckets.values()):>8}{got:>8}")
    print(f"\n  配额总数 = {size}? {'✅' if ok_total else f'❌ ({got})'}"
          f"   每类 ≥ {MIN_PER_STRATUM} 且不超池子? {'✅' if all_ok else '❌'}")
    return all_ok and ok_total


def main() -> None:
    print("用 HotpotQA train split 的实测分布模拟")
    rare = REAL_DIST[("comparison", "hard")] / TOTAL
    print(f"（最稀有的组合 comparison/hard 只占 {rare:.1%}）")

    results = {n: simulate(n, 200) for n in (400, 600, 800)}

    print("\n" + "=" * 72)
    print(" 结论")
    print("=" * 72)
    for n, ok in results.items():
        print(f"  取样池 {n:>4} 题 : {'✅ 可行' if ok else '❌ 有类别凑不够'}")
    print("\n  当前配置 = 600 题")
    print("=" * 72)


if __name__ == "__main__":
    main()
