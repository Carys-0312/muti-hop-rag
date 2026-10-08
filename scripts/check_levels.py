"""检查 HotpotQA 各个 split 的 level / type 分布。

背景：
    实测 HF 上 hotpot_qa/distractor 的 validation split（7405 条）里，
    level 列【全部是 hard】，没有 easy/medium，没法做分难度的分层分析。

    这个脚本检查 train split 是否有混合难度，
    以便决定"换 split"还是"改用 type 分组"。

用法：
    python scripts/check_levels.py
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _n in ("httpx", "httpcore", "urllib3", "huggingface_hub"):
    logging.getLogger(_n).setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)


def check(split: str) -> dict | None:
    """只看 level / type 两列，避免把整个数据集读进内存"""
    from datasets import load_dataset

    print(f"\n{'─' * 68}")
    print(f"  split = {split}")
    print(f"{'─' * 68}")

    try:
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ 加载失败: {type(e).__name__}: {e}")
        return None

    n = len(ds)
    print(f"  条数: {n}")

    # 只取需要的两列，省内存
    try:
        sub = ds.select_columns(["type", "level"])
    except Exception:  # noqa: BLE001
        sub = ds

    try:
        levels = Counter(sub["level"])
        types = Counter(sub["type"])
        print(f"  level 分布: {dict(levels)}")
        print(f"  type  分布: {dict(types)}")
        print(f"  level 种类数: {len(levels)}  → "
              f"{'✅ 有区分度' if len(levels) > 1 else '❌ 只有一种值'}")

        # 交叉表（如果 level 有区分度）
        if len(levels) > 1:
            both = Counter(zip(sub["type"], sub["level"]))
            print(f"  交叉分布: {dict(sorted(both.items()))}")

        return {"n": n, "levels": dict(levels), "types": dict(types)}

    except Exception as e:  # noqa: BLE001
        print(f"  ❌ 读取列失败: {type(e).__name__}: {e}")
        return None


def main() -> None:
    print("=" * 68)
    print(" HotpotQA level / type 分布检查")
    print("=" * 68)

    results = {}
    for split in ["validation", "train"]:
        results[split] = check(split)

    # ── 给结论 ──
    print("\n" + "=" * 68)
    print(" 结论")
    print("=" * 68)

    valid = results.get("validation")
    train = results.get("train")

    v_ok = valid and len(valid["levels"]) > 1
    t_ok = train and len(train["levels"]) > 1

    if v_ok:
        print(" ✅ validation split 的 level 有区分度，可以直接用")
        print("    → 检查是不是数据版本问题")
    elif t_ok:
        print(" ⚠️ validation 的 level 全是单一值，但 train 有区分度")
        print("    → 建议：改用 train split 抽样")
        print("       在 configs/default.yaml 里加：")
        print("         data:")
        print("           split: train")
        print("       或者直接改 load_raw('train')")
    else:
        print(" ❌ 两个 split 的 level 都没有区分度")
        print("    → 说明 HF 这版 HotpotQA 丢失了难度标注")
        print("    → 方案：改用【type】做分组分析（bridge vs comparison）")
        print("       type 分布是正常的，一样能做出有价值的对比结论：")
        print("         '自适应多跳在桥接型问题上提升 +40%，对比型 +15%'")
        print("       同时把 level 相关的分析从计划里去掉")

    print("=" * 68)


if __name__ == "__main__":
    main()
