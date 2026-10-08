"""数据诊断 —— 查清"评测集全是 hard"的真正原因。

用法：
    python scripts/diagnose_data.py

它会查三件事：
  1. 容器里跑的到底是新版代码还是旧版
  2. HotpotQA 数据集的 level / type 真实分布是什么
  3. 用 select_items 随机抽 400 条，分布对不对
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _n in ("httpx", "httpcore", "urllib3", "huggingface_hub", "datasets", "qdrant_client"):
    logging.getLogger(_n).setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

from multihop_rag.config import settings  # noqa: E402
from multihop_rag.data import load_hotpotqa as L  # noqa: E402


def main() -> None:
    print("=" * 72)
    print(" 数据诊断")
    print("=" * 72)

    # ── ① 代码版本检查 ──
    print("\n【1】代码版本检查")
    path = Path(L.__file__)
    print(f"  模块路径      : {path}")
    print(f"  mtime         : {path.stat().st_mtime}")
    text = path.read_text(encoding="utf-8")

    checks = {
        "有 select_items 函数": "def select_items" in text,
        "有 _normalize 函数": "def _normalize" in text,
        "extract_corpus 用了 select_items": "items = select_items(" in text,
        "文件里提到随机抽样": "确定性随机抽样" in text,
    }
    for name, ok in checks.items():
        print(f"  {name:<34}: {'✅' if ok else '❌ 缺失'}")

    if not checks["有 select_items 函数"]:
        print("\n  ❌ 容器里跑的是【旧版代码】！")
        print("     绑定挂载没生效 —— 试试重建容器：docker compose up -d api")
        return

    # ── ② 数据集真实分布 ──
    ds = L.load_raw()   # split 从 config 读（默认 train）
    print(f"\n【2】数据集真实分布（共 {len(ds)} 条）")
    if hasattr(ds, "column_names"):
        print(f"  列名: {ds.column_names}")

    levels: Counter = Counter()
    types: Counter = Counter()
    for i in range(len(ds)):
        row = ds[i]
        levels[str(row.get("level"))] += 1
        types[str(row.get("type"))] += 1
    print(f"  level 分布: {dict(levels)}")
    print(f"  type  分布: {dict(types)}")

    if len(levels) == 1:
        print("  ⚠️ level 只有一种值！说明数据集里 level 列本身就是单一值，"
              "或者字段名读错了")

    # ── ③ 抽样分布 ──
    n = settings.corpus_questions
    print(f"\n【3】用 select_items 随机抽 {n} 条")
    items = L.select_items(ds, n=n, seed=42)
    print(f"  level 分布: {dict(Counter(it['level'] for it in items))}")
    print(f"  type  分布: {dict(Counter(it['type'] for it in items))}")

    # ── ④ 原始字段 ──
    print("\n【4】前 3 条的原始字段（确认字段名对不对）")
    for i in range(3):
        row = ds[i]
        keys = {k: row[k] for k in ("id", "type", "level") if k in row}
        print(f"  [{i}] {keys}")

    print("\n" + "=" * 72)
    print(" 把以上全部输出贴回去")
    print("=" * 72)


if __name__ == "__main__":
    main()
