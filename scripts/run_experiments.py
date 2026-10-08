"""一键跑三种策略并生成对比报告。

用法：
    python scripts/run_experiments.py --limit 20     # 先小样本验证流程（省 API 费）
    python scripts/run_experiments.py                # 跑完整评测集
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="每种策略只跑前 N 条")
    parser.add_argument("--strategies", nargs="+",
                        default=["single", "fixed_2hop", "adaptive"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    from multihop_rag.cli import _write_comparison, _print_metrics
    from multihop_rag.data.build_eval_set import load_eval_set
    from multihop_rag.eval.runner import run_strategy, save_results

    eval_set = load_eval_set()
    n = args.limit or len(eval_set)
    print(f"评测集 {len(eval_set)} 条，本次每种策略跑 {n} 条")
    print(f"策略: {args.strategies}\n")

    all_metrics = {}
    for strategy in args.strategies:
        print(f"\n{'=' * 70}\n▶ 策略: {strategy}\n{'=' * 70}")
        records, metrics = run_strategy(strategy, eval_set, limit=args.limit)
        save_results(records, metrics, strategy)
        all_metrics[strategy] = metrics
        _print_metrics(metrics)

    _write_comparison(all_metrics)
    print("\n完成。结果在 results/ 目录。")


if __name__ == "__main__":
    main()
