"""命令行入口。

用法示例：
    python -m multihop_rag.cli ask "Were Pavel Urysohn and Leonid Levin born in the same country?"
    python -m multihop_rag.cli ask "..." --strategy adaptive --verbose
    python -m multihop_rag.cli eval --strategy single --limit 20
    python -m multihop_rag.cli compare --limit 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def cmd_ask(args) -> None:
    from .agents import strategies

    result = strategies.run(args.question, args.strategy)

    print("=" * 70)
    print(f"问题: {args.question}")
    print(f"策略: {strategies.STRATEGY_LABELS.get(args.strategy, args.strategy)}")
    print("=" * 70)

    if args.verbose:
        print("\n【检索查询】")
        for i, q in enumerate(result["queries"], 1):
            print(f"  第{i}跳: {q}")

        print(f"\n【检索到的证据】共 {len(result['evidences'])} 条")
        for i, ev in enumerate(result["evidences"][:8], 1):
            print(f"  [{i}] ({ev['score']:.3f}) {ev['title']}  ← 来自第{ev['hop']}跳")
            print(f"      {ev['text'][:150]}...")

    print("\n【答案】")
    print(f"  {result['answer']}")

    print("\n【统计】")
    print(f"  跳数: {result['hop']}")
    print(f"  停止原因: {result['stop_reason']}")
    print(f"  实际检索次数: {result['retrieval_calls']}  缓存命中: {result['cache_hits']}")
    if result.get("judge_reasoning"):
        print(f"  judge 理由: {result['judge_reasoning']}")

    if args.json:
        print("\n【原始 JSON】")
        print(json.dumps({k: v for k, v in result.items() if k != "evidences"},
                         ensure_ascii=False, indent=2))


def cmd_eval(args) -> None:
    from .data.build_eval_set import load_eval_set
    from .eval.runner import run_strategy, save_results

    eval_set = load_eval_set()
    print(f"评测集: {len(eval_set)} 条，策略: {args.strategy}，限制: {args.limit or '全部'}")

    records, metrics = run_strategy(args.strategy, eval_set, limit=args.limit)
    save_results(records, metrics, args.strategy)

    _print_metrics(metrics)


def cmd_compare(args) -> None:
    from .data.build_eval_set import load_eval_set
    from .eval.runner import run_strategy, save_results

    eval_set = load_eval_set()
    all_metrics = {}

    for strategy in ["single", "fixed_2hop", "adaptive"]:
        print(f"\n{'=' * 70}\n跑策略: {strategy}\n{'=' * 70}")
        records, metrics = run_strategy(strategy, eval_set, limit=args.limit)
        save_results(records, metrics, strategy)
        all_metrics[strategy] = metrics
        _print_metrics(metrics)

    _write_comparison(all_metrics)


def _print_metrics(metrics: dict) -> None:
    k = 5
    print("\n── 总体指标 ──")
    for key in [f"evidence_recall@{k}", f"full_recall@{k}", "mrr", "em", "f1", "hop"]:
        if key in metrics:
            print(f"  {key:>18}: {metrics[key]}")

    print("\n── 缓存 ──")
    for key, val in metrics.get("cache", {}).items():
        print(f"  {key:>22}: {val}")

    if "by_level" in metrics:
        print("\n── 按难度 ──")
        for lv, m in metrics["by_level"].items():
            print(f"  {lv:>10}: FullRecall={m[f'full_recall@{k}']:.3f}  F1={m['f1']:.3f}  hops={m['hop']:.2f}")

    if metrics.get("stop_reason_dist"):
        print("\n── 停止原因分布 ──")
        for reason, cnt in metrics["stop_reason_dist"].items():
            print(f"  {reason:>20}: {cnt}")


def _write_comparison(all_metrics: dict) -> None:
    from .config import settings

    path = settings.results_dir / "comparison.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    order = ["single", "fixed_2hop", "adaptive"]
    labels = {"single": "S1 单跳", "fixed_2hop": "S2 固定两跳", "adaptive": "S3 自适应多跳"}

    lines = ["# 三种策略对比\n", "| 策略 | Evidence Recall@5 | Full Recall@5 | MRR | EM | F1 | 平均跳数 | 平均延迟(s) |",
             "|---|---|---|---|---|---|---|---|"]
    for s in order:
        m = all_metrics.get(s)
        if not m:
            continue
        lines.append(
            f"| {labels[s]} | {m['evidence_recall@5']:.3f} | {m['full_recall@5']:.3f} | "
            f"{m['mrr']:.3f} | {m['em']:.3f} | {m['f1']:.3f} | {m['hop']:.2f} | "
            f"{m.get('avg_latency_sec', 0):.2f} |"
        )

    # 缓存对比
    lines += ["\n## 缓存收益\n", "| 策略 | 实际检索次数 | 缓存命中 | 命中率 |", "|---|---|---|---|"]
    for s in order:
        m = all_metrics.get(s)
        if not m:
            continue
        c = m["cache"]
        lines.append(
            f"| {labels[s]} | {c['total_retrieval_calls']} | {c['total_cache_hits']} | {c['hit_rate']:.2%} |"
        )

    # 按难度
    lines += ["\n## 按难度分组（Full Recall@5）\n"]
    levels = ["easy", "medium", "hard"]
    lines.append("| 策略 | " + " | ".join(levels) + " |")
    lines.append("|---" * (len(levels) + 1) + "|")
    for s in order:
        m = all_metrics.get(s)
        if not m:
            continue
        row = [labels[s]]
        for lv in levels:
            row.append(f"{m['by_level'].get(lv, {}).get('full_recall@5', 0):.3f}")
        lines.append("| " + " | ".join(row) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✅ 对比报告已写入: {path}")


def main(argv: list[str] | None = None) -> int:
    # ⭐ 用 parents=[common] 让 -v/--verbose 在【子命令前或后】都能用
    #    默认 argparse 只允许写在子命令前面，很反直觉：
    #        cli.py ask "..." --verbose   ❌ 报 unrecognized arguments
    #        cli.py --verbose ask "..."   ✅
    #    用 default=SUPPRESS 是为了让子解析器不把主解析器设的 True 覆盖成 False。
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
        help="打印详细日志（可放在子命令前或后）",
    )

    parser = argparse.ArgumentParser(
        prog="mhrag", description="多跳检索问答 Agent", parents=[common]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ask = sub.add_parser("ask", parents=[common], help="问一个问题")
    p_ask.add_argument("question")
    p_ask.add_argument("--strategy", default="adaptive",
                       choices=["single", "fixed_2hop", "adaptive"])
    p_ask.add_argument("--json", action="store_true", help="额外打印 JSON")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", parents=[common], help="在评测集上跑一个策略")
    p_eval.add_argument("--strategy", default="adaptive",
                        choices=["single", "fixed_2hop", "adaptive"])
    p_eval.add_argument("--limit", type=int, default=None, help="只跑前 N 条（调试用）")
    p_eval.set_defaults(func=cmd_eval)

    p_cmp = sub.add_parser("compare", parents=[common], help="跑三种策略并生成对比报告")
    p_cmp.add_argument("--limit", type=int, default=None)
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    # 用 getattr + 默认值，兼容 default=SUPPRESS 的情况
    verbose = getattr(args, "verbose", False)
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
