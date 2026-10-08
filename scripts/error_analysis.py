"""错误分析：把答错的题自动归因分类，输出可直接写进简历/面试的结论。

用法：
    python scripts/error_analysis.py                    # 分析 adaptive
    python scripts/error_analysis.py --strategy single
    python scripts/error_analysis.py --strategy fixed_2hop

输入：results/<strategy>.records.jsonl（run_experiments.py 生成的逐条记录）
输出：
    · 终端打印完整报告
    · results/error_analysis_<strategy>.md（Markdown，可直接贴进文档）

分类逻辑：
    答错的题先分两大类 ——
      A. 检索失败：金标准文档没找全
           A1 两个都没找到
           A2 只找到一个
      B. 检索成功但生成失败：金标准都找到了，却答错
           B1 回答"无法确定"（模型主动放弃）
           B2 答了但内容错
           B3 部分正确（F1 较高但 EM=0，多半是措辞/格式差异）
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multihop_rag.config import settings  # noqa: E402
from multihop_rag.eval.metrics import exact_match, f1_score  # noqa: E402

# 判定"模型主动放弃"的关键词
GIVE_UP_MARKERS = ("无法确定", "无法回答", "不确定", "无法判断", "cannot determine",
                   "unknown", "insufficient")

CATEGORY_NAMES = {
    "A1": "检索失败：两个金标准都没找到",
    "A2": "检索失败：只找到一个金标准",
    "B1": "检索成功但回答「无法确定」",
    "B2": "检索成功但答案内容错误",
    "B3": "检索成功且部分正确（措辞/格式差异）",
    "C": "其他",
}


def classify(r: dict) -> tuple[str, float]:
    """给一条答错的记录归因"""
    gold = set(r["gold_titles"])
    got = set(r["retrieved_titles"])
    found = gold & got
    pred = (r.get("predicted") or "").strip()
    f1 = f1_score(pred, r["gold_answer"])

    # A. 检索失败
    if not found:
        return "A1", f1
    if len(found) < len(gold):
        return "A2", f1

    # B. 检索到了，但生成失败
    if not pred or any(m in pred.lower() for m in GIVE_UP_MARKERS):
        return "B1", f1
    if f1 >= 0.5:
        return "B3", f1
    return "B2", f1


def bar(n: int, total: int, width: int = 30) -> str:
    if total == 0:
        return ""
    filled = round(width * n / total)
    return "█" * filled + "·" * (width - filled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="adaptive",
                        choices=["single", "fixed_2hop", "adaptive"])
    parser.add_argument("--examples", type=int, default=5, help="每类打印几条样例")
    args = parser.parse_args()

    path = settings.results_dir / f"{args.strategy}.records.jsonl"
    if not path.exists():
        print(f"❌ 找不到 {path}")
        print("   先跑：python scripts/run_experiments.py")
        sys.exit(1)

    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    total = len(rows)

    # ── 逐条判定 ──
    correct, wrong = [], []
    cats: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        em = exact_match(r.get("predicted", ""), r["gold_answer"])
        r["_em"] = em
        r["_f1"] = f1_score(r.get("predicted", ""), r["gold_answer"])
        if em == 1:
            correct.append(r)
        else:
            wrong.append(r)
            cat, _ = classify(r)
            r["_cat"] = cat
            cats[cat].append(r)

    n_wrong = len(wrong)
    em_avg = sum(r["_em"] for r in rows) / total
    f1_avg = sum(r["_f1"] for r in rows) / total

    out = []
    P = out.append

    P("=" * 74)
    P(f" 错误分析报告   策略 = {args.strategy}")
    P("=" * 74)
    P("")
    P("【1】总体")
    P(f"  样本数        : {total}")
    P(f"  答对 (EM=1)   : {len(correct):>4}  ({len(correct)/total:.1%})")
    P(f"  答错 (EM=0)   : {n_wrong:>4}  ({n_wrong/total:.1%})")
    P(f"  平均 EM       : {em_avg:.3f}")
    P(f"  平均 F1       : {f1_avg:.3f}")

    # ── 归因 ──
    P("")
    P(f"【2】答错的 {n_wrong} 条归因")
    P("")
    group_a = sum(len(cats[c]) for c in ("A1", "A2"))
    group_b = sum(len(cats[c]) for c in ("B1", "B2", "B3"))

    P(f"  A. 检索失败（金标准没找全）  {group_a:>4} 条  {group_a/max(n_wrong,1):>6.1%}")
    for c in ("A1", "A2"):
        n = len(cats[c])
        P(f"       {c} {CATEGORY_NAMES[c]:<26} {n:>4}  {n/max(n_wrong,1):>6.1%}  {bar(n, n_wrong)}")
    P("")
    P(f"  B. 检索成功但生成失败        {group_b:>4} 条  {group_b/max(n_wrong,1):>6.1%}")
    for c in ("B1", "B2", "B3"):
        n = len(cats[c])
        P(f"       {c} {CATEGORY_NAMES[c]:<26} {n:>4}  {n/max(n_wrong,1):>6.1%}  {bar(n, n_wrong)}")
    if cats["C"]:
        P("")
        P(f"  C. 其他                      {len(cats['C']):>4} 条")

    # ── 改善空间 ──
    P("")
    P("【3】改善空间估算（这是面试最关心的）")
    P("")
    P(f"  当前 EM = {em_avg:.3f}")
    P("")
    if n_wrong:
        b1 = len(cats["B1"])
        b2 = len(cats["B2"])
        b3 = len(cats["B3"])
        a2 = len(cats["A2"])
        P(f"  · 若把「B1 无法确定」({b1} 条) 修好一半 → EM +{b1*0.5/total:.3f}")
        P(f"  · 若把「B2 内容错误」({b2} 条) 修好一半 → EM +{b2*0.5/total:.3f}")
        P(f"  · 若把「A2 只找到一个」({a2} 条) 补全一半 → EM +{a2*0.5/total:.3f}")
        P("")
        P("  → 优先级应放在占比最大的那一类")

    # ── 按难度 / 类型 ──
    for key, label in (("level", "难度"), ("type", "类型")):
        P("")
        P(f"【4】按{label}拆分")
        P("")
        P(f"  {label:<12}{'样本':>6}{'EM':>9}{'F1':>9}{'平均跳数':>10}")
        P(f"  {'-'*46}")
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[r.get(key, "?")].append(r)
        for g in sorted(groups):
            rs = groups[g]
            e = sum(x["_em"] for x in rs) / len(rs)
            f = sum(x["_f1"] for x in rs) / len(rs)
            h = sum(x.get("hop", 0) for x in rs) / len(rs)
            P(f"  {g:<12}{len(rs):>6}{e:>9.3f}{f:>9.3f}{h:>10.2f}")

    # ── judge 行为 ──
    P("")
    P("【5】Agent 行为统计")
    P("")
    hop_dist = Counter(r.get("hop", 0) for r in rows)
    P("  跳数分布:")
    for h in sorted(hop_dist):
        n = hop_dist[h]
        P(f"    {h} 跳 : {n:>4} 条  {n/total:>6.1%}  {bar(n, total)}")
    P("")
    reason_dist = Counter(r.get("stop_reason", "?") for r in rows)
    P("  停止原因分布:")
    for rs, n in reason_dist.most_common():
        P(f"    {rs:<20} {n:>4} 条  {n/total:>6.1%}")
    P("")
    avg_hops = sum(r.get("hop", 0) for r in rows) / total
    P(f"  平均跳数      : {avg_hops:.2f}")
    P(f"  平均检索次数  : {sum(r.get('retrieval_calls',0) for r in rows)/total:.2f}")
    P(f"  缓存命中率    : {sum(r.get('cache_hits',0) for r in rows)/(sum(r.get('cache_hits',0)+r.get('retrieval_calls',0) for r in rows) or 1):.2%}")

    # ── 典型样例 ──
    P("")
    P("【6】典型错误样例")
    for c in ("A1", "A2", "B1", "B2", "B3"):
        if not cats[c]:
            continue
        P("")
        P(f"  ── {c} {CATEGORY_NAMES[c]}（共 {len(cats[c])} 条，示例 {min(args.examples, len(cats[c]))} 条）──")
        for r in cats[c][: args.examples]:
            gold = r["gold_titles"]
            got = r["retrieved_titles"]
            found = [t for t in gold if t in got]
            P("")
            P(f"   qid={r['qid']} ({r['type']}/{r['level']})  跳数={r.get('hop')}  停止={r.get('stop_reason')}")
            P(f"   Q: {r['question'][:100]}")
            P(f"   检索命中 : {found or '（无）'}   /   应检索 : {gold}")
            P(f"   预测答案 : {r.get('predicted','')[:80]!r}")
            P(f"   标准答案 : {r['gold_answer'][:80]!r}")
            P(f"   查询链   : {' → '.join((r.get('queries') or [])[:3])[:140]}")

    P("")
    P("=" * 74)

    report = "\n".join(out)
    print(report)

    # 保存 Markdown
    md_path = settings.results_dir / f"error_analysis_{args.strategy}.md"
    md = ["# 错误分析报告\n", f"- 策略: `{args.strategy}`", f"- 样本数: {total}",
          f"- EM: {em_avg:.3f}", f"- F1: {f1_avg:.3f}\n", "```", report, "```"]
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n✅ 报告已保存: {md_path}")


if __name__ == "__main__":
    main()
