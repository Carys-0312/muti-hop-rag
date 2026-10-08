"""跑评测：拿一个策略跑完评测集，算指标，保存结果。"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from tqdm import tqdm

from ..agents import strategies
from ..config import settings
from ..runtime import get_cache
from .metrics import compute_breakdown, compute_metrics

logger = logging.getLogger(__name__)


def run_strategy(
    strategy: str,
    eval_set: list[dict],
    limit: int | None = None,
    clear_cache: bool = True,
) -> tuple[list[dict], dict]:
    """跑一个策略，返回 (逐条结果, 指标)"""
    if clear_cache:
        get_cache().clear()          # 保证每个策略从同样的状态开始，对比才公平

    if limit:
        eval_set = eval_set[:limit]

    records: list[dict] = []
    t0 = time.time()

    for item in tqdm(eval_set, desc=f"{strategies.STRATEGY_LABELS.get(strategy, strategy)}"):
        try:
            result = strategies.run(item["question"], strategy)
        except Exception as e:  # noqa: BLE001
            # 单条失败不能让整个评测挂掉——记下来，后面计入错误分析
            logger.warning("策略 %s 处理 qid=%s 失败: %s", strategy, item["qid"], e)
            result = {
                "question": item["question"],
                "answer": "",
                "answer_raw": "",
                "retrieved_titles": [],
                "hop": 0,
                "stop_reason": f"error:{type(e).__name__}",
                "cache_hits": 0,
                "retrieval_calls": 0,
                "evidences": [],
                "queries": [],
                "used_entities": [],
            }

        records.append(
            {
                "qid": item["qid"],
                "question": item["question"],
                "gold_answer": item["answer"],
                "gold_titles": item["gold_titles"],
                "type": item["type"],
                "level": item["level"],
                "predicted": result.get("answer", ""),
                "answer_raw": result.get("answer_raw", ""),
                "retrieved_titles": result.get("retrieved_titles", []),
                "hop": result.get("hop", 0),
                "stop_reason": result.get("stop_reason", ""),
                "queries": result.get("queries", []),
                "cache_hits": result.get("cache_hits", 0),
                "retrieval_calls": result.get("retrieval_calls", 0),
                "used_entities": result.get("used_entities", []),
                "judge_sufficient": result.get("judge_sufficient"),
            }
        )

    elapsed = time.time() - t0
    k = settings.eval.top_k_for_recall
    metrics = compute_metrics(records, k=k)

    # ⭐ 分场景分析（面试的关键素材）
    metrics["by_type"] = compute_breakdown(records, "type", k=k)
    metrics["by_level"] = compute_breakdown(records, "level", k=k)
    metrics["stop_reason_dist"] = _dist(records, "stop_reason")

    # 缓存收益
    total_retrieval = sum(r["retrieval_calls"] for r in records)
    total_cache_hits = sum(r["cache_hits"] for r in records)
    metrics["cache"] = {
        "total_retrieval_calls": total_retrieval,
        "total_cache_hits": total_cache_hits,
        "hit_rate": round(
            total_cache_hits / (total_retrieval + total_cache_hits), 4
        ) if (total_retrieval + total_cache_hits) else 0.0,
    }
    metrics["elapsed_sec"] = round(elapsed, 1)
    metrics["avg_latency_sec"] = round(elapsed / max(len(records), 1), 2)
    metrics["strategy"] = strategy

    return records, metrics


def _dist(records: list[dict], key: str) -> dict[str, int]:
    d: dict[str, int] = {}
    for r in records:
        v = str(r.get(key, "") or "unknown")
        d[v] = d.get(v, 0) + 1
    return dict(sorted(d.items(), key=lambda kv: -kv[1]))


def save_results(records: list[dict], metrics: dict, strategy: str) -> Path:
    out_dir = settings.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"{strategy}.metrics.json"
    records_path = out_dir / f"{strategy}.records.jsonl"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(records_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("结果已保存: %s / %s", metrics_path.name, records_path.name)
    return metrics_path
