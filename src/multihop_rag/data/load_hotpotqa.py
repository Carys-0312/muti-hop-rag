"""加载 HotpotQA 并抽取文档库。

⭐ 关键：supporting_facts 给出了"回答这个问题需要哪几篇文档"，
   这就是算召回率的金标准 —— 不用自己标注。这是选 HotpotQA 的最大理由。

──────────────────────────────────────────────────────────────
两条加载路径（自动降级）：

  路径 A：用 huggingface `datasets` 库
          优点：省事
          ⚠️ 缺点：新版 datasets 对"没有命名空间的旧数据集"有 URI 解析问题

  路径 B：直接下载官方原始 JSON（纯 requests + json）
          优点：不依赖 datasets 库，不受版本问题影响
          缺点：要自己解析

  先试 A，失败自动走 B。
  两条路径返回结构不同，统一由 `iter_items()` 做归一化。
──────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator

from ..config import settings

logger = logging.getLogger(__name__)

# ⭐ 必须带命名空间！否则新版 datasets 会拼出非法的 hf:// URI
DATASET_ID = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"

# 路径 B 用的官方原始数据地址（可用环境变量 HOTPOTQA_URL 覆盖）
RAW_URLS = {
    "validation": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    "train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
}


# ══════════════════════════════════════════════════════════
#  加载：两条路径 + 自动降级
# ══════════════════════════════════════════════════════════
def load_raw(split: str | None = None):
    """加载原始数据。返回 datasets.Dataset 或 list[dict]，交给 iter_items 归一化。

    ⚠️ split 默认从 config 读（默认 "train"），不是 "validation"！
        因为 validation split 的 level 整列都是 hard，做不了分难度分析。
        实测：validation 7405 条全部 level='hard'；
             train 90447 条有完整三档（easy/medium/hard）。
    """
    split = split or settings.data_split
    try:
        return _load_via_datasets(split)
    except Exception as e:  # noqa: BLE001
        logger.warning("路径 A（datasets 库）失败：%s", e)
        logger.warning("→ 自动降级到路径 B（直接下载原始 JSON）")

    return _load_via_json(split)


def _load_via_datasets(split: str):
    """路径 A：用 huggingface datasets 库

    ⚠️ 不再传 trust_remote_code：
        HotpotQA 在 HF 上已经转成 parquet 格式（不再依赖加载脚本），
        直接加载即可。新版 datasets 已经把该参数移除，
        传了会打出一堆吓人的 ERROR（虽然不影响结果）。
    """
    from datasets import load_dataset

    logger.info("路径 A：用 datasets 库加载 %s / %s", DATASET_ID, split)
    return load_dataset(DATASET_ID, DATASET_CONFIG, split=split)


def _load_via_json(split: str) -> list[dict]:
    """路径 B：直接下载官方原始 JSON（不依赖 datasets 库）"""
    url = os.getenv("HOTPOTQA_URL") or RAW_URLS.get(split)
    if not url:
        raise RuntimeError(f"没有为 split={split} 配置下载地址")

    # 缓存到 data/raw/，避免重复下载
    cache_dir = settings.raw_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / Path(url).name

    if cache_file.exists() and cache_file.stat().st_size > 0:
        logger.info("路径 B：使用已缓存的原始文件 %s", cache_file)
    else:
        logger.info("路径 B：从 %s 下载（约 45MB）...", url)
        _download(url, cache_file)

    logger.info("解析 JSON：%s", cache_file)
    with open(cache_file, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("加载完成：%d 条", len(data))
    return data


def _download(url: str, dest: Path) -> None:
    """带进度显示的下载"""
    import requests

    headers = {"User-Agent": "multi-hop-rag/0.1.0"}
    with requests.get(url, stream=True, timeout=120, headers=headers) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    print(f"\r  下载中 {pct:3d}%  {done/1e6:.1f}/{total/1e6:.1f} MB", end="")
        print()
    tmp.replace(dest)
    logger.info("下载完成：%s", dest)


# ══════════════════════════════════════════════════════════
#  归一化：把两种来源统一成同一种结构
# ══════════════════════════════════════════════════════════
def _normalize(row) -> dict[str, Any]:
    """把一行数据统一成标准 dict。

    两种来源的结构差异：
        datasets 库（列式）              原始 JSON（行式）
        supporting_facts = {...}        supporting_facts = [[title, idx], ...]
        context          = {...}        context          = [[title, [sent...]], ...]
        id                              _id
    """
    sf = row["supporting_facts"]
    if isinstance(sf, dict):                      # datasets 库的结构
        titles = list(sf["title"])
    else:                                         # 原始 JSON 的结构
        titles = [pair[0] for pair in sf]

    ctx = row["context"]
    if isinstance(ctx, dict):                     # datasets 库的结构
        pairs = list(zip(ctx["title"], ctx["sentences"]))
    else:                                         # 原始 JSON 的结构
        pairs = [(pair[0], pair[1]) for pair in ctx]

    return {
        "id": row.get("id") or row.get("_id", ""),
        "question": row["question"].strip(),
        "answer": str(row["answer"]).strip(),
        "type": row.get("type", "") or "",
        "level": row.get("level", "") or "",
        "gold_titles": sorted({str(t) for t in titles}),
        "context": pairs,
    }


def iter_items(ds) -> Iterator[dict[str, Any]]:
    """按原始顺序遍历（一般不用这个，用 select_items 抽样）"""
    for row in ds:
        yield _normalize(row)


def select_items(
    ds,
    n: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """⭐ 确定性随机抽样，从【整个数据集】里抽 n 条。

    ⚠️ 为什么不能"取前 n 条"：
        HotpotQA dev 集的开头几乎全是 hard 难度题目。
        取前 300 条会导致抽样严重偏斜（实测 200 条全是 hard），
        分层分析就彻底失效了 —— 没法比较"easy/medium/hard 各提升多少"。

    用固定 seed，保证「建索引」和「建评测集」拿到【完全相同】的一批问题。
    """
    import random

    n = n or settings.corpus_questions
    total = len(ds)

    if n >= total:
        idxs = list(range(total))
    else:
        idxs = sorted(random.Random(seed).sample(range(total), n))

    logger.info("从 %d 条里随机抽取 %d 条（seed=%d）", total, len(idxs), seed)
    return [_normalize(ds[i]) for i in idxs]


# ══════════════════════════════════════════════════════════
#  抽取文档库
# ══════════════════════════════════════════════════════════
def extract_corpus(
    ds,
    min_chars: int = 50,
    max_questions: int | None = None,
    seed: int = 42,
) -> dict[str, dict]:
    """抽取文档库（按标题去重）。

    ⭐ 为什么只取「N 个问题的候选文档」而不是全量 66563 篇：

        HotpotQA distractor 每题自带 10 篇候选文档（2 篇金标准 + 8 篇干扰），
        所以【标准任务】就是"从这 10 篇里挑出需要的那 2 篇"。

        只索引评测问题用到的候选文档（300 题 ≈ 2500 篇）：
          ✅ 建索引 2 小时 → 3~5 分钟
          ✅ 内存压力小，不会 OOM
          ✅ 符合 HotpotQA 的标准设定

        如果索引全量 66563 篇：慢 20 倍，而且偏离了 benchmark 定义。

    ⭐ 抽的是【随机】N 个问题，不是"前 N 个"：
        HotpotQA dev 开头几乎全是 hard 题目，取前 N 个会严重偏斜。

    ⚠️ max_questions 和 seed 必须和 build_eval_set 完全一致，
       否则会出现"评测集里的问题，候选文档不在索引里"。
       两者默认都读 config 的 data.corpus_questions + seed=42，自动一致。
    """
    items = select_items(ds, n=max_questions, seed=seed)

    corpus: dict[str, dict] = {}
    for item in items:
        for title, sentences in item["context"]:
            if title in corpus:
                continue
            text = " ".join(str(s).strip() for s in sentences).strip()
            if len(text) < min_chars:
                continue
            corpus[title] = {
                "chunk_id": title,       # HotpotQA 是文章级，用标题当 id
                "title": title,
                "text": text,
            }

    logger.info("抽取文档库：%d 篇（来自 %d 个问题的候选文档）", len(corpus), len(items))
    return corpus


def save_corpus(corpus: dict[str, dict], path: Path | None = None) -> Path:
    path = path or settings.corpus_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc in corpus.values():
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    logger.info("文档库已保存: %s (%d 篇)", path, len(corpus))
    return path


def load_corpus(path: Path | None = None) -> list[dict]:
    path = path or settings.corpus_path
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ══════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════
def main(split: str = "validation", max_questions: int | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    max_questions = max_questions or settings.corpus_questions

    ds = load_raw(split)
    corpus = extract_corpus(ds, max_questions=max_questions)
    save_corpus(corpus)

    # 打印取样范围的分布，方便写进 README
    items = select_items(ds, n=max_questions)
    types: dict[str, int] = {}
    levels: dict[str, int] = {}
    for item in items:
        types[item["type"]] = types.get(item["type"], 0) + 1
        levels[item["level"]] = levels.get(item["level"], 0) + 1

    print(f"\n取样问题数: {len(items)}")
    print(f"类型分布: {types}")
    print(f"难度分布: {levels}")
    print(f"文档库: {len(corpus)} 篇")


if __name__ == "__main__":
    main()
