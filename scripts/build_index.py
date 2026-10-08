"""建立向量索引。

用法：
    # 完整流程：下载数据 → 抽取候选文档库 → 向量化 → 写入 Qdrant
    python scripts/build_index.py --download

    # 文档库已存在时，直接建索引
    python scripts/build_index.py

    # 调整取样问题数（默认读 configs/default.yaml 的 data.corpus_questions）
    python scripts/build_index.py --download --questions 200

    # 调整批大小（内存不够就调小）
    python scripts/build_index.py --batch-size 32

⚠️ --questions 必须和 build_eval_set.py 的取样范围一致，
   否则评测集里会有"候选文档不在索引里"的问题。
   两个脚本默认都读 config 的 data.corpus_questions，一般不用手动指定。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 便于不安装也能直接跑
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multihop_rag.config import settings  # noqa: E402


def quiet_noisy_loggers() -> None:
    """关掉噪音日志。

    不关的话，每批向量化都会打一堆
    `INFO HTTP Request: PUT http://qdrant...200 OK`，
    几千行刷屏，既干扰看进度、也拖慢速度。
    """
    for name in ("httpx", "httpcore", "urllib3", "huggingface_hub", "datasets"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="先下载 HotpotQA 并抽取文档库")
    parser.add_argument("--questions", type=int, default=None,
                        help="用多少个问题的候选文档建库（默认读 config）")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="向量化批大小（内存不够就调小，比如 32）")
    parser.add_argument("--no-resume", action="store_true",
                        help="不从断点续传，删库重建（默认会自动续传）")
    parser.add_argument("--retries", type=int, default=3,
                        help="每批写入 Qdrant 的重试次数（默认 3）")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(name)s: %(message)s")
    quiet_noisy_loggers()

    from multihop_rag.data.load_hotpotqa import (
        extract_corpus,
        load_corpus,
        load_raw,
        save_corpus,
    )
    from multihop_rag.runtime import get_retriever

    n_questions = args.questions or settings.corpus_questions

    # ── 1. 准备文档库 ──
    if args.download or not settings.corpus_path.exists():
        print(f"→ 加载 HotpotQA（split={settings.data_split}），"
              f"抽取随机 {n_questions} 个问题的候选文档...")
        ds = load_raw()   # split 从 config 读（默认 train）
        corpus = extract_corpus(ds, max_questions=n_questions)
        save_corpus(corpus)
    else:
        print(f"→ 读取已有文档库: {settings.corpus_path}")

    chunks = load_corpus()
    print(f"→ 文档数: {len(chunks)}")

    if args.batch_size:
        settings._raw["retrieval"]["batch_size"] = args.batch_size  # noqa: SLF001
        print(f"→ 批大小: {args.batch_size}")

    # ── 2. 建索引 ──
    retriever = get_retriever()
    embedder = retriever.embedder
    print(f"→ Embedding 模型: {embedder.model_name} (dim={embedder.dim})")
    print(f"→ Qdrant: {settings.qdrant_url}（timeout="
          f"{settings.retrieval.get('qdrant_timeout', 120)}s）")

    n_batches = (len(chunks) + settings.retrieval.batch_size - 1) // settings.retrieval.batch_size
    resume = not args.no_resume
    if resume:
        existing = retriever.store.count_safe()
        if existing:
            print(f"→ ⭐ 检测到已有 {existing} 条索引，将【断点续传】（跳过已完成的批次）")
        else:
            print("→ 集合为空，全量建立")
    else:
        print("→ --no-resume：删除已有集合，全量重建")

    print(f"→ 开始向量化并写入 Qdrant（{n_batches} 批，预计几分钟）...")

    retriever.build_index(chunks, resume=resume)

    total = retriever.store.count()
    print(f"✅ 索引完成，共 {total} 条")

    if total == 0:
        print("⚠️ 索引是空的！请检查 Qdrant 连接和上面的报错。", file=sys.stderr)
        sys.exit(1)

    if total < len(chunks):
        print(f"⚠️ 只写入了 {total}/{len(chunks)} 条。"
              f"可以【直接重跑同一条命令】续传剩余部分。", file=sys.stderr)


if __name__ == "__main__":
    main()
