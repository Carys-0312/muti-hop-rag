"""打印当前生效的配置 —— 跑实验前用它确认参数改对了。

用法：
    python scripts/show_config.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multihop_rag.config import settings  # noqa: E402


def main() -> None:
    a = settings.agent
    r = settings.retrieval
    d = settings.data

    print("=" * 66)
    print(" 当前生效配置")
    print("=" * 66)

    print("\n【Agent 多跳】")
    print(f"  min_hop                  : {a.get('min_hop', 1)}     ← 1=信任 judge，2=强制最少跳2次")
    print(f"  max_hop                  : {a.max_hop}")
    print(f"  temperature              : {a.temperature}")
    print(f"  evidence_snippet_chars   : {a.get('evidence_snippet_chars')}   ← 单条证据截断长度")
    print(f"  max_evidence_chars       : {a.get('max_evidence_chars')}  ← 证据总长度上限")
    print(f"  structured_output_method : {a.get('structured_output_method')}")

    print("\n【检索】")
    print(f"  top_k                    : {r.top_k}")
    print(f"  embedding_model          : {r.embedding_model}")
    print(f"  collection               : {r.collection}")

    print("\n【数据】")
    print(f"  split                    : {d.get('split')}")
    print(f"  corpus_questions         : {d.get('corpus_questions')}")
    print(f"  corpus_file              : {settings.corpus_path}")

    print("\n【LLM】")
    print(f"  model                    : {settings.llm_model}")
    print(f"  base_url                 : {settings.llm_base_url}")

    print("\n【评测】")
    print(f"  eval_set                 : {settings.eval_set_path}")
    print(f"  sample_size              : {settings.eval.sample_size}")

    print("\n" + "=" * 66)
    print(" ⚠️ 跑实验前请确认：min_hop 和 evidence_snippet_chars 是你想要的值")
    print("=" * 66)


if __name__ == "__main__":
    main()
