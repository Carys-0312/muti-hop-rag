"""LLM 调用层：judge / rewrite / generate 三条链。

设计要点：
1. judge 和 rewrite 用 with_structured_output 拿结构化结果
2. generate 用普通文本输出（HotpotQA 答案本来就是短文本）
3. 所有链在这里统一构造，nodes.py 只管编排
"""

from __future__ import annotations

import logging
import re

from ..config import settings
from ..runtime import get_llm
from ..agents.prompts import (
    GENERATE_PROMPT,
    JUDGE_PROMPT,
    REWRITE_PROMPT,
    SIMPLE_REWRITE_PROMPT,
)
from ..agents.state import JudgeResult, RewriteResult

logger = logging.getLogger(__name__)


def _chain(prompt_str: str, structured=None):
    """构造 prompt | llm 链。

    ⭐ 结构化输出必须指定 method="function_calling"：
        LangChain 的 with_structured_output 默认用 OpenAI 的
        response_format={"type": "json_schema"}，
        但 DeepSeek【不支持】这种格式，会返回 400：
            "This response_format type is unavailable now"

        改成 function_calling（走 tools 机制），DeepSeek 就支持了。
        方法可以在 configs/default.yaml 的 agent.structured_output_method 里改：
            function_calling | json_mode | json_schema
    """
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(prompt_str)
    llm = get_llm()

    if structured is None:
        return prompt | llm

    method = settings.agent.structured_output_method
    return prompt | llm.with_structured_output(structured, method=method)


# ── 三条链（懒构造，避免 import 时就要求有 API Key）──
def judge_chain():
    return _chain(JUDGE_PROMPT, structured=JudgeResult)


def rewrite_chain():
    return _chain(REWRITE_PROMPT, structured=RewriteResult)


def simple_rewrite_chain():
    return _chain(SIMPLE_REWRITE_PROMPT)


def generate_chain():
    return _chain(GENERATE_PROMPT)


# ── 对外函数 ──
def judge(question: str, evidences_text: str) -> JudgeResult:
    return judge_chain().invoke({"question": question, "evidences": evidences_text})


def rewrite(
    question: str, evidences_text: str, missing: str, used_queries: list[str]
) -> RewriteResult:
    used = "\n".join(f"- {q}" for q in used_queries) or "（无）"
    return rewrite_chain().invoke(
        {
            "question": question,
            "evidences": evidences_text,
            "missing": missing,
            "used_queries": used,
        }
    )


def simple_rewrite(question: str, evidences_text: str) -> str:
    """固定两跳策略用：不做判断，直接改写一次"""
    result = simple_rewrite_chain().invoke(
        {"question": question, "evidences": evidences_text}
    )
    return (result.content or "").strip()


def generate(question: str, evidences_text: str) -> str:
    result = generate_chain().invoke({"question": question, "evidences": evidences_text})
    return (result.content or "").strip()


def parse_answer(raw: str) -> str:
    """从 generate 的输出里抽出纯答案，并做规范化后处理。

    输出格式约定是：
        答案：xxx
        依据：yyy
    → 抽出 "xxx"

    ⭐ 后处理（根据错误分析加的）：
      200 条实验里发现一大批"答案对但格式不对"的错误，例如
          'Yes. The Conewago Canal enabled rivercraft to...'  vs 标准 'yes'
          'It consists of 1,340,000 square feet'             vs 标准 '1340000 sqft'
      所以这里做两步规范化：
        ① 是/否类：只保留开头的 yes / no
        ② 太长的：截到第一个句子
      这是标准的"答案规范化"，真实问答系统都会做（否则 EM 会严重低估）。
    """
    if not raw:
        return ""

    # ── ① 抽出 "答案：xxx" 之后的内容 ──
    ans = ""
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("答案"):
            ans = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            break
    if not ans:
        ans = raw.splitlines()[0].strip() if raw.splitlines() else ""

    # ── ② 去掉包裹的引号和多余空白 ──
    ans = ans.strip().strip('"\u201c\u201d\'').strip()
    if not ans:
        return ""

    # ── ③ 是/否类：只保留 yes / no ──
    m = re.match(r"^\s*(yes|no|true|false)\b", ans, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # ── ④ 太长就截到第一个句子（避免把解释当成答案）──
    if len(ans) > 120:
        # 按句号/分号/换行切，取第一段
        first = re.split(r"(?<=[.;])\s+|\n", ans, maxsplit=1)[0].strip()
        if first:
            ans = first

    # ── ⑤ 去掉常见的解释性前缀 ──
    ans = re.sub(
        r"^(the answer is|it is|answer:|答案是|根据证据)[:：\s]*",
        "", ans, flags=re.IGNORECASE,
    ).strip()

    return ans
