"""FastAPI 应用。

启动：
    uvicorn multihop_rag.api.main:app --reload --port 8000

接口：
    GET  /                 演示页面（可以直接在浏览器里看多跳过程）
    GET  /health           健康检查
    GET  /stats            统计（缓存命中、配置）
    GET  /strategies       可用策略列表
    POST /ask              普通问答（同步返回）
    POST /ask/stream       ⭐ SSE 流式，实时推送每一跳的过程
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from .. import __version__
from ..agents import strategies
from ..agents.state import initial_state
from ..config import settings
from ..runtime import get_cache, get_retriever
from ..startup_check import CheckReport, run_startup_checks
from .schemas import (
    AskRequest,
    AskResponse,
    ExampleQuestion,
    HealthResponse,
    StatsResponse,
)

logger = logging.getLogger(__name__)


def _setup_package_logging() -> None:
    """让 multihop_rag.* 的日志真正输出到容器日志。

    ⚠️ 踩过的坑：
        uvicorn 只配置它自己的三个 logger（uvicorn / uvicorn.error /
        uvicorn.access），根 logger 没有 handler。
        这种情况下 Python 回退到 logging.lastResort ——
        而 lastResort 只处理 **WARNING 及以上**。

        结果：启动自检里那句
            logger.info("启动自检: 全部通过 ✅")
        被【静默丢弃】，日志里什么都看不到，
        看起来像"自检根本没执行"。实际执行了，只是输出被吞了。

    解决：给 multihop_rag 这个包 logger 挂上自己的 handler。
        用 MHRAG_LOG_LEVEL 环境变量可以调级别（默认 INFO）。
    """
    level_name = os.getenv("MHRAG_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    pkg_logger = logging.getLogger("multihop_rag")
    pkg_logger.setLevel(level)

    if not pkg_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        pkg_logger.addHandler(handler)
        # 不向根 logger 传播，避免和 lastResort 重复打印
        pkg_logger.propagate = False


_setup_package_logging()

# 启动自检结果，挂在 app.state 上供 /health 读取
_startup_report: CheckReport | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期。

    ⭐ 启动时跑一遍配置自检 —— 这是踩坑后的改进：
       之前 .env 里的 API Key 还是中文占位符，服务照样启动成功、
       页面也能打开，只有用户点"提问"时才抛 UnicodeEncodeError。
       现在启动阶段就把这类配置问题暴露到日志和 /health 里。
    """
    global _startup_report
    _startup_report = run_startup_checks()
    yield


app = FastAPI(
    title="多跳检索问答 Agent",
    description="基于 LangGraph 的自适应多跳检索问答系统",
    version=__version__,
    lifespan=lifespan,
)


# ══════════════════════════════════════════════════════════
#  基础接口
# ══════════════════════════════════════════════════════════
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """健康检查：确认配置、缓存、向量库都正常。

    ⭐ status 的判定：
        degraded = 有配置错误（比如 API Key 没配好）或向量库不可用
                   —— 服务能启动，但问答一定会失败
        ok       = 一切正常
    """
    try:
        cache_info: dict[str, Any] = get_cache().stats()
    except Exception as e:  # noqa: BLE001
        cache_info = {"enabled": False, "error": str(e)}

    try:
        vs_info: dict[str, Any] = {
            "url": settings.qdrant_url,
            "collection": settings.retrieval.collection,
            "count": get_retriever().store.count(),
        }
    except Exception as e:  # noqa: BLE001
        vs_info = {"error": str(e)}

    config_errors = list(_startup_report.errors) if _startup_report else []
    config_warnings = list(_startup_report.warnings) if _startup_report else []

    if config_errors or "error" in vs_info:
        status = "degraded"
    elif config_warnings:
        status = "warning"
    else:
        status = "ok"

    return HealthResponse(
        status=status,
        version=__version__,
        redis=cache_info,
        vector_store=vs_info,
        config_errors=config_errors,
        config_warnings=config_warnings,
    )


@app.get("/examples", response_model=list[ExampleQuestion])
def examples(n: int = 6) -> list[ExampleQuestion]:
    """⭐ 返回评测集里的真实题目，给演示页面做「示例问题」按钮。

    为什么要这个接口：
        向量库里只有 600 个问题对应的候选文档，随便问一个问题检索不到东西，
        页面会显示"无法确定"，看起来像系统坏了。
        给几个【一定能答】的题目，演示体验完全不同。
    """
    import json as _json

    path = settings.eval_set_path
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="评测集不存在。请先生成：bash scripts/deploy-server.sh evalset",
        )

    out: list[ExampleQuestion] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = _json.loads(line)
                out.append(ExampleQuestion(
                    qid=r["qid"],
                    question=r["question"],
                    type=r.get("type", ""),
                    level=r.get("level", ""),
                    answer=r.get("answer", ""),
                ))
                if len(out) >= max(1, min(n, 50)):
                    break
    except Exception as e:  # noqa: BLE001
        logger.exception("读取评测集失败")
        raise HTTPException(status_code=500, detail=f"读取评测集失败: {type(e).__name__}") from e

    return out



@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    return StatsResponse(
        redis=get_cache().stats(),
        strategies=list(strategies.STRATEGIES),
        config={
            "top_k": settings.retrieval.top_k,
            "max_hop": settings.agent.max_hop,
            "embedding_model": settings.retrieval.embedding_model,
            "llm_model": settings.llm_model,
            "cache_enabled": settings.cache_enabled,
        },
    )


@app.get("/strategies")
def list_strategies() -> dict[str, str]:
    return strategies.STRATEGY_LABELS


# ══════════════════════════════════════════════════════════
#  问答接口 1：普通同步
# ══════════════════════════════════════════════════════════
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        result = strategies.run(req.question, req.strategy)
    except Exception as e:  # noqa: BLE001
        logger.exception("处理失败")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        hop=result["hop"],
        stop_reason=result["stop_reason"],
        queries=result["queries"],
        retrieved_titles=result["retrieved_titles"],
        cache_hits=result["cache_hits"],
        retrieval_calls=result["retrieval_calls"],
    )


# ══════════════════════════════════════════════════════════
#  问答接口 2：⭐ SSE 流式（实时推送每一跳的过程）
# ══════════════════════════════════════════════════════════
def _sse(event: str, data: dict[str, Any]) -> str:
    """Server-Sent Events 格式：event: xxx\\ndata: {...}\\n\\n"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _summarize_update(node: str, update: dict[str, Any]) -> dict[str, Any]:
    """把某个节点的 state 更新，压缩成给前端看的小摘要。

    ⭐ 这里刻意只暴露"人能看懂"的字段，不把整个 state 推出去。
    """
    if node == "retrieve":
        new_evs = update.get("evidences", [])
        return {
            "hop": update.get("hop"),
            "query": new_evs[0].from_query if new_evs else None,
            "new_evidence_count": len(new_evs),
            "titles": [e.title for e in new_evs][:5],
        }
    if node == "judge":
        jr = update.get("judge_result")
        return {
            "sufficient": jr.sufficient if jr else None,
            "missing": jr.missing if jr else "",
            "reasoning": jr.reasoning if jr else "",
        }
    if node == "rewrite":
        queries = update.get("queries", [])
        return {
            "new_query": queries[-1] if queries else None,
            "used_entities": update.get("used_entities", []),
        }
    if node == "generate":
        raw = update.get("answer", "")
        return {"answer_raw": raw}
    return update


async def _stream_events(question: str) -> AsyncIterator[str]:
    """核心：用 LangGraph 的 astream 逐节点产出中间状态。

    注意：我们【没有】重写任何检索/编排逻辑，
         只是把 graph 的中间输出转成 SSE 格式。
    """
    from ..agents.graph import get_graph

    yield _sse("start", {"question": question})

    answer = ""
    hop = 0
    try:
        graph = get_graph()
        async for chunk in graph.astream(
            initial_state(question),
            stream_mode="updates",          # ⭐ 每个节点执行完就产出一块
            config={"recursion_limit": 20},
        ):
            for node, update in chunk.items():
                if node == "retrieve":
                    hop = update.get("hop", hop)
                if node == "generate":
                    answer = update.get("answer", "")
                yield _sse(node, _summarize_update(node, update))

        # 用统一的解析函数抽出纯答案
        from ..generation.generator import parse_answer

        yield _sse("done", {"answer": parse_answer(answer), "hop": hop})

    except Exception as e:  # noqa: BLE001
        # 完整堆栈只写服务端日志
        logger.exception("流式处理失败")
        # ⚠️ 不要把手底异常原样透给客户端：
        #    异常信息里可能带内部路径、依赖版本、堆栈等敏感内容。
        #    以前这里直接返回 f"{type(e).__name__}: {e}"，
        #    结果把 UnicodeEncodeError 的完整信息暴露到了浏览器上。
        yield _sse("error", {
            "message": f"服务内部错误（{type(e).__name__}），详情请查看服务端日志",
            "type": type(e).__name__,
        })


@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    """流式问答：把 Agent 每一步决策实时推给客户端。"""
    return StreamingResponse(
        _stream_events(req.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ══════════════════════════════════════════════════════════
#  演示页面（单文件，不依赖前端框架）
# ══════════════════════════════════════════════════════════
DEMO_HTML = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>多跳检索问答 Agent</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 860px; margin: 40px auto; padding: 0 16px; line-height: 1.7; }
  h1 { font-size: 22px; }
  .row { display: flex; gap: 8px; margin: 16px 0; }
  input[type=text] { flex: 1; padding: 10px; font-size: 15px; border: 1px solid #ccc; border-radius: 6px; }
  select, button { padding: 10px 16px; font-size: 15px; border-radius: 6px; border: 1px solid #ccc; cursor: pointer; }
  button { background: #2563eb; color: #fff; border-color: #2563eb; }
  button:disabled { background: #94a3b8; border-color: #94a3b8; cursor: not-allowed; }
  #log { border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; min-height: 200px; background: #f8fafc; }
  .ev { margin: 10px 0; padding: 10px 12px; border-radius: 6px; border-left: 4px solid #94a3b8; background: #fff; font-size: 14px; }
  .ev b { display: inline-block; min-width: 78px; }
  .ev.retrieve { border-left-color: #3b82f6; }
  .ev.judge    { border-left-color: #f59e0b; }
  .ev.rewrite  { border-left-color: #8b5cf6; }
  .ev.generate { border-left-color: #10b981; }
  .ev.done     { border-left-color: #059669; background: #ecfdf5; font-weight: 600; }
  .ev.error    { border-left-color: #ef4444; background: #fef2f2; }
  code { background: #f1f5f9; padding: 1px 5px; border-radius: 3px; }
  .hint { color: #64748b; font-size: 13px; }
  .examples { margin: 4px 0 16px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  .ex { background: #fff; color: #1e293b; border: 1px solid #cbd5e1; font-size: 12.5px;
        padding: 5px 10px; border-radius: 14px; }
  .ex:hover { background: #eff6ff; border-color: #93c5fd; }
  .notice { background: #fffbeb; border: 1px solid #fde68a; color: #92400e;
            font-size: 13px; padding: 8px 12px; border-radius: 6px; margin: 0 0 16px; }
</style>
</head>
<body>
<h1>🔍 多跳检索问答 Agent</h1>
<p class="hint">下面能看到 Agent 的每一步决策：检索 → 判断证据是否充分 → 改写查询 → 再检索 → 生成答案。</p>
<p class="notice">
  ⚠️ 向量库里是 HotpotQA <b>600 道题</b>对应的候选文档（5881 篇），<b>不是完整维基百科</b>。
  随意提问很可能检索不到内容 —— 请点下面的<b>示例问题</b>（都来自评测集，一定能答）。
</p>

<div class="row">
  <input type="text" id="q" placeholder="输入问题，或点下面的示例问题">
  <select id="s">
    <option value="adaptive">自适应多跳</option>
    <option value="fixed_2hop">固定两跳</option>
    <option value="single">单跳</option>
  </select>
  <button id="go">提问</button>
</div>

<div id="examples" class="examples"><span class="hint">正在加载示例问题…</span></div>

<div id="log"><span class="hint">等待提问…</span></div>

<script>
const $ = id => document.getElementById(id);
const log = $('log');

function add(cls, label, text) {
  const d = document.createElement('div');
  d.className = 'ev ' + cls;
  d.innerHTML = `<b>${label}</b>${text}`;
  log.appendChild(d);
}

$('go').onclick = async () => {
  const question = $('q').value.trim();
  if (!question) return;
  log.innerHTML = '';
  $('go').disabled = true;

  try {
    const resp = await fetch('/ask/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question, strategy: $('s').value}),
    });
    if (!resp.ok) { add('error', '错误', await resp.text()); return; }

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const blocks = buf.split('\\n\\n');
      buf = blocks.pop();
      for (const blk of blocks) {
        const ev = (blk.match(/^event: (.+)$/m) || [])[1];
        const raw = (blk.match(/^data: (.+)$/m) || [])[1];
        if (!ev || !raw) continue;
        const d = JSON.parse(raw);
        if (ev === 'retrieve') add('retrieve', `第 ${d.hop} 跳检索`,
            `查询 <code>${d.query}</code><br>新增证据 ${d.new_evidence_count} 条：${(d.titles||[]).join(' · ')}`);
        else if (ev === 'judge') add('judge', '证据判断',
            `${d.sufficient ? '✅ 够了' : '❌ 不够'} — ${d.reasoning}` + (d.missing ? `<br>缺少：${d.missing}` : ''));
        else if (ev === 'rewrite') add('rewrite', '改写查询',
            `新查询 <code>${d.new_query}</code>` + (d.used_entities?.length ? `<br>用到实体：${d.used_entities.join(', ')}` : ''));
        else if (ev === 'generate') add('generate', '生成答案', '正在生成…');
        else if (ev === 'done') add('done', '✅ 最终答案', `<span style="font-size:17px">${d.answer}</span>（共 ${d.hop} 跳）`);
        else if (ev === 'error') add('error', '错误', d.message);
      }
    }
  } catch (e) {
    add('error', '请求失败', e.message);
  } finally {
    $('go').disabled = false;
  }
};

$('q').addEventListener('keydown', e => { if (e.key === 'Enter') $('go').click(); });

// ⭐ 加载示例问题（来自评测集，保证能答出来）
//    为什么要这个：向量库里只有 600 道题对应的文档，随便问会检索不到，
//    页面显示"cannot determine"看起来像系统坏了。给几个必答的题目，
//    演示体验完全不同。
(async () => {
  const box = $('examples');
  try {
    const resp = await fetch('/examples?n=6');
    if (!resp.ok) throw new Error(await resp.text());
    const rows = await resp.json();
    if (!rows.length) throw new Error('评测集是空的');
    box.innerHTML = '<span class="hint">示例问题（点一下填入）：</span>';
    for (const r of rows) {
      const b = document.createElement('button');
      b.className = 'ex';
      b.textContent = r.question.length > 58 ? r.question.slice(0, 58) + '…' : r.question;
      b.title = `[${r.type}/${r.level}] ${r.question}\\n标准答案：${r.answer}`;
      b.onclick = () => { $('q').value = r.question; $('q').focus(); };
      box.appendChild(b);
    }
    // 默认填第一道
    $('q').value = rows[0].question;
  } catch (e) {
    box.innerHTML = '<span class="hint">示例问题加载失败：' + e.message +
                    '（如果是"评测集不存在"，在服务器执行：bash scripts/deploy-server.sh evalset）</span>';
  }
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def demo() -> HTMLResponse:
    """演示页面：在浏览器里直接看多跳过程"""
    return HTMLResponse(DEMO_HTML)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
