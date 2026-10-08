"""启动期配置自检。

【为什么需要这个模块 —— 真实线上故障复盘】

部署到服务器后，服务启动完全正常：
    ✅ 容器 Up
    ✅ 页面能打开
    ✅ /health 返回 ok
    ✅ 第 1 跳检索正常（不调 LLM）
只有用户点"提问"时才崩：
    ❌ UnicodeEncodeError: 'ascii' codec can't encode characters in position 10-13

排查过程（花了很久）：
    ① 一开始以为是容器缺 locale，Python 用 ASCII 写中文日志
       → 加了 LANG=C.UTF-8 / PYTHONUTF8=1，没解决
    ② 加代码级兜底 sys.stdout.reconfigure(encoding="utf-8")
       → 还是没解决
    ③ 打完整堆栈才看到真正位置：
         openai/_base_client.py → httpx2.Headers(headers_dict)
         → value.encode("ascii")
       是在【构造 HTTP 请求头】时失败的
    ④ 定位到具体 header：Authorization: Bearer sk-你的真实key
       —— .env 里的 Key 还是中文占位符没替换

根因链条：
    .env.example 写的是   sk-在这里填你的key
    config.py 只校验       key.startswith("sk-在这里")
    但部署命令里写的是     sk-你的真实key      ← 两者不一致！
    → 占位符绕过校验 → 中文进了 HTTP 头 → 崩

结论：**配置错误必须在启动时暴露，而不是等第一次请求才炸。**
     这就是本模块的作用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CheckReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def status(self) -> str:
        if self.errors:
            return "degraded"
        return "ok" if not self.warnings else "warning"


def _check_llm(report: CheckReport) -> None:
    """LLM 配置：Key / base_url / model。配置错误直接是 error。"""
    from .config import settings

    # 这三个属性内部已做校验（非 ASCII、占位符），会抛 RuntimeError
    for name, getter in (
        ("DEEPSEEK_API_KEY", lambda: settings.llm_api_key),
        ("LLM_BASE_URL", lambda: settings.llm_base_url),
        ("LLM_MODEL", lambda: settings.llm_model),
    ):
        try:
            value = getter()
        except RuntimeError as e:
            report.errors.append(f"[LLM] {e}")
            continue
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"[LLM] 读取 {name} 失败: {e}")
            continue

        if name == "DEEPSEEK_API_KEY":
            # 走不到这里的非 ASCII 已经在上面抛了，这里只做软提示
            if not value.startswith("sk-"):
                report.warnings.append(
                    f"[LLM] {name} 不以 'sk-' 开头，确认是不是有效 Key。"
                )
            if len(value) < 20:
                report.warnings.append(
                    f"[LLM] {name} 只有 {len(value)} 个字符，可能不是完整 Key。"
                )


def _check_vector_store(report: CheckReport) -> None:
    """向量库：能连上、集合存在、点数 > 0。"""
    from .config import settings
    from .runtime import get_retriever

    try:
        retriever = get_retriever()
    except Exception as e:  # noqa: BLE001
        report.errors.append(f"[向量库] 初始化失败: {e}")
        return

    if not retriever.store.collection_exists():
        report.errors.append(
            f"[向量库] 集合 '{settings.retrieval.collection}' 不存在。"
            "先建索引：bash scripts/deploy-server.sh index"
        )
        return

    try:
        count = retriever.store.count()
    except Exception as e:  # noqa: BLE001
        report.errors.append(f"[向量库] 统计点数失败: {e}")
        return

    if count == 0:
        report.errors.append(
            f"[向量库] 集合 '{settings.retrieval.collection}' 是空的，检索不会有结果。"
            "先建索引：bash scripts/deploy-server.sh index"
        )
    else:
        logger.info("[自检] 向量库正常，%s 集合共 %d 条", settings.retrieval.collection, count)


def _check_redis(report: CheckReport) -> None:
    """Redis：只做 warning，因为它挂了业务还能降级跑。"""
    from .runtime import get_cache

    try:
        stats = get_cache().stats()
    except Exception as e:  # noqa: BLE001
        report.warnings.append(f"[缓存] 初始化失败，将降级为不缓存: {e}")
        return

    if stats.get("enabled") and not stats.get("connected", True):
        report.warnings.append(f"[缓存] Redis 未连接，检索缓存不生效: {stats}")
    else:
        logger.info("[自检] 缓存状态: %s", stats)


def _check_eval_set(report: CheckReport) -> None:
    """评测集：没有它就没法跑 test / eval（不影响线上问答）。"""
    from .config import settings

    if not settings.eval_set_path.exists():
        report.warnings.append(
            f"[评测] 评测集不存在: {settings.eval_set_path}。"
            "跑评测前先生成：bash scripts/deploy-server.sh evalset"
        )


def run_startup_checks() -> CheckReport:
    """跑一遍全部自检，并把结果打成醒目的日志。"""
    report = CheckReport()

    for fn in (_check_llm, _check_vector_store, _check_redis, _check_eval_set):
        try:
            fn(report)
        except Exception as e:  # noqa: BLE001
            # 自检本身出问题不能影响启动
            report.warnings.append(f"[自检] {fn.__name__} 异常: {e}")

    line = "=" * 68
    if report.ok and not report.warnings:
        logger.info("%s\n启动自检: 全部通过 ✅\n%s", line, line)
    elif report.ok:
        logger.warning(
            "%s\n启动自检: 通过，但有 %d 条提醒 ⚠️\n%s\n%s",
            line, len(report.warnings), "\n".join("  ⚠️ " + w for w in report.warnings), line,
        )
    else:
        logger.error(
            "%s\n"
            "⚠️⚠️  启动自检发现 %d 个【严重问题】—— 服务能起来，但问答会失败  ⚠️⚠️\n"
            "%s\n"
            "%s\n"
            "%s",
            line,
            len(report.errors),
            "\n".join("  ❌ " + e for e in report.errors),
            "\n".join("  ⚠️ " + w for w in report.warnings) if report.warnings else "",
            line,
        )

    return report
