"""多跳检索问答 Agent

基于 LangGraph 的自适应多跳检索问答系统。
"""

from __future__ import annotations

import sys

__version__ = "0.1.0"


def _force_utf8_streams() -> None:
    """强制把 stdout / stderr 切换成 UTF-8 输出。

    ⚠️ 为什么必须做这件事（真实线上 bug）：
        python:3.12-slim 这类镜像为了体积不带 locale 数据，
        容器里 LANG 为空 → locale 退化成 POSIX/C
        → Python 启动时把 sys.stdout.encoding 设成 'ascii'。

        此时任何"写中文"的动作都会抛：
            UnicodeEncodeError: 'ascii' codec can't encode characters in position 10-13

        更糟的是 Python logging 的 handleError 只捕获 OSError，
        而 UnicodeEncodeError 是 ValueError 的子类，
        于是异常会一路冒泡，把业务代码（比如 rewrite_node）整个打断。

    修法（两层，这里做第二层）：
        第一层：环境变量 LANG=C.UTF-8 / LC_ALL=C.UTF-8 / PYTHONUTF8=1
                （见 Dockerfile 和 deploy-server.sh）
        第二层：这层代码兜底 —— 即使部署环境忘了设环境变量也不会崩。

    errors="replace" 保证极端情况下最多把字符显示成 '?'，而不是抛异常。
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            # 被重定向成非 TextIOWrapper 时没有 reconfigure，放弃即可
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            # 兜底逻辑本身绝不能影响启动
            pass


_force_utf8_streams()
