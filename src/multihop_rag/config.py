"""配置加载：configs/default.yaml + 环境变量覆盖。

用法：
    from multihop_rag.config import settings
    settings.top_k          # 5
    settings.agent.max_hop  # 3
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# 项目根目录。
# 默认用 src/multihop_rag/config.py → 上三层。
# 但容器里如果装成非 editable 包，__file__ 会落在 site-packages，
# 所以支持用环境变量 MHRAG_ROOT 显式指定（Dockerfile 里就设了这个）。
PROJECT_ROOT = Path(os.getenv("MHRAG_ROOT") or Path(__file__).resolve().parents[2]).resolve()
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

load_dotenv(PROJECT_ROOT / ".env")


# ⭐ 所有"占位符"写法都要列进来。
#    教训：之前只检查了一个字符串，而文档和命令里出现过另一种写法，
#    导致占位符绕过了校验、一直跑到运行时才炸（详见 llm_api_key 的注释）。
_KEY_PLACEHOLDERS = (
    "在这里",      # .env.example 里的写法
    "你的真实",    # 部署命令里出现过的写法
    "你的key",
    "你的Key",
    "换成",
    "填你",
    "xxx",
    "your-key",
    "your_key",
    "placeholder",
)


class _Section:
    """支持点号访问的配置段：settings.retrieval.top_k"""

    def __init__(self, data: dict[str, Any]):
        self._data = data or {}

    def __getattr__(self, name: str) -> Any:
        if name not in self._data:
            raise AttributeError(f"配置项不存在: {name}（可用: {list(self._data)}）")
        value = self._data[name]
        return _Section(value) if isinstance(value, dict) else value

    def get(self, name: str, default: Any = None) -> Any:
        """带默认值的读取，配置项缺失时不报错。

        用于"新增的可选配置"——老配置文件里没有这个键也能跑。
        """
        if name not in self._data:
            return default
        value = self._data[name]
        return _Section(value) if isinstance(value, dict) else value

    def __repr__(self) -> str:
        return f"_Section({self._data})"


class Settings:
    def __init__(self, path: Path = CONFIG_PATH):
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        self._raw = raw

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._raw:
            raise AttributeError(f"配置段不存在: {name}")
        value = self._raw[name]
        return _Section(value) if isinstance(value, dict) else value

    # ── 路径便捷属性（都返回绝对路径）──
    @property
    def root(self) -> Path:
        return PROJECT_ROOT

    @property
    def eval_set_path(self) -> Path:
        return PROJECT_ROOT / self._raw["eval"]["eval_set"]

    @property
    def results_dir(self) -> Path:
        return PROJECT_ROOT / self._raw["eval"]["results_dir"]

    @property
    def corpus_path(self) -> Path:
        return PROJECT_ROOT / self._raw["data"]["corpus_file"]

    @property
    def raw_dir(self) -> Path:
        return PROJECT_ROOT / self._raw["data"]["raw_dir"]

    @property
    def processed_dir(self) -> Path:
        return PROJECT_ROOT / self._raw["data"]["processed_dir"]

    @property
    def eval_dir(self) -> Path:
        return PROJECT_ROOT / self._raw["data"]["eval_dir"]

    @property
    def corpus_questions(self) -> int:
        """用多少个问题来构建文档库（同时约束评测集的取样范围）"""
        return int(self._raw["data"].get("corpus_questions", 600))

    @property
    def data_split(self) -> str:
        """用哪个 split。

        ⭐ 默认 train，因为 HotpotQA 的 validation split 里 level 整列都是 hard，
           做不了分难度分析（实测：validation 7405 条全部 level='hard'）。
        """
        return self._raw["data"].get("split", "train")

    # ── 环境变量（带默认值）──
    @property
    def llm_api_key(self) -> str:
        key = (os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()

        if not key:
            raise RuntimeError(
                "未配置 API Key。请复制 .env.example 为 .env，"
                "并填入 DEEPSEEK_API_KEY（https://platform.deepseek.com/）。"
            )

        # ⭐ ① 非 ASCII 检查（最关键，踩过坑）
        #
        #    API Key 会被 OpenAI SDK 拼进 HTTP 请求头：
        #        Authorization: Bearer sk-xxxxx
        #    而 HTTP 头【只能是 ASCII】。Key 里只要有一个中文字符，
        #    调用 LLM 时就会抛：
        #        UnicodeEncodeError: 'ascii' codec can't encode characters
        #                            in position 10-13
        #
        #    这个报错完全看不出是 Key 的问题（"position 10-13" 是天书），
        #    所以必须在这里提前拦掉并给出人话提示。
        bad_chars = sorted({c for c in key if ord(c) >= 128})
        if bad_chars:
            raise RuntimeError(
                "DEEPSEEK_API_KEY 含非 ASCII 字符（%s）。\n"
                "API Key 只能是 ASCII 字符（英文字母 + 数字 + 短横线）。\n"
                "最常见原因：.env 里还是中文占位符没替换成真实 Key。\n"
                "真实 Key 形如 sk-a1b2c3d4e5f6...，在 "
                "https://platform.deepseek.com/ 的 API Keys 页面复制。"
                % "、".join(bad_chars[:5])
            )

        # ② 占位符检查
        #    ⚠️ 这里必须覆盖【所有出现过的占位符写法】。
        #    之前只检查 "sk-在这里"，而文档/命令里出现过 "sk-你的真实key"，
        #    两者不一致 → 校验被绕过 → 中文占位符溜进了运行时。
        for ph in _KEY_PLACEHOLDERS:
            if ph in key:
                raise RuntimeError(
                    f"DEEPSEEK_API_KEY 里还含有占位符「{ph}」，"
                    "请替换成 https://platform.deepseek.com/ 上的真实 Key。"
                )

        return key

    @property
    def llm_base_url(self) -> str:
        url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").strip()
        if any(ord(c) >= 128 for c in url):
            raise RuntimeError(f"LLM_BASE_URL 含非 ASCII 字符: {url!r}")
        return url

    @property
    def llm_model(self) -> str:
        model = os.getenv("LLM_MODEL", "deepseek-chat").strip()
        if any(ord(c) >= 128 for c in model):
            raise RuntimeError(f"LLM_MODEL 含非 ASCII 字符: {model!r}")
        return model

    @property
    def redis_url(self) -> str:
        return os.getenv("REDIS_URL", "redis://localhost:6379/0")

    @property
    def qdrant_url(self) -> str:
        return os.getenv("QDRANT_URL", "http://localhost:6333")

    @property
    def cache_enabled(self) -> bool:
        """环境变量 CACHE_ENABLED=false 可以强制关缓存（用于对比实验）"""
        env = os.getenv("CACHE_ENABLED")
        if env is not None:
            return env.strip().lower() not in ("false", "0", "no")
        return bool(self._raw["cache"]["enabled"])


settings = Settings()
