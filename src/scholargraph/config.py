"""运行配置：全项目唯一读取环境变量的地方。

其他模块只接收 `Settings` 对象，不直接碰 `os.environ`，
这样测试时可以随手构造一份配置，不依赖本机环境。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


class ConfigError(RuntimeError):
    """配置缺失或不合法。"""


@dataclass(frozen=True)
class Settings:
    """一次调研运行所需的全部配置。"""

    # —— 大模型 ——
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    thinking: str = "disabled"  # "enabled" / "disabled" / ""（不发送该参数）
    temperature: float = 0.2

    # —— 文献检索 ——
    search_sources: tuple[str, ...] = (
        "arxiv",
        "openalex",
    )  # 可选：arxiv / openalex / semantic_scholar
    openalex_api_key: str | None = None  # 可选，免费注册后每日额度更高
    semantic_scholar_api_key: str | None = None  # 可选

    # —— 工作流预算：每个回环都有上限，保证图一定会终止 ——
    max_sub_questions: int = 4
    max_queries_per_sub_question: int = 3
    papers_per_query: int = 6  # 每组检索词从每个检索源各取几篇
    max_candidates_per_sub_question: int = 15  # 融合去重后，交给模型阅读的候选论文上限
    max_findings_per_sub_question: int = 8  # 每个子问题最多保留几条发现
    max_research_rounds: int = 2
    max_revisions: int = 1
    verify_batch_size: int = 15  # 引用校验时每次交给模型的论断数

    # —— 本地文件 ——
    # 状态结构有不兼容的改动时递增文件名里的版本号，避免新代码读到旧格式的 checkpoint
    checkpoint_path: Path = Path("checkpoints/scholargraph-v3.sqlite")
    output_dir: Path = Path("outputs")

    @classmethod
    def from_env(cls) -> Settings:
        """从当前目录（或其上级目录）的 `.env` 文件和环境变量构造配置。"""
        # utf-8-sig：Windows 记事本保存的文件开头可能带 BOM，不处理的话第一行的变量名会读错
        load_dotenv(find_dotenv(usecwd=True), encoding="utf-8-sig")

        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key or "在这里填入" in api_key:
            raise ConfigError("未找到 LLM_API_KEY：请把 .env.example 复制为 .env 并填入密钥。")

        defaults = cls(api_key=api_key)
        return cls(
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", defaults.base_url),
            model=os.getenv("LLM_MODEL", defaults.model),
            thinking=os.getenv("LLM_THINKING", defaults.thinking).strip(),
            search_sources=_name_list("SEARCH_SOURCES", defaults.search_sources),
            openalex_api_key=_optional("OPENALEX_API_KEY"),
            semantic_scholar_api_key=_optional("SEMANTIC_SCHOLAR_API_KEY"),
            max_sub_questions=_positive_int("MAX_SUB_QUESTIONS", defaults.max_sub_questions),
            max_queries_per_sub_question=_positive_int(
                "MAX_QUERIES_PER_SUB_QUESTION", defaults.max_queries_per_sub_question
            ),
            papers_per_query=_positive_int("PAPERS_PER_QUERY", defaults.papers_per_query),
            max_candidates_per_sub_question=_positive_int(
                "MAX_CANDIDATES_PER_SUB_QUESTION", defaults.max_candidates_per_sub_question
            ),
            max_findings_per_sub_question=_positive_int(
                "MAX_FINDINGS_PER_SUB_QUESTION", defaults.max_findings_per_sub_question
            ),
            max_research_rounds=_positive_int("MAX_RESEARCH_ROUNDS", defaults.max_research_rounds),
            max_revisions=_non_negative_int("MAX_REVISIONS", defaults.max_revisions),
        )


def _optional(name: str) -> str | None:
    """未设置或留空时返回 None。"""
    return os.getenv(name, "").strip() or None


def _name_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """解析逗号分隔的名称列表，例如 `arxiv, openalex`；去重并统一成小写。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    names = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not names:
        raise ConfigError(f"环境变量 {name} 里没有任何有效的名称。")
    return tuple(dict.fromkeys(names))


def _positive_int(name: str, default: int) -> int:
    value = _non_negative_int(name, default)
    if value == 0:
        raise ConfigError(f"环境变量 {name} 必须大于 0。")
    return value


def _non_negative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ConfigError(f"环境变量 {name} 必须是整数，当前值为 {raw!r}。") from error
    if value < 0:
        raise ConfigError(f"环境变量 {name} 不能为负数。")
    return value
