"""配置加载的测试。"""

from __future__ import annotations

import os

import pytest

from scholargraph.config import ConfigError, Settings
from scholargraph.search import build_searcher

LLM_VARIABLES = [
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_THINKING",
    "MAX_SUB_QUESTIONS",
    "MAX_QUERIES_PER_SUB_QUESTION",
    "PAPERS_PER_QUERY",
    "MAX_CANDIDATES_PER_SUB_QUESTION",
    "MAX_FINDINGS_PER_SUB_QUESTION",
    "SEARCH_SOURCES",
    "OPENALEX_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "MAX_RESEARCH_ROUNDS",
    "MAX_REVISIONS",
]


@pytest.fixture
def project_dir(tmp_path, monkeypatch):
    """一个干净的工作目录：没有 .env，环境变量里也没有相关配置。"""
    # 换成一份环境变量的副本：.env 写入的值只影响当前测试，结束后自动还原
    clean_environ = {k: v for k, v in os.environ.items() if k not in LLM_VARIABLES}
    monkeypatch.setattr(os, "environ", clean_environ)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_settings_are_read_from_dotenv_in_the_working_directory(project_dir):
    (project_dir / ".env").write_text(
        "LLM_API_KEY=sk-test\nMAX_RESEARCH_ROUNDS=3\n", encoding="utf-8"
    )

    settings = Settings.from_env()

    assert settings.api_key == "sk-test"
    assert settings.max_research_rounds == 3
    assert settings.model == "deepseek-flash"  # 未设置的项使用默认值


def test_dotenv_saved_with_a_bom_by_windows_notepad_is_read_correctly(project_dir):
    (project_dir / ".env").write_text("LLM_API_KEY=sk-test\n", encoding="utf-8-sig")

    assert Settings.from_env().api_key == "sk-test"


def test_missing_key_is_reported_clearly(project_dir):
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        Settings.from_env()


def test_placeholder_key_from_the_example_file_is_rejected(project_dir):
    (project_dir / ".env").write_text("LLM_API_KEY=sk-在这里填入你的密钥\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        Settings.from_env()


def test_non_numeric_budget_is_reported_clearly(project_dir):
    (project_dir / ".env").write_text(
        "LLM_API_KEY=sk-test\nPAPERS_PER_QUERY=many\n", encoding="utf-8"
    )

    with pytest.raises(ConfigError, match="PAPERS_PER_QUERY"):
        Settings.from_env()


# ───────────── 检索源 ─────────────


def test_default_sources_are_arxiv_and_openalex_without_keys(project_dir):
    (project_dir / ".env").write_text("LLM_API_KEY=sk-test\n", encoding="utf-8")
    settings = Settings.from_env()

    assert settings.search_sources == ("arxiv", "openalex")
    assert settings.openalex_api_key is None and settings.semantic_scholar_api_key is None
    assert list(build_searcher(settings).stats()) == ["arXiv", "OpenAlex"]


def test_sources_and_keys_can_be_configured(project_dir):
    (project_dir / ".env").write_text(
        "LLM_API_KEY=sk-test\n"
        "SEARCH_SOURCES= OpenAlex , semantic_scholar,openalex\n"
        "OPENALEX_API_KEY=oa-key\n"
        "SEMANTIC_SCHOLAR_API_KEY=\n",
        encoding="utf-8",
    )
    settings = Settings.from_env()

    assert settings.search_sources == ("openalex", "semantic_scholar")  # 忽略大小写、空白和重复
    assert settings.openalex_api_key == "oa-key"
    assert settings.semantic_scholar_api_key is None  # 留空等于没有设置
    assert list(build_searcher(settings).stats()) == ["OpenAlex", "Semantic Scholar"]


def test_unknown_source_name_is_reported_clearly(project_dir):
    (project_dir / ".env").write_text(
        "LLM_API_KEY=sk-test\nSEARCH_SOURCES=arxiv,google\n", encoding="utf-8"
    )

    with pytest.raises(ConfigError, match="google"):
        build_searcher(Settings.from_env())
