"""检索结果磁盘缓存的测试。"""

from __future__ import annotations

import pytest
from fakes import FakeSearcher, make_paper

from scholargraph.search import SearchError
from scholargraph.search.cache import CachedSearcher

PAPER = make_paper("2401.00001", "Cached paper", venue="ACL", citation_count=7)


def test_second_identical_query_is_served_from_disk(tmp_path):
    inner = FakeSearcher([PAPER])
    cached = CachedSearcher(inner, tmp_path / "cache.sqlite")

    first = cached.search("Graph  RAG", 5)
    second = cached.search("graph rag", 5)  # 忽略大小写和多余空白

    assert first == second == [PAPER]  # 论文的全部字段都能从缓存里还原
    assert len(inner.queries) == 1
    assert (cached.hits, cached.misses) == (1, 1)


def test_cache_survives_a_restart(tmp_path):
    path = tmp_path / "nested" / "cache.sqlite"  # 目录不存在时自动创建
    first_run = CachedSearcher(FakeSearcher([PAPER]), path)
    first_run.search("graph rag", 5)
    first_run.close()

    inner = FakeSearcher([])
    second_run = CachedSearcher(inner, path)

    assert second_run.search("graph rag", 5) == [PAPER]
    assert inner.queries == []


def test_different_result_sizes_and_namespaces_are_cached_separately(tmp_path):
    path = tmp_path / "cache.sqlite"
    inner = FakeSearcher([PAPER])
    cached = CachedSearcher(inner, path, namespace="arxiv,openalex")
    cached.search("graph rag", 5)
    cached.search("graph rag", 10)
    cached.close()

    other_sources = CachedSearcher(inner, path, namespace="openalex")
    other_sources.search("graph rag", 5)  # 换了检索源组合，不能复用旧缓存

    assert len(inner.queries) == 3


def test_failed_search_is_not_cached(tmp_path):
    inner = FakeSearcher([PAPER], failing=("graph rag",))
    cached = CachedSearcher(inner, tmp_path / "cache.sqlite")

    for _ in range(2):
        with pytest.raises(SearchError):
            cached.search("graph rag", 5)
    assert len(inner.queries) == 2  # 每次都重新尝试
