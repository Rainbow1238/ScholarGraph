"""联邦检索的测试：多源融合、单源故障的容错与熔断、缓存。"""

from __future__ import annotations

import pytest
from fakes import FakeSearcher, make_paper

from scholargraph.search import FederatedSearcher, SearchError

PAPER_A = make_paper("2401.00001", "Paper A from arXiv", sources=["arXiv"])
PAPER_B = make_paper("OpenAlex:W2", "Paper B from OpenAlex", sources=["OpenAlex"], venue="ACL")


class AlwaysFailing:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query: str, max_results: int):
        self.calls += 1
        raise SearchError("HTTP 429")


def test_results_from_every_source_are_combined():
    searcher = FederatedSearcher(
        {"arXiv": FakeSearcher([PAPER_A]), "OpenAlex": FakeSearcher([PAPER_B])}
    )
    papers = searcher.search("graph rag", max_results=5)

    assert {paper.paper_id for paper in papers} == {"arXiv:2401.00001", "OpenAlex:W2"}


def test_one_failing_source_does_not_lose_the_results_of_the_others():
    searcher = FederatedSearcher({"arXiv": FakeSearcher([PAPER_A]), "OpenAlex": AlwaysFailing()})

    assert [paper.paper_id for paper in searcher.search("q", 5)] == ["arXiv:2401.00001"]
    assert searcher.stats()["OpenAlex"]["failures"] == 1


def test_source_is_disabled_after_repeated_failures_and_not_called_again():
    failing = AlwaysFailing()
    searcher = FederatedSearcher({"arXiv": FakeSearcher([PAPER_A]), "OpenAlex": failing})

    for query in ("q1", "q2", "q3", "q4"):
        searcher.search(query, 5)

    assert failing.calls == 2  # 连续失败 2 次后熔断，后面的检索不再请求它
    assert searcher.stats()["OpenAlex"]["disabled_reason"] == "HTTP 429"
    assert searcher.stats()["arXiv"]["disabled_reason"] is None


def test_a_success_resets_the_failure_count():
    class FailsEveryOtherTime:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, query: str, max_results: int):
            self.calls += 1
            if self.calls % 2 == 1:
                raise SearchError("偶发错误")
            return [PAPER_B]

    flaky = FailsEveryOtherTime()
    searcher = FederatedSearcher({"arXiv": FakeSearcher([PAPER_A]), "OpenAlex": flaky})

    for query in ("q1", "q2", "q3", "q4"):
        searcher.search(query, 5)

    assert flaky.calls == 4  # 没有连续失败 2 次，所以一直在用
    assert searcher.stats()["OpenAlex"]["disabled_reason"] is None


def test_search_error_is_raised_only_when_every_source_fails():
    searcher = FederatedSearcher({"arXiv": AlwaysFailing(), "OpenAlex": AlwaysFailing()})

    with pytest.raises(SearchError, match="所有检索源上都失败"):
        searcher.search("q1", 5)


def test_clear_error_when_every_source_has_been_disabled():
    searcher = FederatedSearcher({"arXiv": AlwaysFailing()})
    for query in ("q1", "q2"):
        with pytest.raises(SearchError):
            searcher.search(query, 5)

    with pytest.raises(SearchError, match="所有检索源都已停用"):
        searcher.search("q3", 5)


def test_identical_queries_are_served_from_cache():
    arxiv = FakeSearcher([PAPER_A])
    searcher = FederatedSearcher({"arXiv": arxiv})

    first = searcher.search("Graph  RAG", 5)
    second = searcher.search("graph rag", 5)  # 忽略大小写和多余空白后是同一个检索词

    assert first == second
    assert len(arxiv.queries) == 1


def test_failed_queries_are_not_cached():
    arxiv = FakeSearcher([PAPER_A], failing=("q",))
    searcher = FederatedSearcher({"arXiv": arxiv})

    with pytest.raises(SearchError):
        searcher.search("q", 5)
    assert ("q", 5) not in searcher._cache


def test_stats_count_requests_and_papers_per_source():
    searcher = FederatedSearcher(
        {"arXiv": FakeSearcher([PAPER_A]), "OpenAlex": FakeSearcher([PAPER_B])}
    )
    searcher.search("q1", 5)
    searcher.search("q2", 5)

    assert searcher.stats()["arXiv"] == {
        "requests": 2,
        "failures": 0,
        "papers": 2,
        "disabled_reason": None,
    }


def test_at_least_one_source_is_required():
    with pytest.raises(ValueError):
        FederatedSearcher({})
