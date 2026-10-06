"""arXiv 检索器的测试：不联网，只验证查询构造、结果转换、缓存和错误处理。"""

from __future__ import annotations

from datetime import datetime, timezone

import arxiv
import pytest
import requests
from requests.adapters import HTTPAdapter

from scholargraph.search import ArxivSearcher, SearchError
from scholargraph.search.arxiv_search import (
    REQUEST_TIMEOUT_SECONDS,
    build_arxiv_query,
    extract_terms,
    to_paper,
)


def make_arxiv_result() -> arxiv.Result:
    published = datetime(2024, 1, 15, tzinfo=timezone.utc)
    return arxiv.Result(
        entry_id="http://arxiv.org/abs/2401.12345v2",
        updated=published,
        published=published,
        title="Retrieval-Augmented\n  Generation: A Survey",
        authors=[arxiv.Result.Author("Ada Lovelace"), arxiv.Result.Author("Alan Turing")],
        summary="We survey\nretrieval-augmented   generation.",
    )


class FakeArxivClient:
    """替换 `arxiv.Client.results`：记录收到的查询语句，并按规则返回结果。"""

    def __init__(self, respond) -> None:
        self._respond = respond
        self.queries: list[str] = []

    def results(self, request: arxiv.Search):
        self.queries.append(request.query)
        return iter(self._respond(request.query))


def make_searcher(monkeypatch, respond) -> tuple[ArxivSearcher, FakeArxivClient]:
    searcher = ArxivSearcher(min_interval_seconds=0)
    fake_client = FakeArxivClient(respond)
    monkeypatch.setattr(searcher, "_client", fake_client)
    return searcher, fake_client


# ───────────── 查询构造 ─────────────


def test_extract_terms_splits_hyphens_and_drops_stopwords_and_duplicates():
    assert extract_terms("The Multi-Hop reasoning of RAG and rag") == [
        "multi",
        "hop",
        "reasoning",
        "rag",
    ]


def test_extract_terms_drops_boolean_operator_words_and_punctuation():
    assert extract_terms('graph AND "RAG" OR (summarization)') == ["graph", "rag", "summarization"]


def test_extract_terms_ignores_non_english_text():
    assert extract_terms("图检索增强") == []


def test_extract_terms_is_capped():
    assert len(extract_terms("one two three four five six seven eight nine")) == 6


def test_build_arxiv_query_uses_explicit_field_and_operator():
    assert build_arxiv_query(["graph", "rag"], "AND") == "all:graph AND all:rag"
    assert build_arxiv_query(["graph", "rag"], "OR") == "all:graph OR all:rag"


# ───────────── 检索行为 ─────────────


def test_strict_query_is_enough_when_it_returns_results(monkeypatch):
    searcher, client = make_searcher(monkeypatch, lambda query: [make_arxiv_result()])

    papers = searcher.search("graph RAG", max_results=5)

    assert [paper.paper_id for paper in papers] == ["arXiv:2401.12345"]
    assert client.queries == ["all:graph AND all:rag"]


def test_falls_back_to_relaxed_query_when_strict_query_finds_nothing(monkeypatch):
    def respond(query: str):
        return [make_arxiv_result()] if " OR " in query else []

    searcher, client = make_searcher(monkeypatch, respond)

    papers = searcher.search("graph RAG", max_results=5)

    assert len(papers) == 1
    assert client.queries == ["all:graph AND all:rag", "all:graph OR all:rag"]


def test_single_term_query_has_no_relaxed_form(monkeypatch):
    searcher, client = make_searcher(monkeypatch, lambda query: [])

    assert searcher.search("GraphRAG", max_results=5) == []
    assert client.queries == ["all:graphrag"]


def test_query_without_usable_terms_returns_nothing_without_a_request(monkeypatch):
    searcher, client = make_searcher(monkeypatch, lambda query: [make_arxiv_result()])

    assert searcher.search("图检索增强", max_results=5) == []
    assert client.queries == []


def test_network_failure_is_reported_as_search_error(monkeypatch):
    attempts = []

    def respond(query: str):
        attempts.append(query)
        if len(attempts) == 1:
            raise arxiv.HTTPError("http://export.arxiv.org/api/query", 3, 503)
        return [make_arxiv_result()]

    searcher, _ = make_searcher(monkeypatch, respond)

    with pytest.raises(SearchError):
        searcher.search("graph RAG", max_results=5)
    assert len(searcher.search("graph RAG", max_results=5)) == 1  # 下一次检索能成功


# ───────────── 结果转换 ─────────────


def test_to_paper_converts_an_arxiv_result():
    paper = to_paper(make_arxiv_result())

    assert paper.paper_id == "arXiv:2401.12345"
    assert paper.sources == ["arXiv"]
    assert paper.title == "Retrieval-Augmented Generation: A Survey"
    assert paper.abstract == "We survey retrieval-augmented generation."
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert paper.year == 2024
    assert paper.url == "https://arxiv.org/abs/2401.12345"


# ───────────── 超时 ─────────────


def test_requests_to_arxiv_have_a_timeout(monkeypatch):
    """不联网：拦截底层发送函数，检查真实客户端发出的请求是否带上了超时。"""
    sent_timeouts = []

    def fake_send(self, request, **kwargs):
        sent_timeouts.append(kwargs.get("timeout"))
        raise requests.ConnectionError("测试中不允许真实联网")

    monkeypatch.setattr(HTTPAdapter, "send", fake_send)
    searcher = ArxivSearcher(min_interval_seconds=0)

    with pytest.raises(SearchError):
        searcher.search("graph RAG", max_results=5)

    assert sent_timeouts and all(timeout == REQUEST_TIMEOUT_SECONDS for timeout in sent_timeouts)
