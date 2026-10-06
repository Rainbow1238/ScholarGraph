"""Semantic Scholar 检索器的测试：不联网，用文档里的响应格式构造假数据。"""

from __future__ import annotations

import pytest
from http_fakes import FakeResponse, FakeSession

from scholargraph.search import SearchError, SemanticScholarSearcher
from scholargraph.search.semantic_scholar_search import to_paper


def make_record(**overrides) -> dict:
    record = {
        "paperId": "649def34f8be52c8b66281af98ae884c09aef38b",
        "externalIds": {
            "ArXiv": "2303.08896",
            "DOI": "10.18653/v1/2023.ijcnlp-main.58",
            "CorpusId": 257557820,
        },
        "url": "https://www.semanticscholar.org/paper/649def34f8be52c8b66281af98ae884c09aef38b",
        "title": "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection",
        "abstract": "We propose  SelfCheckGPT.",
        "venue": "EMNLP",
        "year": 2023,
        "citationCount": 800,
        "authors": [
            {"authorId": "1", "name": "Potsawee Manakul"},
            {"authorId": "2", "name": "Mark Gales"},
        ],
    }
    return {**record, **overrides}


def test_record_with_an_arxiv_id_gets_the_arxiv_id():
    paper = to_paper(make_record())

    assert paper.paper_id == "arXiv:2303.08896"
    assert paper.url == "https://arxiv.org/abs/2303.08896"
    assert paper.abstract == "We propose SelfCheckGPT."
    assert (paper.venue, paper.year, paper.citation_count) == ("EMNLP", 2023, 800)
    assert paper.authors == ["Potsawee Manakul", "Mark Gales"]
    assert paper.sources == ["Semantic Scholar"]


def test_record_without_an_arxiv_id_uses_the_corpus_id():
    paper = to_paper(make_record(externalIds={"CorpusId": 257557820}))

    assert paper.paper_id == "S2:257557820"
    assert paper.url.startswith("https://www.semanticscholar.org/paper/")
    assert paper.doi is None


@pytest.mark.parametrize(
    "overrides",
    [{"abstract": None}, {"title": None}, {"externalIds": None}, {"externalIds": {}}],
)
def test_unusable_records_are_skipped(overrides):
    assert to_paper(make_record(**overrides)) is None


def test_arxiv_is_not_reported_as_a_venue():
    assert to_paper(make_record(venue="arXiv.org")).venue is None
    assert to_paper(make_record(venue="")).venue is None


def test_search_sends_the_documented_parameters_and_key_header():
    searcher = SemanticScholarSearcher(api_key="secret-key", min_interval_seconds=0)
    searcher._session = session = FakeSession(
        FakeResponse(200, {"total": 1, "data": [make_record()]})
    )

    papers = searcher.search("selfcheckgpt", max_results=5)

    request = session.requests[0]
    assert request["url"] == "https://api.semanticscholar.org/graph/v1/paper/search"
    assert request["params"]["query"] == "selfcheckgpt"
    assert "abstract" in request["params"]["fields"]
    assert request["headers"]["x-api-key"] == "secret-key"
    assert [paper.paper_id for paper in papers] == ["arXiv:2303.08896"]


def test_response_without_data_means_no_results():
    searcher = SemanticScholarSearcher(min_interval_seconds=0)
    searcher._session = FakeSession(FakeResponse(200, {"total": 0, "offset": 0}))

    assert searcher.search("q", 5) == []


def test_rate_limit_is_reported_as_search_error(monkeypatch):
    from scholargraph.search import http

    monkeypatch.setattr(http, "RETRY_WAIT_SECONDS", 0)
    searcher = SemanticScholarSearcher(min_interval_seconds=0)
    searcher._session = FakeSession(FakeResponse(429, text="Too Many Requests"))

    with pytest.raises(SearchError, match="HTTP 429"):
        searcher.search("q", 5)
