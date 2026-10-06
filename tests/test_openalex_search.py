"""OpenAlex 检索器的测试：不联网，用文档里的响应格式构造假数据。"""

from __future__ import annotations

import pytest
from http_fakes import FakeResponse, FakeSession

from scholargraph.search import OpenAlexSearcher, SearchError
from scholargraph.search.openalex_search import rebuild_abstract, to_paper


def make_work(**overrides) -> dict:
    """一条期刊/会议论文的 work 记录（字段名与 OpenAlex 文档一致）。"""
    work = {
        "id": "https://openalex.org/W4385245566",
        "doi": "https://doi.org/10.18653/v1/2023.acl-long.1",
        "display_name": "Chain-of-Verification  Reduces Hallucination",
        "publication_year": 2023,
        "cited_by_count": 412,
        "type": "article",
        "is_retracted": False,
        "authorships": [
            {"author": {"id": "https://openalex.org/A1", "display_name": "Ada Lovelace"}},
            {"author": {"id": "https://openalex.org/A2", "display_name": "Alan Turing"}},
        ],
        "abstract_inverted_index": {"We": [0], "verify": [1, 4], "answers": [2], "and": [3]},
        "primary_location": {
            "landing_page_url": "https://aclanthology.org/2023.acl-long.1",
            "pdf_url": None,
            "source": {"display_name": "ACL", "type": "conference"},
        },
        "locations": [],
    }
    return {**work, **overrides}


def make_searcher(*responses) -> tuple[OpenAlexSearcher, FakeSession]:
    searcher = OpenAlexSearcher(api_key="secret-key", min_interval_seconds=0)
    session = FakeSession(*responses)
    searcher._session = session
    return searcher, session


# ───────────── 记录转换 ─────────────


def test_rebuild_abstract_restores_word_order_from_the_inverted_index():
    assert rebuild_abstract({"world": [1], "hello": [0, 2]}) == "hello world hello"
    assert rebuild_abstract(None) == ""


def test_published_work_without_arxiv_copy_gets_an_openalex_id():
    paper = to_paper(make_work())

    assert paper.paper_id == "OpenAlex:W4385245566"
    assert paper.title == "Chain-of-Verification Reduces Hallucination"
    assert paper.abstract == "We verify answers and verify"
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert (paper.year, paper.venue, paper.citation_count) == (2023, "ACL", 412)
    assert paper.doi == "10.18653/v1/2023.acl-long.1"
    assert paper.url == "https://doi.org/10.18653/v1/2023.acl-long.1"
    assert paper.sources == ["OpenAlex"]


def test_work_with_an_arxiv_doi_gets_the_arxiv_id():
    paper = to_paper(make_work(doi="https://doi.org/10.48550/arXiv.2309.11495"))

    assert paper.paper_id == "arXiv:2309.11495"
    assert paper.url == "https://arxiv.org/abs/2309.11495"


def test_work_with_an_arxiv_copy_among_its_locations_gets_the_arxiv_id():
    locations = [{"landing_page_url": "https://arxiv.org/abs/2309.11495v2", "pdf_url": None}]
    paper = to_paper(make_work(locations=locations))

    assert paper.paper_id == "arXiv:2309.11495"  # 这样能与 arXiv 检索到的同一篇论文合并
    assert paper.venue == "ACL"  # 正式发表的出处仍然保留


def test_arxiv_pdf_link_is_also_recognised():
    locations = [{"landing_page_url": None, "pdf_url": "https://arxiv.org/pdf/2309.11495v1.pdf"}]
    assert to_paper(make_work(locations=locations)).paper_id == "arXiv:2309.11495"


def test_preprint_repository_is_not_reported_as_a_venue():
    primary = {"source": {"display_name": "arXiv (Cornell University)", "type": "repository"}}
    assert to_paper(make_work(primary_location=primary)).venue is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"abstract_inverted_index": None},  # 没有摘要：无法提炼发现和校验引用
        {"display_name": None},
        {"is_retracted": True},
        {"type": "paratext"},
        {"id": None},
    ],
)
def test_unusable_records_are_skipped(overrides):
    assert to_paper(make_work(**overrides)) is None


def test_missing_optional_fields_do_not_break_conversion():
    sparse = make_work(
        doi=None,
        authorships=None,
        primary_location=None,
        locations=None,
        cited_by_count=None,
        publication_year=None,
    )
    paper = to_paper(sparse)

    assert paper.paper_id == "OpenAlex:W4385245566"
    assert (paper.authors, paper.venue, paper.citation_count, paper.year) == ([], None, None, None)
    assert paper.url == "https://openalex.org/W4385245566"


# ───────────── 检索 ─────────────


def test_search_sends_the_documented_parameters_and_keeps_the_key_out_of_the_url():
    searcher, session = make_searcher(FakeResponse(200, {"results": [make_work()]}))

    papers = searcher.search("chain of verification", max_results=5)

    request = session.requests[0]
    assert request["url"] == "https://api.openalex.org/works"
    assert request["params"]["search"] == "chain of verification"
    assert "abstract_inverted_index" in request["params"]["select"]
    assert "api_key" not in request["params"]  # 密钥不进 URL，出错信息里就不会带出密钥
    assert request["headers"]["Authorization"] == "Bearer secret-key"
    assert [paper.paper_id for paper in papers] == ["OpenAlex:W4385245566"]


def test_search_skips_unusable_records_and_respects_the_limit():
    works = [make_work(abstract_inverted_index=None)] + [
        make_work(id=f"https://openalex.org/W{n}") for n in range(1, 6)
    ]
    searcher, _ = make_searcher(FakeResponse(200, {"results": works}))

    assert [paper.paper_id for paper in searcher.search("q", max_results=3)] == [
        "OpenAlex:W1",
        "OpenAlex:W2",
        "OpenAlex:W3",
    ]


def test_search_works_without_an_api_key():
    searcher = OpenAlexSearcher(min_interval_seconds=0)
    searcher._session = session = FakeSession(FakeResponse(200, {"results": []}))

    assert searcher.search("q", 5) == []
    assert "Authorization" not in session.requests[0]["headers"]


def test_unexpected_response_shape_is_reported_as_search_error():
    searcher, _ = make_searcher(FakeResponse(200, {"error": "something"}))

    with pytest.raises(SearchError):
        searcher.search("q", 5)


def test_journal_listed_after_a_repository_copy_is_still_the_venue():
    """OpenAlex 常把 PubMed Central 的开放获取副本列为主要出处，期刊排在其余出处里。"""
    pmc = {"source": {"display_name": "PubMed Central", "type": "repository"}}
    journal = {"source": {"display_name": "Canadian Family Physician", "type": "journal"}}
    paper = to_paper(make_work(type="review", primary_location=pmc, locations=[pmc, journal]))

    assert paper.venue == "Canadian Family Physician"
    assert paper.work_type == "review"


def test_work_found_only_in_repositories_keeps_its_type():
    pmc = {"source": {"display_name": "PubMed Central", "type": "repository"}}
    paper = to_paper(make_work(type="article", primary_location=pmc, locations=[pmc]))

    assert paper.venue is None
    assert paper.work_type == "article"  # 没有正式出处，但也不是预印本
