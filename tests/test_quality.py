"""发表状态与证据强度的测试（纯函数）。"""

from __future__ import annotations

import pytest
from fakes import make_paper

from scholargraph.quality import (
    demote_weak_evidence,
    is_weak_evidence,
    publication_label,
    weakness_reasons,
)


@pytest.mark.parametrize(
    ("overrides", "label"),
    [
        ({"venue": "NEJM"}, "NEJM"),
        ({"work_type": "preprint"}, "预印本"),
        ({"work_type": "dissertation"}, "学位论文"),
        ({"work_type": "article"}, "出处不详"),  # 不是预印本，只是不知道发表在哪里
        ({"work_type": None}, "出处不详"),
    ],
)
def test_publication_label(overrides, label):
    assert publication_label(make_paper("OpenAlex:W1", **overrides)) == label


def test_well_cited_journal_article_is_not_weak():
    paper = make_paper("OpenAlex:W1", venue="NEJM", citation_count=529, year=2022)
    assert weakness_reasons(paper, current_year=2026) == []


def test_uncited_paper_older_than_a_year_is_weak():
    paper = make_paper("OpenAlex:W1", venue="J Chem Health Risks", citation_count=0, year=2025)
    assert weakness_reasons(paper, current_year=2026) == ["被引 0 次"]


def test_uncited_paper_from_this_year_is_not_penalised():
    paper = make_paper("OpenAlex:W1", venue="Some Journal", citation_count=0, year=2026)
    assert not is_weak_evidence(paper, current_year=2026)


def test_unknown_citation_count_is_not_penalised():
    """只被 arXiv 找到的论文没有被引次数，不能因此判为弱证据。"""
    assert not is_weak_evidence(make_paper("2401.00001", citation_count=None, year=2020))


def test_dissertation_and_self_archive_doi_are_weak():
    paper = make_paper(
        "OpenAlex:W1",
        work_type="dissertation",
        doi="10.5281/zenodo.22339471",
        citation_count=3,
    )
    assert weakness_reasons(paper) == ["学位论文", "DOI 来自自存档平台 Zenodo"]


def test_weak_papers_are_moved_down_but_not_dropped():
    strong = [make_paper(f"OpenAlex:W{n}", venue="J", citation_count=10) for n in range(4)]
    weak = make_paper("OpenAlex:W99", venue="J", citation_count=0, year=2000)
    ranked = demote_weak_evidence([weak, *strong], positions=2)

    assert [paper.paper_id for paper in ranked] == [
        "OpenAlex:W0",
        "OpenAlex:W1",
        "OpenAlex:W99",
        "OpenAlex:W2",
        "OpenAlex:W3",
    ]
