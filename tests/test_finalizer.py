"""定稿（来源构成统计）的测试。"""

from __future__ import annotations

from fakes import make_paper

from scholargraph.nodes.finalizer import render_source_composition


def test_composition_counts_publication_status_venues_and_interfaces():
    papers = [
        make_paper("1", "Paper one", sources=["arXiv"]),
        make_paper("2", "Paper two", sources=["arXiv", "OpenAlex"], venue="ACL"),
        make_paper("OpenAlex:W3", "Paper three", sources=["OpenAlex"], venue="Nature"),
        make_paper("OpenAlex:W4", "Paper four", sources=["OpenAlex"], work_type="dissertation"),
    ]
    text = render_source_composition(papers)

    assert "共引用 4 篇论文" in text
    assert "已在期刊或会议正式发表 2 篇，预印本 1 篇，学位论文 1 篇" in text
    assert "来自 2 种期刊或会议" in text
    assert "弱证据（发表满一年仍被引 0 次、学位论文或自存档平台）：1 篇" in text
    assert "arXiv 2 篇" in text and "OpenAlex 3 篇" in text  # 被两个接口找到的论文两边都计数
    assert "注意" not in text


def test_a_single_search_interface_is_not_reported_as_concentration():
    """OpenAlex 是索引库：35 篇都经它检索到，不代表来自同一个出版方。"""
    venues = ["Nutrients", "Obesity", "NEJM", "Cell", "JAMA", "BMJ"]
    papers = [
        make_paper(f"OpenAlex:W{n}", f"Paper {n}", sources=["OpenAlex"], venue=venue)
        for n, venue in enumerate(venues)
    ]
    text = render_source_composition(papers)

    assert "OpenAlex 6 篇" in text
    assert "注意" not in text


def test_warning_when_one_venue_dominates():
    venues = ["Nutrients"] * 4 + ["Obesity", "NEJM"]
    papers = [
        make_paper(f"OpenAlex:W{n}", f"Paper {n}", sources=["OpenAlex"], venue=venue)
        for n, venue in enumerate(venues)
    ]
    text = render_source_composition(papers)

    assert "最多的是 Nutrients 4 篇" in text
    assert "注意：4 篇（67%）来自 Nutrients，出处较为集中" in text


def test_warning_when_most_papers_are_not_formally_published():
    papers = [make_paper("1", "Paper one"), make_paper("2", "Paper two")]
    text = render_source_composition(papers)

    assert "一半以上的论文（2 篇）没有正式发表的出处" in text


def test_no_papers_is_stated_plainly():
    assert render_source_composition([]) == "正文没有引用任何论文。"
