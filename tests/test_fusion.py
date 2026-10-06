"""多路检索结果融合的测试：同一篇论文的识别与合并、RRF 排序。"""

from fakes import make_paper

from scholargraph.search.fusion import fuse, merge_papers, title_key


def ids(papers) -> list[str]:
    return [paper.paper_id for paper in papers]


# ───────────── 排序 ─────────────


def test_paper_ranked_high_in_several_lists_comes_first():
    a, b, c = (
        make_paper("1", "Alpha paper title"),
        make_paper("2", "Beta paper title"),
        make_paper("3", "Gamma paper title"),
    )
    fused = fuse([[a, b], [b, c], [b, a]])

    assert ids(fused) == ["arXiv:2", "arXiv:1", "arXiv:3"]


def test_single_list_keeps_its_order():
    papers = [make_paper(str(n), f"Distinct paper title {n}") for n in range(5)]
    assert ids(fuse([papers])) == ids(papers)


def test_citation_count_breaks_ties():
    quiet = make_paper("1", "Quiet paper title here", citation_count=3)
    famous = make_paper("2", "Famous paper title here", citation_count=900)
    fused = fuse([[quiet], [famous]])  # 两篇都只在某个列表里排第一，得分相同

    assert ids(fused) == ["arXiv:2", "arXiv:1"]


def test_limit_truncates_after_ranking():
    papers = [make_paper(str(n), f"Distinct paper title {n}") for n in range(5)]
    assert len(fuse([papers], limit=2)) == 2


def test_empty_input_gives_empty_output():
    assert fuse([]) == [] and fuse([[], []]) == []


# ───────────── 识别同一篇论文 ─────────────


def test_same_id_from_two_sources_is_merged_into_one_record():
    from_arxiv = make_paper("2401.00001", "RAG Survey", sources=["arXiv"])
    from_openalex = make_paper(
        "2401.00001", "RAG Survey", sources=["OpenAlex"], venue="ACL", citation_count=120
    )
    [merged] = fuse([[from_arxiv], [from_openalex]])

    assert merged.sources == ["arXiv", "OpenAlex"]
    assert (merged.venue, merged.citation_count) == ("ACL", 120)


def test_same_doi_is_recognised_as_the_same_paper():
    preprint = make_paper("2401.00001", "Completely Different Preprint Title", doi="10.1/ABC")
    published = make_paper("OpenAlex:W9", "The Published Title Of The Work", doi="10.1/abc")

    assert len(fuse([[preprint], [published]])) == 1


def test_same_title_is_recognised_despite_case_and_punctuation():
    first = make_paper("2401.00001", "Retrieval-Augmented Generation: A Survey")
    second = make_paper("OpenAlex:W9", "retrieval augmented generation - a survey")
    [merged] = fuse([[first], [second]])

    assert merged.paper_id == "arXiv:2401.00001"  # 合并后保留 arXiv 的 ID


def test_short_titles_are_not_used_to_identify_papers():
    first, second = make_paper("1", "Editorial"), make_paper("OpenAlex:W9", "Editorial")
    assert len(fuse([[first], [second]])) == 2


def test_title_key_ignores_case_punctuation_and_whitespace():
    assert (
        title_key("  DoLa:  Decoding by Contrasting Layers! ") == "doladecodingbycontrastinglayers"
    )


# ───────────── 合并记录 ─────────────


def test_merge_prefers_arxiv_id_whatever_the_order():
    arxiv = make_paper("arXiv:2401.00001", "Shared title of the paper")
    openalex = make_paper("OpenAlex:W9", "Shared title of the paper")

    assert merge_papers(openalex, arxiv).paper_id == "arXiv:2401.00001"
    assert merge_papers(arxiv, openalex).paper_id == "arXiv:2401.00001"


def test_merge_takes_the_more_complete_value_of_each_field():
    first = make_paper(
        "OpenAlex:W9", "T" * 20, abstract="short", year=2024, citation_count=5, sources=["OpenAlex"]
    )
    second = make_paper(
        "S2:7",
        "T" * 20,
        abstract="a much longer abstract",
        year=2023,
        citation_count=9,
        venue="NeurIPS",
        doi="10.1/x",
        sources=["Semantic Scholar"],
    )
    merged = merge_papers(first, second)

    assert merged.paper_id == "OpenAlex:W9"
    assert merged.abstract == "a much longer abstract"
    assert merged.year == 2023  # 最早公开的年份
    assert merged.citation_count == 9
    assert (merged.venue, merged.doi) == ("NeurIPS", "10.1/x")
    assert merged.sources == ["OpenAlex", "Semantic Scholar"]
