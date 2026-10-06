"""论文 ID 与引用解析（纯函数）的测试。"""

import pytest
from fakes import make_paper

from scholargraph.citations import (
    canonicalize_citations,
    core_claims,
    extract_paper_ids,
    format_references,
    normalize_paper_id,
    split_cited_claims,
)

# ───────────── 论文 ID ─────────────


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("arXiv:2401.12345", "arXiv:2401.12345"),
        ("arxiv: 2401.12345v2", "arXiv:2401.12345"),
        (" [ARXIV:2401.12345v10] ", "arXiv:2401.12345"),
        ("2401.12345", "arXiv:2401.12345"),  # 省略前缀的新式编号
        ("2401.12345v3", "arXiv:2401.12345"),
        ("hep-th/9901001v1", "arXiv:hep-th/9901001"),  # 旧式编号
        ("openalex:w4385245566", "OpenAlex:W4385245566"),
        ("W4385245566", "OpenAlex:W4385245566"),
        ("s2:215416146", "S2:215416146"),
        ("S2:215416146", "S2:215416146"),
        ("arXiv：2401.12345", "arXiv:2401.12345"),  # 全角冒号
    ],
)
def test_normalize_paper_id_unifies_the_ways_an_id_can_be_written(written, expected):
    assert normalize_paper_id(written) == expected


def test_normalize_paper_id_leaves_unrecognised_text_alone():
    assert normalize_paper_id("not an id") == "not an id"


def test_version_suffix_is_only_stripped_from_arxiv_ids():
    assert normalize_paper_id("S2:1234v5") == "S2:1234v5"


# ───────────── 提取引用 ─────────────


def test_extract_paper_ids_keeps_first_seen_order_without_duplicates():
    text = "A [arXiv:2402.00002]。B [arXiv:2401.00001][arXiv:2402.00002]。"
    assert extract_paper_ids(text) == ["arXiv:2402.00002", "arXiv:2401.00001"]


def test_extract_paper_ids_handles_every_source():
    text = "甲 [arXiv:2401.00001]，乙 [OpenAlex:W123456]，丙 [S2:987654]。"
    assert extract_paper_ids(text) == ["arXiv:2401.00001", "OpenAlex:W123456", "S2:987654"]


def test_extract_paper_ids_accepts_several_ids_in_one_bracket():
    text = "见 [arXiv:2401.00001; OpenAlex:W123456, 2402.00002]"
    assert extract_paper_ids(text) == ["arXiv:2401.00001", "OpenAlex:W123456", "arXiv:2402.00002"]


def test_ordinary_square_brackets_are_not_citations():
    assert extract_paper_ids("见表 [1] 和 [注]，以及 [TODO: 补充]。") == []


def test_citation_with_a_version_suffix_refers_to_the_same_paper():
    assert extract_paper_ids("结论 [arXiv:2401.00001v3]。") == ["arXiv:2401.00001"]


# ───────────── 规范化正文里的引用 ─────────────


def test_canonicalize_citations_rewrites_every_variant_to_the_standard_form():
    text = "甲 [arxiv:2401.00001v2]。乙 [arXiv: 2401.00001; openalex:w123456]，丙没有引用。"
    assert canonicalize_citations(text) == (
        "甲 [arXiv:2401.00001]。乙 [arXiv:2401.00001][OpenAlex:W123456]，丙没有引用。"
    )


def test_canonicalize_moves_a_citation_written_after_the_final_punctuation():
    text = "第一段的结论。[arXiv:2401.00001]\nEnglish sentence.[arXiv:2402.00002][S2:1]\n"
    assert canonicalize_citations(text) == (
        "第一段的结论 [arXiv:2401.00001]。\nEnglish sentence [arXiv:2402.00002][S2:1].\n"
    )


def test_canonicalize_converts_fullwidth_brackets_and_colons():
    text = "甲【arXiv：2401.00001】。乙［OpenAlex:W123456］。普通的【注释】不受影响。"
    assert canonicalize_citations(text) == (
        "甲[arXiv:2401.00001]。乙[OpenAlex:W123456]。普通的【注释】不受影响。"
    )


def test_canonicalize_handles_windows_line_endings():
    assert canonicalize_citations("结论。[arXiv:2401.00001]\r\n下一段") == (
        "结论 [arXiv:2401.00001]。\r\n下一段"
    )


def test_canonicalize_does_not_move_a_citation_that_starts_the_next_sentence():
    text = "前一句没有引用。[arXiv:2401.00001] 提出了一种新方法。"
    assert canonicalize_citations(text) == text


# ───────────── 拆分论断 ─────────────


def test_split_cited_claims_keeps_only_sentences_with_citations():
    report = "# 标题 [arXiv:9999.99999]\n\n这句没有引用。检索增强能减少幻觉 [arXiv:2401.00001]。"
    claims = split_cited_claims(report)

    assert len(claims) == 1
    assert claims[0].text == "检索增强能减少幻觉 [arXiv:2401.00001]。"
    assert claims[0].paper_ids == ("arXiv:2401.00001",)


def test_dot_inside_arxiv_id_is_not_a_sentence_boundary():
    claims = split_cited_claims(
        "RAG reduces hallucination [arXiv:2401.00001]. It is cheap [arXiv:2402.00002]."
    )
    assert [claim.paper_ids for claim in claims] == [("arXiv:2401.00001",), ("arXiv:2402.00002",)]


def test_citation_after_the_full_stop_belongs_to_the_previous_sentence():
    claims = split_cited_claims("第一句 [arXiv:2401.00001]。第二句。[arXiv:2402.00002]")
    assert [(claim.text, claim.paper_ids) for claim in claims] == [
        ("第一句 [arXiv:2401.00001]。", ("arXiv:2401.00001",)),
        ("第二句。", ("arXiv:2402.00002",)),
    ]


def test_every_claim_text_is_an_exact_substring_of_the_report():
    """逐句修订靠字符串替换定位句子，所以拆出来的句子必须能在原文里原样找到。"""
    report = "概述。\n\n- 要点一 [arXiv:2401.00001]。  要点二 [OpenAlex:W1]！\n\n结尾 [S2:2]. Next one [arXiv:2402.00002]."
    claims = split_cited_claims(report)

    assert len(claims) == 4
    assert all(claim.text in report for claim in claims)


# ───────────── 参考文献 ─────────────


def test_format_references_lists_only_known_papers_that_are_cited():
    evidence = {
        "arXiv:2401.00001": make_paper("2401.00001", "Cited"),
        "arXiv:2402.00002": make_paper("2402.00002", "Unused"),
    }
    references = format_references("结论 [arXiv:2401.00001][arXiv:0000.00000]。", evidence)

    assert "Cited" in references
    assert "Unused" not in references
    assert "0000.00000" not in references


def test_reference_shows_venue_year_and_citation_count_when_known():
    paper = make_paper(
        "OpenAlex:W123456", "Published Work", venue="ACL", year=2023, citation_count=321
    )
    references = format_references("结论 [OpenAlex:W123456]。", {paper.paper_id: paper})

    assert "*Published Work*. ACL, 2023. 被引 321 次." in references


def test_reference_without_venue_is_labelled_as_a_preprint():
    paper = make_paper("2401.00001", "Preprint Only")
    references = format_references("结论 [arXiv:2401.00001]。", {paper.paper_id: paper})

    assert "*Preprint Only*. 预印本, 2024." in references
    assert "被引" not in references  # 被引次数未知时不显示


def test_reference_with_many_authors_has_a_single_full_stop_after_et_al():
    paper = make_paper(
        "2401.00001", "Many Authors", authors=["A One", "B Two", "C Three", "D Four"]
    )
    references = format_references("结论 [arXiv:2401.00001]。", {paper.paper_id: paper})

    assert "A One, B Two, C Three et al. *Many Authors*" in references
    assert "et al.." not in references


def test_title_ending_with_a_full_stop_does_not_produce_a_double_full_stop():
    paper = make_paper(
        "OpenAlex:W1", "Intermittent fasting and weight loss: Systematic review.", venue="CFP"
    )
    references = format_references("结论 [OpenAlex:W1]。", {paper.paper_id: paper})

    assert "*Intermittent fasting and weight loss: Systematic review*. CFP, 2024." in references


def test_weak_evidence_is_flagged_in_the_reference_list():
    paper = make_paper("OpenAlex:W1", "Uncited", venue="J", citation_count=0, year=2020)
    references = format_references("结论 [OpenAlex:W1]。", {paper.paper_id: paper})

    assert "被引 0 次. 弱证据（被引 0 次）." in references


def test_core_claims_are_overview_sentences_and_generalising_sentences():
    report = (
        "# 标题\n\n"
        "导语里的结论 [arXiv:2401.00001]。\n\n"
        "## 概述\n\n"
        "概述里的结论 [arXiv:2401.00002]。\n\n"
        "## 减重\n\n"
        "一项试验报告体重下降 [arXiv:2401.00003]。总体而言，效果与热量限制相当 [arXiv:2401.00004]。\n"
    )
    texts = [claim.text for claim in core_claims(report)]

    assert texts == [
        "导语里的结论 [arXiv:2401.00001]。",
        "概述里的结论 [arXiv:2401.00002]。",
        "总体而言，效果与热量限制相当 [arXiv:2401.00004]。",
    ]
