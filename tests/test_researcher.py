"""Researcher 节点的测试：多路检索的融合、候选与发现的上限、出处校验、失败容忍。"""

from __future__ import annotations

import pytest
from fakes import FakeLLM, FakeSearcher, make_paper

from scholargraph.config import Settings
from scholargraph.nodes import research_sub_question
from scholargraph.schemas import Finding, FindingList, PaperConclusion, SubQuestion
from scholargraph.search import SearchError

PAPER_A = make_paper("2401.00001", "Paper A title long enough")
PAPER_B = make_paper("2402.00002", "Paper B title long enough")
PAPER_C = make_paper("2403.00003", "Paper C title long enough")
SCOPE = "只研究纯文本模型"


def run_researcher(
    searcher: FakeSearcher,
    findings: list[Finding],
    queries=("q one", "q two"),
    **settings_overrides,
):
    llm = FakeLLM(structured={FindingList: [FindingList(findings=findings)]}, texts=[])
    task = {
        "question": "研究问题",
        "scope": SCOPE,
        "sub_question": SubQuestion(question="子问题", search_queries=list(queries)),
    }
    settings = Settings(api_key="test-key", **settings_overrides)
    update = research_sub_question(task, llm=llm, searcher=searcher, settings=settings)
    return update, llm


def finding(paper_id: str, statement: str = "发现") -> Finding:
    return Finding(statement=statement, paper_ids=[paper_id])


# ───────────── 检索与融合 ─────────────


def test_candidates_from_all_queries_are_merged_without_duplicates():
    searcher = FakeSearcher([], by_query={"q one": [PAPER_A, PAPER_B], "q two": [PAPER_B, PAPER_C]})
    update, llm = run_researcher(searcher, findings=[])

    note = update["notes"][0]
    assert searcher.queries == ["q one", "q two"]
    assert note.retrieved_count == 3
    assert note.search_queries == ["q one", "q two"]
    prompt = llm.prompts["FindingList"][0]
    assert prompt.count("ID: arXiv:2402.00002") == 1  # 重复检索到的论文只给模型看一次


def test_paper_returned_by_several_queries_is_shown_to_the_model_first():
    searcher = FakeSearcher([], by_query={"q one": [PAPER_A, PAPER_B], "q two": [PAPER_B, PAPER_C]})
    _, llm = run_researcher(searcher, findings=[])

    prompt = llm.prompts["FindingList"][0]
    assert prompt.index("arXiv:2402.00002") < prompt.index("arXiv:2401.00001")


def test_candidates_shown_to_the_model_are_capped():
    many = [make_paper(f"2401.{n:05d}", f"Distinct paper title number {n}") for n in range(30)]
    update, llm = run_researcher(
        FakeSearcher(many),
        findings=[],
        queries=("q",),
        papers_per_query=30,
        max_candidates_per_sub_question=5,
    )

    assert update["notes"][0].retrieved_count == 5
    assert llm.prompts["FindingList"][0].count("ID: arXiv:") == 5


def test_prompt_tells_the_model_the_scope_venue_and_citation_count():
    cited = make_paper("OpenAlex:W9", "A well cited paper", venue="ICLR", citation_count=1234)
    _, llm = run_researcher(FakeSearcher([cited]), findings=[])

    prompt = llm.prompts["FindingList"][0]
    assert SCOPE in prompt
    assert "ID: OpenAlex:W9 | A well cited paper | ICLR，2024，被引 1234 次" in prompt


# ───────────── 发现的校验与上限 ─────────────


def test_only_cited_papers_enter_the_evidence_pool():
    update, _ = run_researcher(FakeSearcher([PAPER_A, PAPER_B]), [finding("arXiv:2401.00001")])

    assert set(update["evidence"]) == {"arXiv:2401.00001"}


@pytest.mark.parametrize("written_id", ["arxiv:2401.00001", "2401.00001v2", "[arXiv:2401.00001]"])
def test_paper_id_variants_written_by_the_model_are_recognised(written_id):
    update, _ = run_researcher(FakeSearcher([PAPER_A]), [finding(written_id)])

    note = update["notes"][0]
    assert note.findings[0].paper_ids == ["arXiv:2401.00001"]
    assert note.discarded_findings == 0


def test_findings_without_a_valid_source_are_discarded_and_counted():
    update, _ = run_researcher(
        FakeSearcher([PAPER_A]),
        [
            finding("arXiv:2401.00001", "有出处"),
            finding("arXiv:9999.99999", "出处不在候选论文里"),
            Finding(statement="没有出处", paper_ids=[]),
        ],
    )

    note = update["notes"][0]
    assert [f.statement for f in note.findings] == ["有出处"]
    assert note.discarded_findings == 2


def test_findings_are_capped_keeping_the_most_important_ones_first():
    proposed = [finding("arXiv:2401.00001", f"发现 {n}") for n in range(6)]
    update, _ = run_researcher(FakeSearcher([PAPER_A]), proposed, max_findings_per_sub_question=2)

    note = update["notes"][0]
    assert [f.statement for f in note.findings] == ["发现 0", "发现 1"]
    assert note.discarded_findings == 0  # 超出上限被截掉的不算"出处无效"


def test_llm_is_not_called_when_nothing_was_retrieved():
    update, llm = run_researcher(FakeSearcher([]), findings=[])

    note = update["notes"][0]
    assert (note.retrieved_count, note.findings) == (0, [])
    assert update["evidence"] == {}
    assert "FindingList" not in llm.prompts


# ───────────── 检索失败 ─────────────


def test_a_failing_query_does_not_discard_results_of_the_others():
    searcher = FakeSearcher([PAPER_A], failing=("q one",))
    update, _ = run_researcher(searcher, [finding("arXiv:2401.00001")])

    assert update["notes"][0].retrieved_count == 1


def test_search_error_is_raised_when_every_query_fails():
    searcher = FakeSearcher([PAPER_A], failing=("q one", "q two"))
    with pytest.raises(SearchError):
        run_researcher(searcher, findings=[])


# ───────────── 主要结论与证据强度 ─────────────


def test_main_conclusions_are_attached_to_the_papers_in_the_evidence_pool():
    reply = FindingList(
        paper_conclusions=[
            PaperConclusion(paper_id="arxiv:2401.00001v2", conclusion="A 并不比 B 更有效"),
            PaperConclusion(paper_id="arXiv:9999.99999", conclusion="不在候选里的论文被忽略"),
        ],
        findings=[finding("arXiv:2401.00001", "A 组不良事件与 B 组相当")],
    )
    llm = FakeLLM(structured={FindingList: [reply]}, texts=[])
    task = {
        "question": "研究问题",
        "scope": SCOPE,
        "sub_question": SubQuestion(question="子问题", search_queries=["q"]),
    }
    update = research_sub_question(
        task, llm=llm, searcher=FakeSearcher([PAPER_A]), settings=Settings(api_key="k")
    )

    assert update["evidence"]["arXiv:2401.00001"].main_conclusion == "A 并不比 B 更有效"


def test_weak_evidence_is_demoted_out_of_the_candidate_list_when_enough_others_exist():
    weak = make_paper("OpenAlex:W1", "Uncited paper title", venue="J", citation_count=0, year=2000)
    strong = [make_paper(f"2401.{n:05d}", f"Distinct paper title number {n}") for n in range(15)]
    _, llm = run_researcher(
        FakeSearcher([weak, *strong]),
        findings=[],
        queries=("q",),
        papers_per_query=20,
        max_candidates_per_sub_question=10,
    )

    assert "OpenAlex:W1" not in llm.prompts["FindingList"][0]


def test_weak_evidence_is_labelled_in_the_prompt():
    weak = make_paper("OpenAlex:W1", "Uncited paper title", venue="J", citation_count=0, year=2000)
    _, llm = run_researcher(FakeSearcher([weak]), findings=[])

    assert "J，2000，被引 0 次，弱证据：被引 0 次" in llm.prompts["FindingList"][0]
