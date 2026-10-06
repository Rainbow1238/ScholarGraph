"""完整系统及其消融配置在评测中的运行方式。"""

from __future__ import annotations

from fakes import FakeSearcher
from test_graph import (
    FABRICATED_DRAFT,
    INSUFFICIENT,
    PAPER_A,
    PAPER_B,
    PLANNED_QUERIES,
    SENTENCE_A,
    delete,
    make_llm,
    make_settings,
    verdicts,
)

from scholargraph.evaluation.dataset import EvalQuestion
from scholargraph.evaluation.systems import SYSTEMS, run_full, run_without_critic

QUESTION = EvalQuestion(id="q1", question="RAG 有哪些主要改进方向？", aspects=["检索"])


def make_searcher() -> FakeSearcher:
    return FakeSearcher([PAPER_A, PAPER_B])


def test_full_system_runs_unattended_and_returns_the_report_body():
    output = run_full(QUESTION, make_llm(), make_searcher(), make_settings())

    assert SENTENCE_A in output.report
    assert "## 参考文献" not in output.report  # 评的是正文，不含程序自动附加的章节
    assert set(output.evidence) == {PAPER_A.paper_id, PAPER_B.paper_id}
    assert output.first_draft is None  # 没有发生修订：初稿就是终稿
    assert output.details["research_rounds"] == 1
    assert output.details["verification_rounds"][0]["checked"] == 2


def test_full_system_keeps_the_draft_from_before_citation_checking():
    llm = make_llm(
        drafts=[FABRICATED_DRAFT], verdict_replies=[verdicts(True)], corrections=[[delete(0)]]
    )
    output = run_full(QUESTION, llm, make_searcher(), make_settings())

    assert "9999.99999" in output.first_draft  # 初稿里有虚构引用
    assert "9999.99999" not in output.report  # 终稿里已被修订掉
    assert output.details["revisions"] == 1


def test_ablation_without_critic_never_does_a_second_round_of_search():
    searcher_full, searcher_ablated = make_searcher(), make_searcher()
    settings = make_settings(max_research_rounds=2)

    run_full(QUESTION, make_llm(critiques=[INSUFFICIENT]), searcher_full, settings)
    output = run_without_critic(
        QUESTION, make_llm(critiques=[INSUFFICIENT]), searcher_ablated, settings
    )

    assert len(searcher_full.queries) == len(PLANNED_QUERIES) + 1  # 完整系统补充检索了一次
    assert len(searcher_ablated.queries) == len(PLANNED_QUERIES)
    assert output.details["research_rounds"] == 1


def test_systems_are_listed_from_simplest_to_most_complete():
    assert list(SYSTEMS) == ["direct", "single_agent", "no_critic", "full"]
