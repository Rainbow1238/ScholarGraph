"""两个基线系统的测试。"""

from __future__ import annotations

from fakes import FakeLLM, FakeSearcher, make_paper

from scholargraph.config import Settings
from scholargraph.evaluation.baselines import (
    MAX_AGENT_PAPERS,
    MAX_AGENT_SEARCHES,
    NextStep,
    run_direct,
    run_single_agent,
)
from scholargraph.evaluation.dataset import EvalQuestion

QUESTION = EvalQuestion(id="q1", question="RAG 有哪些改进方向？", aspects=["检索"])
SETTINGS = Settings(api_key="test-key")
PAPER_A = make_paper("2401.00001", "Paper A")
PAPER_B = make_paper("2402.00002", "Paper B")


def search(query: str) -> NextStep:
    return NextStep(thought="还需要检索", done=False, query=query)


DONE = NextStep(thought="已经足够", done=True)


def make_llm(steps: list[NextStep], report: str = "综述正文。") -> FakeLLM:
    return FakeLLM(structured={NextStep: steps}, texts=[report])


# ───────────── 直接回答 ─────────────


def test_direct_baseline_answers_without_searching():
    llm = FakeLLM(structured={}, texts=["RAG 能减少幻觉。[arxiv:2005.11401v4]"])
    searcher = FakeSearcher([PAPER_A])

    output = run_direct(QUESTION, llm, searcher, SETTINGS)

    assert output.report == "RAG 能减少幻觉 [arXiv:2005.11401]。"  # 引用写法被规范化
    assert output.evidence == {}  # 没有检索结果：引用是否存在留给评测阶段核实
    assert searcher.queries == []
    assert QUESTION.question in llm.prompts["text"][0]


# ───────────── 单智能体 ─────────────


def test_single_agent_searches_until_it_decides_to_stop():
    llm = make_llm([search("rag retrieval"), search("rag generation"), DONE])
    searcher = FakeSearcher([], by_query={"rag retrieval": [PAPER_A], "rag generation": [PAPER_B]})

    output = run_single_agent(QUESTION, llm, searcher, SETTINGS)

    assert searcher.queries == ["rag retrieval", "rag generation"]
    assert set(output.evidence) == {PAPER_A.paper_id, PAPER_B.paper_id}
    assert output.details["searches"] == ["rag retrieval", "rag generation"]
    assert "Paper A" in llm.prompts["NextStep"][1]  # 下一步的决定能看到之前的检索结果
    writer_prompt = llm.prompts["text"][0]
    assert "Abstract of Paper A." in writer_prompt and "Abstract of Paper B." in writer_prompt


def test_single_agent_stops_when_it_repeats_a_query():
    llm = make_llm([search("rag retrieval"), search("RAG  Retrieval")])  # 第二次只是换了大小写
    searcher = FakeSearcher([PAPER_A])

    run_single_agent(QUESTION, llm, searcher, SETTINGS)

    assert searcher.queries == ["rag retrieval"]


def test_single_agent_cannot_exceed_its_search_budget():
    steps = [search(f"query number {n}") for n in range(MAX_AGENT_SEARCHES + 5)]
    searcher = FakeSearcher([PAPER_A])

    run_single_agent(QUESTION, make_llm(steps), searcher, SETTINGS)

    assert len(searcher.queries) == MAX_AGENT_SEARCHES


def test_single_agent_reads_a_bounded_number_of_papers():
    many = [make_paper(f"2401.{n:05d}", f"Paper {n}") for n in range(MAX_AGENT_PAPERS + 10)]
    llm = make_llm([search("rag"), DONE])
    settings = Settings(api_key="test-key", papers_per_query=len(many))

    output = run_single_agent(QUESTION, llm, FakeSearcher(many), settings)

    assert len(output.evidence) == MAX_AGENT_PAPERS
    assert llm.prompts["text"][0].count("ID: arXiv:") == MAX_AGENT_PAPERS


def test_single_agent_still_writes_when_it_never_searches():
    output = run_single_agent(
        QUESTION, make_llm([DONE], report="没有检索。"), FakeSearcher([]), SETTINGS
    )

    assert output.report == "没有检索。"
    assert output.evidence == {}
