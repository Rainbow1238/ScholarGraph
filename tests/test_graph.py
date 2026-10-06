"""整张图的端到端测试：用假模型和假检索器离线运行，验证流程控制是否正确。"""

from __future__ import annotations

import pytest
from fakes import FakeLLM, FakeSearcher, make_paper
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from scholargraph.checkpoint import open_checkpointer
from scholargraph.config import Settings
from scholargraph.graph import build_graph
from scholargraph.llm import LLMOutputError
from scholargraph.nodes.human_review import approve, request_changes
from scholargraph.schemas import (
    Correction,
    CorrectionList,
    Critique,
    Finding,
    FindingList,
    ResearchPlan,
    SubQuestion,
    SupportVerdict,
    SupportVerdicts,
)

QUESTION = "RAG 有哪些主要改进方向？"
SCOPE = "面向纯文本大语言模型的 RAG；不包括多模态检索"
PAPER_A = make_paper("2401.00001", "RAG Survey", sources=["arXiv"])
PAPER_B = make_paper(
    "OpenAlex:W2", "Agentic RAG", sources=["OpenAlex"], venue="ACL", citation_count=150
)
CITE_A, CITE_B = f"[{PAPER_A.paper_id}]", f"[{PAPER_B.paper_id}]"

PLAN = ResearchPlan(
    scope=SCOPE,
    sub_questions=[
        SubQuestion(
            question="检索阶段有哪些改进？",
            search_queries=["RAG retrieval improvement", "dense retrieval reranking"],
        ),
        SubQuestion(
            question="生成阶段有哪些改进？",
            search_queries=["RAG generation improvement", "retrieval augmented generation"],
        ),
    ],
)
PLANNED_QUERIES = [query for q in PLAN.sub_questions for query in q.search_queries]
FINDINGS = FindingList(
    findings=[
        Finding(statement="RAG 能减少幻觉", paper_ids=[PAPER_A.paper_id]),
        Finding(statement="Agentic RAG 引入了规划", paper_ids=[PAPER_B.paper_id]),
    ]
)
SUFFICIENT = Critique(expected_aspects=["检索", "生成"], is_sufficient=True, gaps=[], follow_ups=[])
INSUFFICIENT = Critique(
    expected_aspects=["检索", "生成", "评测"],
    is_sufficient=False,
    gaps=["缺少评测方法"],
    follow_ups=[
        SubQuestion(question="RAG 如何评测？", search_queries=["RAG evaluation benchmark"])
    ],
)

SENTENCE_A = f"RAG 能减少幻觉 {CITE_A}。"
SENTENCE_B = f"Agentic RAG 引入了规划 {CITE_B}。"
SENTENCE_FABRICATED = "RAG 已被彻底解决 [arXiv:9999.99999]。"
GOOD_DRAFT = f"# 综述\n\n{SENTENCE_A}{SENTENCE_B}"
FABRICATED_DRAFT = f"# 综述\n\n{SENTENCE_A}{SENTENCE_FABRICATED}"


def verdicts(*supported_flags: bool) -> SupportVerdicts:
    """按顺序为每条论断给出判断：True 表示摘要支撑。"""
    return SupportVerdicts(
        verdicts=[
            SupportVerdict(
                claim_index=i, supported=flag, reason="摘要支撑" if flag else "摘要未提及"
            )
            for i, flag in enumerate(supported_flags)
        ]
    )


def rewrite(issue_index: int, new_text: str) -> Correction:
    return Correction(issue_index=issue_index, action="rewrite", new_text=new_text)


def delete(issue_index: int) -> Correction:
    return Correction(issue_index=issue_index, action="delete", new_text="")


def make_settings(**overrides) -> Settings:
    return Settings(api_key="test-key", **overrides)


ALL_SUPPORTED = verdicts(True, True)


def make_llm(
    *,
    critiques=(SUFFICIENT,),
    drafts=(GOOD_DRAFT,),
    verdict_replies=(ALL_SUPPORTED,),
    corrections=((),),
) -> FakeLLM:
    return FakeLLM(
        structured={
            ResearchPlan: [PLAN],
            FindingList: [FINDINGS],
            Critique: list(critiques),
            SupportVerdicts: list(verdict_replies),
            CorrectionList: [CorrectionList(corrections=list(batch)) for batch in corrections],
        },
        texts=list(drafts),
    )


def make_graph(llm, searcher, *, settings=None, checkpointer=None):
    return build_graph(
        llm=llm,
        searcher=searcher,
        settings=settings or make_settings(),
        checkpointer=checkpointer or InMemorySaver(),
    )


@pytest.fixture
def searcher() -> FakeSearcher:
    return FakeSearcher([PAPER_A, PAPER_B])


@pytest.fixture
def config() -> dict:
    return {"configurable": {"thread_id": "test-thread"}}


def start(graph, config) -> None:
    """启动一次调研，运行到人工审批处暂停。"""
    graph.invoke({"question": QUESTION}, config)


def resume(graph, config, decision) -> dict:
    """提交审批结果并继续运行，返回此时的状态。"""
    return graph.invoke(Command(resume=decision), config)


def run(graph, config) -> dict:
    """启动、通过审批并运行到结束。"""
    start(graph, config)
    return resume(graph, config, approve())


# ───────────── 规划与人工审批 ─────────────


def test_graph_pauses_for_human_review_before_any_search(searcher, config):
    graph = make_graph(make_llm(), searcher)

    start(graph, config)

    snapshot = graph.get_state(config)
    request = snapshot.interrupts[0].value
    assert snapshot.next == ("human_review",)
    assert request["scope"] == SCOPE
    assert request["plan"] == [q.model_dump() for q in PLAN.sub_questions]
    assert searcher.queries == []


def test_rejected_plan_returns_to_planner_with_the_feedback(searcher, config):
    llm = make_llm()
    graph = make_graph(llm, searcher)

    start(graph, config)
    resume(graph, config, request_changes("请增加评测方向"))

    assert graph.get_state(config).next == ("human_review",)  # 重新规划后再次等待审批
    assert len(llm.prompts["ResearchPlan"]) == 2
    assert "请增加评测方向" in llm.prompts["ResearchPlan"][1]
    assert SCOPE in llm.prompts["ResearchPlan"][1]  # 上一版的范围也交给 Planner 参考
    assert searcher.queries == []


def test_plan_without_any_usable_query_is_rejected(searcher, config):
    blank_plan = ResearchPlan(
        scope=SCOPE, sub_questions=[SubQuestion(question="子问题", search_queries=["   "])]
    )
    graph = make_graph(FakeLLM(structured={ResearchPlan: [blank_plan]}, texts=[]), searcher)

    with pytest.raises(LLMOutputError):
        start(graph, config)


# ───────────── 检索与 Critic 回环 ─────────────


def test_approved_plan_runs_to_a_final_report(searcher, config):
    state = run(make_graph(make_llm(), searcher), config)

    assert sorted(searcher.queries) == sorted(PLANNED_QUERIES)  # 每组检索词都被用到
    assert len(state["notes"]) == 2  # 两个并行分支的笔记都被 reducer 合并进来
    assert set(state["evidence"]) == {PAPER_A.paper_id, PAPER_B.paper_id}
    assert state["issues"] == []
    assert "## 参考文献" in state["report"]
    assert "RAG Survey" in state["report"] and "Agentic RAG" in state["report"]


def test_scope_reaches_every_agent_that_selects_or_writes_content(searcher, config):
    llm = make_llm()
    run(make_graph(llm, searcher), config)

    for agent in ("FindingList", "Critique", "text"):  # Researcher、Critic、Writer
        assert all(SCOPE in prompt for prompt in llm.prompts[agent]), agent


def test_research_loop_stops_when_round_budget_is_used_up(searcher, config):
    llm = make_llm(critiques=[INSUFFICIENT])  # Critic 永远不满意
    state = run(make_graph(llm, searcher, settings=make_settings(max_research_rounds=2)), config)

    assert state["research_rounds"] == 2
    assert searcher.queries.count("RAG evaluation benchmark") == 1  # 补充检索只做了一轮
    assert len(searcher.queries) == len(PLANNED_QUERIES) + 1
    assert state["open_gaps"] == ["缺少评测方法"]
    assert "缺少评测方法" in llm.prompts["text"][0]  # 未补上的缺口被告知 Writer
    assert state["report"]


def test_follow_up_that_repeats_tried_queries_ends_the_loop_early(searcher, config):
    llm = make_llm(critiques=[INSUFFICIENT])  # 每一轮都提出同一个补充子问题、同一组检索词
    state = run(make_graph(llm, searcher, settings=make_settings(max_research_rounds=5)), config)

    # 第 2 轮之后，补充子问题的检索词都已经用过，被代码过滤掉，于是直接进入写作
    assert state["research_rounds"] == 2
    assert searcher.queries.count("RAG evaluation benchmark") == 1


def test_critic_is_told_which_queries_were_used_and_what_they_returned(searcher, config):
    llm = make_llm()
    run(make_graph(llm, searcher), config)

    research_log = llm.prompts["Critique"][0]
    assert all(query in research_log for query in PLANNED_QUERIES)
    assert "阅读了 2 篇候选论文" in research_log


def test_findings_per_sub_question_are_capped(searcher, config):
    many = FindingList(
        findings=[Finding(statement=f"发现 {n}", paper_ids=[PAPER_A.paper_id]) for n in range(10)]
    )
    llm = make_llm()
    llm._structured[FindingList] = [many]
    settings = make_settings(max_findings_per_sub_question=3)

    state = run(make_graph(llm, searcher, settings=settings), config)

    assert all(len(note.findings) == 3 for note in state["notes"])
    assert [f.statement for f in state["notes"][0].findings] == ["发现 0", "发现 1", "发现 2"]


# ───────────── 引用校验与逐句修订 ─────────────


def test_fabricated_citation_is_rewritten_and_only_the_new_sentence_is_rechecked(searcher, config):
    corrected = f"RAG 仍有待解决的问题 {CITE_B}。"
    llm = make_llm(
        drafts=[FABRICATED_DRAFT],
        verdict_replies=[verdicts(True), verdicts(True)],
        corrections=[[rewrite(0, corrected)]],
    )
    state = run(make_graph(llm, searcher), config)

    assert state["revisions"] == 1
    assert "9999.99999" in llm.prompts["CorrectionList"][0]  # Reviser 被告知哪一句有问题
    assert "9999.99999" not in state["report"]
    assert SENTENCE_A in state["report"] and corrected in state["report"]  # 没问题的句子一字未改
    assert state["issues"] == []

    first_check, second_check = llm.prompts["SupportVerdicts"]
    assert "RAG 能减少幻觉" in first_check  # 虚构引用由代码检出，不需要问模型
    assert "RAG 已被彻底解决" not in first_check
    assert "RAG 仍有待解决的问题" in second_check
    assert "RAG 能减少幻觉" not in second_check  # 上一轮已通过的句子不重复校验


def test_sentence_is_deleted_when_the_reviser_cannot_rewrite_it(searcher, config):
    llm = make_llm(
        drafts=[FABRICATED_DRAFT], verdict_replies=[verdicts(True)], corrections=[[delete(0)]]
    )
    state = run(make_graph(llm, searcher), config)

    assert "RAG 已被彻底解决" not in state["report"]
    assert SENTENCE_A in state["report"]
    assert len(llm.prompts["SupportVerdicts"]) == 1  # 删除后没有新句子，不需要再问模型


def test_flagged_sentence_is_deleted_when_the_reviser_gives_no_correction(searcher, config):
    llm = make_llm(drafts=[FABRICATED_DRAFT], verdict_replies=[verdicts(True)], corrections=[[]])
    state = run(make_graph(llm, searcher), config)

    assert "RAG 已被彻底解决" not in state["report"]
    assert state["issues"] == []


def test_unsupported_claim_is_sent_for_revision_with_the_reason(searcher, config):
    llm = make_llm(
        verdict_replies=[verdicts(True, False), verdicts(True)],
        corrections=[[rewrite(0, f"Agentic RAG 被用于多步检索 {CITE_B}。")]],
    )
    state = run(make_graph(llm, searcher), config)

    assert "摘要未提及" in llm.prompts["CorrectionList"][0]
    assert "Abstract of Agentic RAG." in llm.prompts["CorrectionList"][0]  # 附上了相关摘要
    assert "Agentic RAG 被用于多步检索" in state["report"]
    assert [entry.unsupported for entry in state["verification_log"]] == [1, 0]


def test_remaining_issues_are_disclosed_when_revision_budget_is_used_up(searcher, config):
    llm = make_llm(verdict_replies=[verdicts(True, False)])
    state = run(make_graph(llm, searcher, settings=make_settings(max_revisions=0)), config)

    assert state.get("revisions", 0) == 0
    assert [issue.kind for issue in state["issues"]] == ["unsupported"]
    assert "未通过自动校验" in state["report"]
    assert "摘要未提及" in state["report"]


def test_claim_the_model_never_judges_is_reported_as_unverified_not_passed(searcher, config):
    only_first = SupportVerdicts(
        verdicts=[SupportVerdict(claim_index=0, supported=True, reason="ok")]
    )
    nothing = SupportVerdicts(verdicts=[])
    llm = make_llm(verdict_replies=[only_first, nothing])  # 第二条论断两次都没有得到判断
    state = run(make_graph(llm, searcher), config)

    assert state["unverified_claims"] == [SENTENCE_B]
    assert state["verification_log"][0].unverified == 1
    assert len(llm.prompts["SupportVerdicts"]) == 2  # 漏判的论断被再问了一次
    assert "未能完成论断与摘要的一致性校验" in state["report"]
    assert "均已通过自动校验" not in state["report"]


def test_claims_are_verified_in_batches(searcher, config):
    llm = make_llm(verdict_replies=[verdicts(True)])
    state = run(make_graph(llm, searcher, settings=make_settings(verify_batch_size=1)), config)

    assert len(llm.prompts["SupportVerdicts"]) == 2  # 两条论断，每批一条
    assert state["verification_log"][0].checked == 2
    assert state["issues"] == [] and state["unverified_claims"] == []


def test_citations_in_the_report_are_written_in_canonical_form(searcher, config):
    draft_with_variants = "# 综述\n\nRAG 能减少幻觉。[arxiv:2401.00001v3]"
    llm = make_llm(drafts=[draft_with_variants], verdict_replies=[verdicts(True)])
    state = run(make_graph(llm, searcher), config)

    assert f"RAG 能减少幻觉 {CITE_A}。" in state["report"]
    assert "v3" not in state["report"]
    assert state["issues"] == []  # 写法不规范的引用不会被误判为虚构引用


# ───────────── 定稿 ─────────────


def test_report_ends_with_references_source_composition_and_verification_note(searcher, config):
    report = run(make_graph(make_llm(), searcher), config)["report"]

    assert (
        report.index("## 参考文献")
        < report.index("## 文献来源构成")
        < report.index("## 引用校验说明")
    )
    assert "arXiv 1 篇" in report and "OpenAlex 1 篇" in report
    assert "已在期刊或会议正式发表 1 篇，预印本 1 篇" in report
    assert "*Agentic RAG*. ACL, 2024. 被引 150 次." in report


def test_report_without_citations_does_not_claim_to_be_verified(config):
    llm = FakeLLM(
        structured={
            ResearchPlan: [PLAN],
            FindingList: [FindingList(findings=[])],
            Critique: [SUFFICIENT],
        },
        texts=["# 综述\n\n没有检索到相关文献。"],
    )
    state = run(make_graph(llm, FakeSearcher([])), config)  # 检索器什么都找不到

    assert "没有可校验的论断" in state["report"]
    assert "均已通过自动校验" not in state["report"]
    assert "FindingList" not in llm.prompts  # 没有候选论文时不调用模型


# ───────────── 断点续跑 ─────────────


def test_run_can_resume_after_process_restart(searcher, config, tmp_path):
    database = tmp_path / "checkpoints.sqlite"

    with open_checkpointer(database) as checkpointer:
        start(make_graph(make_llm(), searcher, checkpointer=checkpointer), config)
    # 离开 with 块后连接已关闭，相当于进程退出

    with open_checkpointer(database) as checkpointer:  # 相当于重新启动：全新的图对象和数据库连接
        graph = make_graph(make_llm(), searcher, checkpointer=checkpointer)
        state = resume(graph, config, approve())

    assert "## 参考文献" in state["report"]
    assert isinstance(state["plan"][0], SubQuestion)  # 自定义类型从数据库里被完整还原
    assert state["verification_log"][0].checked == 2


def test_evidence_reducer_keeps_a_main_conclusion_given_by_an_earlier_researcher():
    from scholargraph.state import merge_evidence

    with_conclusion = PAPER_A.model_copy(update={"main_conclusion": "主要结论"})
    merged = merge_evidence({PAPER_A.paper_id: with_conclusion}, {PAPER_A.paper_id: PAPER_A})

    assert merged[PAPER_A.paper_id].main_conclusion == "主要结论"
