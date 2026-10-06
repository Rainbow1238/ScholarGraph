"""交互式会话的测试：审批提问、自动通过、断点恢复。"""

from __future__ import annotations

import io

import pytest
from fakes import FakeSearcher
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from test_graph import (
    FABRICATED_DRAFT,
    PAPER_A,
    PAPER_B,
    QUESTION,
    SCOPE,
    delete,
    make_llm,
    make_settings,
    verdicts,
)

from scholargraph.graph import build_graph
from scholargraph.session import ResearchSession, UnknownThreadError


class ScriptedConsole(Console):
    """输出写入内存、输入按脚本回答的控制台。"""

    def __init__(self, answers: list[str], width: int = 120) -> None:
        super().__init__(file=io.StringIO(), width=width)
        self._answers = list(answers)

    def input(self, *args, **kwargs) -> str:
        return self._answers.pop(0)

    @property
    def output(self) -> str:
        return self.file.getvalue()


@pytest.fixture
def llm():
    return make_llm()


@pytest.fixture
def graph(llm):
    return build_graph(
        llm=llm,
        searcher=FakeSearcher([PAPER_A, PAPER_B]),
        settings=make_settings(),
        checkpointer=InMemorySaver(),
    )


def test_pressing_enter_approves_the_plan(graph):
    console = ScriptedConsole(answers=[""])
    state = ResearchSession(graph, "t1", console).start(QUESTION)

    assert "## 参考文献" in state["report"]
    assert "调研范围：" in console.output and SCOPE in console.output  # 审批时能看到范围
    assert "调研计划" in console.output and "报告已定稿" in console.output
    assert "阅读 2 篇候选论文，提炼出 2 条发现" in console.output  # 进度里能看到检索统计
    assert "共 2 条带引用的论断，全部通过" in console.output


def test_typed_feedback_triggers_a_new_plan_before_approval(graph, llm):
    console = ScriptedConsole(answers=["请增加评测方向", ""])
    ResearchSession(graph, "t1", console).start(QUESTION)

    assert len(llm.prompts["ResearchPlan"]) == 2


def test_auto_approve_never_asks(graph):
    console = ScriptedConsole(answers=[])  # 一旦提问就会因为没有答案而报错
    state = ResearchSession(graph, "t1", console, auto_approve=True).start(QUESTION)

    assert state["report"]


def test_resume_continues_a_session_paused_at_review(graph):
    graph.invoke({"question": QUESTION}, {"configurable": {"thread_id": "t1"}})  # 停在审批处

    console = ScriptedConsole(answers=[""])
    state = ResearchSession(graph, "t1", console).resume()

    assert "## 参考文献" in state["report"]


def test_resume_of_unknown_session_fails_clearly(graph):
    with pytest.raises(UnknownThreadError):
        ResearchSession(graph, "no-such-thread", ScriptedConsole(answers=[])).resume()


def test_plan_table_wraps_instead_of_truncating_in_a_narrow_window(graph):
    console = ScriptedConsole(answers=[""], width=50)
    ResearchSession(graph, "t1", console).start(QUESTION)

    assert "…" not in console.output
    shown = console.output.replace("\n", "").replace(" ", "").replace("│", "")
    assert "denseretrievalreranking" in shown  # 每一组检索词都完整显示


def test_progress_reports_what_the_verifier_found_and_the_revision():
    llm = make_llm(
        drafts=[FABRICATED_DRAFT], verdict_replies=[verdicts(True)], corrections=[[delete(0)]]
    )
    graph = build_graph(
        llm=llm,
        searcher=FakeSearcher([PAPER_A, PAPER_B]),
        settings=make_settings(),
        checkpointer=InMemorySaver(),
    )
    console = ScriptedConsole(answers=[""])
    ResearchSession(graph, "t1", console).start(QUESTION)

    assert "虚构引用 1 条，交给 Reviser 修订" in console.output
    assert "Reviser 完成第 1 次修订" in console.output
