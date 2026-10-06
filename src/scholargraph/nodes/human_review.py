"""人工审批：在花钱检索之前，让人确认调研计划。

`interrupt()` 会让整张图暂停，并把当前状态存入 checkpoint；
外部用 `Command(resume=审批结果)` 恢复时，这个节点从头重新执行，
而 `interrupt()` 这一次直接返回审批结果。
因此 `interrupt()` 之前不能有带副作用的代码。
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.types import interrupt

from scholargraph.state import ResearchState


class ReviewDecision(TypedDict):
    """审批结果：恢复执行时传回给图的数据。"""

    approved: bool
    feedback: str


def approve() -> ReviewDecision:
    return {"approved": True, "feedback": ""}


def request_changes(feedback: str) -> ReviewDecision:
    return {"approved": False, "feedback": feedback}


def review_plan(state: ResearchState) -> dict:
    """暂停并等待审批；通过则把计划排入检索队列，否则带着意见退回 Planner。"""
    decision: ReviewDecision = interrupt(
        {
            "question": state["question"], # type: ignore
            "scope": state["scope"], # pyright: ignore[reportTypedDictNotRequiredAccess]
            "plan": [sub_question.model_dump() for sub_question in state["plan"]], # pyright: ignore[reportTypedDictNotRequiredAccess]
        }
    )
    if decision["approved"]:
        return {"plan_feedback": None, "pending": state["plan"]} # type: ignore
    return {"plan_feedback": decision["feedback"]}
