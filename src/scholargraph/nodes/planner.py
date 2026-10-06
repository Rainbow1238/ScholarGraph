"""Planner：界定调研范围，并把研究问题拆解成可并行检索的子问题。"""

from __future__ import annotations

from scholargraph.config import Settings
from scholargraph.llm import LLM, LLMOutputError
from scholargraph.prompts import PLANNER_SYSTEM, planner_prompt
from scholargraph.queries import tidy_sub_questions
from scholargraph.schemas import ResearchPlan
from scholargraph.state import ResearchState


def plan_research(state: ResearchState, *, llm: LLM, settings: Settings) -> dict:
    """生成调研范围和计划；如果审批人提了修改意见，则在上一版的基础上重做。"""
    prompt = planner_prompt(
        question=state["question"], # type: ignore
        max_sub_questions=settings.max_sub_questions,
        previous_scope=state.get("scope", ""),
        previous_plan=state.get("plan", []),
        feedback=state.get("plan_feedback"),
    )
    plan = llm.complete_structured(PLANNER_SYSTEM, prompt, ResearchPlan)
    sub_questions = tidy_sub_questions(
        plan.sub_questions,
        max_sub_questions=settings.max_sub_questions,
        max_queries=settings.max_queries_per_sub_question,
    )
    if not sub_questions:
        raise LLMOutputError("Planner 给出的计划里没有任何可用的检索词。")
    return {"scope": plan.scope.strip(), "plan": sub_questions}
