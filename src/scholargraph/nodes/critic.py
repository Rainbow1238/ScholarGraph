"""Critic：检查检索结果的覆盖情况，决定是否再检索一轮。"""

from __future__ import annotations

from scholargraph.config import Settings
from scholargraph.llm import LLM
from scholargraph.prompts import CRITIC_SYSTEM, critic_prompt
from scholargraph.queries import tidy_sub_questions, tried_query_keys
from scholargraph.schemas import Critique
from scholargraph.state import ResearchState


def critique_coverage(state: ResearchState, *, llm: LLM, settings: Settings) -> dict:
    """评估覆盖缺口。

    是否回到检索由三个条件共同决定：Critic 认为不足、轮数预算还没用完、
    并且补充的子问题里确实有没用过的检索词。
    后两个条件用代码强制执行，所以无论模型怎么回答，回环都一定会结束，
    也不会把已经失败过的检索原样再做一遍。
    """
    notes = state.get("notes", [])
    rounds_done = state.get("research_rounds", 0) + 1
    prompt = critic_prompt(state["question"], state["scope"], notes, settings.max_sub_questions) # type: ignore
    critique = llm.complete_structured(CRITIC_SYSTEM, prompt, Critique)

    follow_ups = []
    has_budget = rounds_done < settings.max_research_rounds
    if has_budget and not critique.is_sufficient:
        follow_ups = tidy_sub_questions(
            critique.follow_ups,
            max_sub_questions=settings.max_sub_questions,
            max_queries=settings.max_queries_per_sub_question,
            already_tried=tried_query_keys(notes),
        )

    return {
        "research_rounds": rounds_done,
        "pending": follow_ups,
        "open_gaps": [] if critique.is_sufficient else critique.gaps,
    }
