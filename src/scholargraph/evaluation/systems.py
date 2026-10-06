"""被评测的系统配置。每个配置都是同一种函数：给一道题，返回一份输出。"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import replace

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from scholargraph.checkpoint import build_serializer
from scholargraph.config import Settings
from scholargraph.evaluation.baselines import run_direct, run_single_agent
from scholargraph.evaluation.dataset import EvalQuestion
from scholargraph.evaluation.records import SystemOutput
from scholargraph.graph import build_graph
from scholargraph.llm import LLM
from scholargraph.nodes import names
from scholargraph.nodes.human_review import approve
from scholargraph.search import PaperSearcher

System = Callable[[EvalQuestion, LLM, PaperSearcher, Settings], SystemOutput]


def run_full(
    question: EvalQuestion, llm: LLM, searcher: PaperSearcher, settings: Settings
) -> SystemOutput:
    """完整的多智能体工作流。评测时没有人值守，调研计划自动通过。"""
    # 评测不需要断点续跑，状态放在内存里即可；序列化器与正式运行用同一个（登记了自定义类型）
    checkpointer = InMemorySaver(serde=build_serializer())
    graph = build_graph(llm=llm, searcher=searcher, settings=settings, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex}"}}

    first_draft: str | None = None
    graph_input: dict | Command = {"question": question.question}
    while True:
        for update in graph.stream(graph_input, config, stream_mode="updates"):
            if names.WRITER in update:  # Writer 只运行一次，它的输出就是引用校验之前的初稿
                first_draft = update[names.WRITER]["draft"]
        snapshot = graph.get_state(config)
        if not snapshot.next:
            break
        graph_input = Command(resume=approve())  # 停在人工审批处：自动通过

    state = snapshot.values
    was_revised = state.get("revisions", 0) > 0
    return SystemOutput(
        report=state["draft"],
        evidence=state.get("evidence", {}),
        first_draft=first_draft if was_revised else None,
        details={
            "scope": state.get("scope"),
            "research_rounds": state.get("research_rounds", 0),
            "sub_questions": len(state.get("notes", [])),
            "revisions": state.get("revisions", 0),
            "verification_rounds": [
                entry.model_dump(exclude={"issues"}) for entry in state.get("verification_log", [])
            ],
        },
    )


def run_without_critic(
    question: EvalQuestion, llm: LLM, searcher: PaperSearcher, settings: Settings
) -> SystemOutput:
    """消融：去掉 Critic 的补充检索。只检索一轮，其余与完整系统相同。"""
    return run_full(question, llm, searcher, replace(settings, max_research_rounds=1))


# 名称 -> (系统, 在对比表里显示的说明)。顺序就是对比表里的行序：从最简单到最完整。
SYSTEMS: dict[str, tuple[System, str]] = {
    "direct": (run_direct, "直接回答（无检索）"),
    "single_agent": (run_single_agent, "单智能体"),
    "no_critic": (run_without_critic, "完整系统去掉 Critic"),
    "full": (run_full, "完整系统"),
}
