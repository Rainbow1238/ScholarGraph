"""两个基线系统：用来回答"多智能体工作流比简单做法好多少"。

- 直接回答：不检索，模型凭自己的知识写综述。
- 单智能体：一个智能体自己决定检索什么、何时停止，然后写综述；
  没有问题拆解、没有并行、没有 Critic、没有引用校验。

两个基线使用与完整系统相同的模型、相同的检索器和相同的引用格式，差别只在工作流。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from scholargraph.citations import canonicalize_citations
from scholargraph.config import Settings
from scholargraph.evaluation.dataset import EvalQuestion
from scholargraph.evaluation.prompts import (
    AGENT_STEP_SYSTEM,
    AGENT_WRITER_SYSTEM,
    DIRECT_SYSTEM,
    agent_step_prompt,
    agent_writer_prompt,
    direct_prompt,
)
from scholargraph.evaluation.records import SystemOutput
from scholargraph.llm import LLM
from scholargraph.queries import query_key
from scholargraph.schemas import Paper
from scholargraph.search import PaperSearcher

MAX_AGENT_SEARCHES = 8
MAX_AGENT_PAPERS = 30  # 交给模型阅读的论文上限，避免提示词过长


class NextStep(BaseModel):
    """单智能体每一步的决定。"""

    thought: str = Field(description="已有的论文覆盖了哪些方面，还缺什么")
    done: bool = Field(description="是否已经可以动笔")
    query: str = Field(default="", description="done 为 false 时，下一次检索用的英文检索词")


def run_direct(
    question: EvalQuestion, llm: LLM, searcher: PaperSearcher, settings: Settings
) -> SystemOutput:
    """基线一：直接回答。引用的论文是否存在，留给评测阶段联网核实。"""
    report = llm.complete(DIRECT_SYSTEM, direct_prompt(question.question))
    return SystemOutput(report=canonicalize_citations(report))


def run_single_agent(
    question: EvalQuestion, llm: LLM, searcher: PaperSearcher, settings: Settings
) -> SystemOutput:
    """基线二：单智能体循环"决定下一步 -> 检索"，直到它认为够了或用完检索次数，然后写作。"""
    search_log: list[tuple[str, list[Paper]]] = []
    evidence: dict[str, Paper] = {}
    tried: set[str] = set()

    for used in range(MAX_AGENT_SEARCHES):
        prompt = agent_step_prompt(question.question, search_log, MAX_AGENT_SEARCHES - used)
        step = llm.complete_structured(AGENT_STEP_SYSTEM, prompt, NextStep)
        key = query_key(step.query)
        if step.done or not key or key in tried:  # 重复的检索词不会带来新论文，视为结束
            break

        tried.add(key)
        papers = searcher.search(step.query, settings.papers_per_query)
        search_log.append((step.query, papers))
        for paper in papers:
            evidence.setdefault(paper.paper_id, paper)

    readable = dict(list(evidence.items())[:MAX_AGENT_PAPERS])
    prompt = agent_writer_prompt(question.question, list(readable.values()))
    report = llm.complete(AGENT_WRITER_SYSTEM, prompt)
    return SystemOutput(
        report=canonicalize_citations(report),
        evidence=readable,
        details={"searches": [query for query, _ in search_log]},
    )
