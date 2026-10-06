"""组装工作流图。读懂这个文件，就读懂了整个系统的流程。

START
  │
  ▼
planner ◄──────────── 打回并附意见 ──┐
  │                                  │
  ▼                                  │
human_review ────────────────────────┘
  │ 通过（Send 并行分发）
  ▼
researcher × N ◄──── 有缺口且有预算 ─┐
  │                                  │
  ▼                                  │
critic ──────────────────────────────┘
  │ 覆盖充分或不再补充
  ▼
writer
  │
  ▼
verifier ◄───────────────────────────┐
  │         有问题且有预算           │
  ├────────────────────────► reviser ┘
  │ 通过或预算用尽
  ▼
finalizer
  │
  ▼
 END
"""

from __future__ import annotations

from functools import partial

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from scholargraph.config import Settings
from scholargraph.llm import LLM
from scholargraph.nodes import (
    critique_coverage,
    finalize_report,
    names,
    plan_research,
    research_sub_question,
    review_plan,
    revise_report,
    verify_citations,
    write_report,
)
from scholargraph.routing import route_after_critic, route_after_review, route_after_verification
from scholargraph.search import PaperSearcher, SearchError
from scholargraph.state import ResearchState

# 检索依赖外部网络，失败时按指数退避自动重试
SEARCH_RETRY_POLICY = RetryPolicy(max_attempts=3, initial_interval=2.0, retry_on=SearchError)


def build_graph(
    *,
    llm: LLM,
    searcher: PaperSearcher,
    settings: Settings,
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """构建并编译工作流。

    大模型、检索器、配置都从参数传入（依赖注入），节点函数本身不创建任何外部资源，
    所以测试时换成假的实现就能离线跑通整张图。
    人工审批依赖 checkpoint 来暂停和恢复，因此 checkpointer 是必需的。
    """
    builder = StateGraph(ResearchState)

    # 节点：用 partial 把依赖预先绑定好，LangGraph 运行时只需传入状态
    builder.add_node(names.PLANNER, partial(plan_research, llm=llm, settings=settings))
    builder.add_node(names.HUMAN_REVIEW, review_plan)
    builder.add_node(
        names.RESEARCHER,
        partial(research_sub_question, llm=llm, searcher=searcher, settings=settings),
        retry_policy=SEARCH_RETRY_POLICY,
    )
    builder.add_node(names.CRITIC, partial(critique_coverage, llm=llm, settings=settings))
    builder.add_node(names.WRITER, partial(write_report, llm=llm))
    builder.add_node(names.VERIFIER, partial(verify_citations, llm=llm, settings=settings))
    builder.add_node(names.REVISER, partial(revise_report, llm=llm))
    builder.add_node(names.FINALIZER, finalize_report)

    # 边：固定边写死流向；条件边的第三个参数列出所有可能的去向
    builder.add_edge(START, names.PLANNER)
    builder.add_edge(names.PLANNER, names.HUMAN_REVIEW)
    builder.add_conditional_edges(
        names.HUMAN_REVIEW, route_after_review, [names.PLANNER, names.RESEARCHER]
    )
    builder.add_edge(names.RESEARCHER, names.CRITIC)
    builder.add_conditional_edges(
        names.CRITIC, route_after_critic, [names.RESEARCHER, names.WRITER]
    )
    builder.add_edge(names.WRITER, names.VERIFIER)
    builder.add_conditional_edges(
        names.VERIFIER, route_after_verification, [names.REVISER, names.FINALIZER]
    )
    builder.add_edge(names.REVISER, names.VERIFIER)
    builder.add_edge(names.FINALIZER, END)

    return builder.compile(checkpointer=checkpointer)
