"""条件边：根据状态决定下一步去哪个节点。

路由函数只读状态、不调用模型，也不做预算判断——
预算（最多几轮检索、几次修订）已由 Critic 和 Verifier 节点写进状态，
所以这里的每个函数都只有一个简单的分支。
"""

from __future__ import annotations

from langgraph.types import Send

from scholargraph.nodes import names
from scholargraph.state import ResearcherInput, ResearchState


def route_after_review(state: ResearchState) -> str | list[Send]:
    """计划被打回 -> 回到 Planner；通过 -> 并行启动 Researcher。"""
    if state.get("plan_feedback"):
        return names.PLANNER
    return _dispatch_researchers(state)


def route_after_critic(state: ResearchState) -> str | list[Send]:
    """还有待补充的子问题 -> 再检索一轮；否则 -> 开始写作。"""
    if state.get("pending"):
        return _dispatch_researchers(state)
    return names.WRITER


def route_after_verification(state: ResearchState) -> str:
    """有问题且还能修订 -> 交给 Reviser 逐句修订；否则 -> 定稿。"""
    return names.REVISER if state.get("needs_revision") else names.FINALIZER


def _dispatch_researchers(state: ResearchState) -> list[Send]:
    """为每个待检索的子问题创建一个 `Send`（map-reduce 中的 map）。

    每个 `Send` 会启动一个独立的 Researcher，并把它自己的输入单独交给它；
    同一批 `Send` 在同一个 superstep 内并行执行，全部完成后才进入下一个节点。
    """
    return [
        Send(
            names.RESEARCHER,
            ResearcherInput(
                question=state["question"], scope=state["scope"], sub_question=sub_question # type: ignore
            ),
        )
        for sub_question in state["pending"] # type: ignore
    ]
