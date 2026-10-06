"""图的状态：所有节点共享的"黑板"。

节点不直接修改状态，而是返回一个只含变更字段的字典，由 LangGraph 合并进去。
合并方式有两种：
- 普通字段：新值覆盖旧值。
- 带 reducer 的字段（下面用 `Annotated` 标注的三个）：新旧值交给 reducer 合并。
  多个 Researcher 并行写同一个字段时，必须靠 reducer 才不会互相覆盖。
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from scholargraph.schemas import CitationIssue, Note, Paper, SubQuestion, VerificationRound


def merge_evidence(current: dict[str, Paper], new: dict[str, Paper]) -> dict[str, Paper]:
    """证据池的 reducer：按论文 ID 合并，同一篇论文只保留一份。

    同一篇论文可能被几个 Researcher 分别采用；后来者没有给出主要结论时，保留先前的。
    """
    merged = {**current, **new}
    for paper_id, paper in new.items():
        previous = current.get(paper_id)
        if previous is not None and previous.main_conclusion and not paper.main_conclusion:
            merged[paper_id] = paper.model_copy(
                update={"main_conclusion": previous.main_conclusion}
            )
    return merged


class ResearchState(TypedDict, total=False):
    """一次调研的完整状态。字段按它们在流程中出现的先后排列。"""

    # 输入
    question: str

    # 规划与人工审批
    scope: str  # 调研范围的界定：研究对象是什么、不包括什么
    plan: list[SubQuestion]
    plan_feedback: str | None  # 审批人要求修改时的意见；None 表示已通过

    # 检索（可能进行多轮）
    pending: list[SubQuestion]  # 下一轮要并行检索的子问题
    evidence: Annotated[dict[str, Paper], merge_evidence]  # 证据池：论文 ID -> 论文
    notes: Annotated[list[Note], operator.add]
    research_rounds: int  # 已完成的检索轮数
    open_gaps: list[str]  # 不再补充检索时仍未覆盖的要点，Writer 会如实写进"局限"

    # 写作、引用校验与修订
    draft: str
    issues: list[CitationIssue]  # 最近一轮校验发现的问题
    verified_claims: list[str]  # 已确认被摘要支撑的句子；修订后只需校验改动过的句子
    unverified_claims: list[str]  # 模型没有给出判断的句子，会在报告里如实注明
    verification_log: Annotated[list[VerificationRound], operator.add]  # 每一轮校验的统计
    needs_revision: bool
    revisions: int  # 已完成的修订次数

    # 输出
    report: str


class ResearcherInput(TypedDict):
    """通过 `Send` 派发给单个 Researcher 的私有输入，不属于共享状态。"""

    question: str
    scope: str
    sub_question: SubQuestion
