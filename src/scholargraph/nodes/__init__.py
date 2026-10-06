"""图的节点。每个节点是一个普通函数：读取状态，返回要更新的字段。"""

from scholargraph.nodes import names
from scholargraph.nodes.critic import critique_coverage
from scholargraph.nodes.finalizer import finalize_report
from scholargraph.nodes.human_review import review_plan
from scholargraph.nodes.planner import plan_research
from scholargraph.nodes.researcher import research_sub_question
from scholargraph.nodes.reviser import revise_report
from scholargraph.nodes.verifier import verify_citations
from scholargraph.nodes.writer import write_report

__all__ = [
    "critique_coverage",
    "finalize_report",
    "names",
    "plan_research",
    "research_sub_question",
    "review_plan",
    "revise_report",
    "verify_citations",
    "write_report",
]
