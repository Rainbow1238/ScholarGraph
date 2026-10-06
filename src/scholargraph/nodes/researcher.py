"""Researcher：针对一个子问题检索论文并提炼有出处的发现。

这个节点会被 `Send` 同时启动多份，每份处理一个子问题；
它们各自返回的 evidence 和 notes 由状态里的 reducer 合并。
"""

from __future__ import annotations

import logging

from scholargraph.citations import normalize_paper_id
from scholargraph.config import Settings
from scholargraph.llm import LLM
from scholargraph.prompts import RESEARCHER_SYSTEM, researcher_prompt
from scholargraph.quality import demote_weak_evidence
from scholargraph.schemas import Finding, FindingList, Note, Paper, PaperConclusion
from scholargraph.search import PaperSearcher, SearchError
from scholargraph.search.fusion import fuse
from scholargraph.state import ResearcherInput

logger = logging.getLogger(__name__)

# 弱证据论文在候选排序中后移的位数：有足够多的其他论文时它们排不进候选列表，
# 某个方向只有弱证据时仍然能被模型看到
WEAK_EVIDENCE_DEMOTION = 10


def research_sub_question(
    task: ResearcherInput,
    *,
    llm: LLM,
    searcher: PaperSearcher,
    settings: Settings,
) -> dict:
    """流程：多组检索词分别检索 -> 融合成候选论文 -> 由模型挑选并提炼发现 -> 校验出处。"""
    sub_question = task["sub_question"]
    result_lists = _search_each_query(
        sub_question.search_queries, searcher, settings.papers_per_query
    )
    # 被多组检索词同时排在前面的论文最可能相关；弱证据降权后，只把排名靠前的交给模型
    ranked = demote_weak_evidence(fuse(result_lists), WEAK_EVIDENCE_DEMOTION)
    candidates = ranked[: settings.max_candidates_per_sub_question]
    candidate_by_id = {paper.paper_id: paper for paper in candidates}

    proposed: list[Finding] = []
    conclusions: list[PaperConclusion] = []
    if candidates:
        prompt = researcher_prompt(
            question=task["question"],
            scope=task["scope"],
            sub_question=sub_question,
            papers=candidates,
            max_findings=settings.max_findings_per_sub_question,
        )
        reply = llm.complete_structured(RESEARCHER_SYSTEM, prompt, FindingList)
        proposed, conclusions = reply.findings, reply.paper_conclusions

    grounded = _keep_grounded(proposed, candidate_by_id)
    # 模型已按重要性排序，这里用代码保证数量上限：下游的综述要的是代表性工作，不是论文清单
    findings = grounded[: settings.max_findings_per_sub_question]
    note = Note(
        sub_question=sub_question.question,
        search_queries=sub_question.search_queries,
        retrieved_count=len(candidates),
        discarded_findings=len(proposed) - len(grounded),
        findings=findings,
    )
    evidence = _cited_papers(findings, candidate_by_id, _conclusion_by_id(conclusions))
    return {"notes": [note], "evidence": evidence}


def _search_each_query(
    queries: list[str], searcher: PaperSearcher, per_query: int
) -> list[list[Paper]]:
    """依次用每组检索词检索，返回每组各自的结果列表（保留各自的排序，供融合使用）。

    个别检索词失败时继续使用其余检索词的结果；只有全部失败才向上抛出，
    交给图上配置的重试策略处理。
    """
    result_lists: list[list[Paper]] = []
    last_error: SearchError | None = None

    for query in queries:
        try:
            result_lists.append(searcher.search(query, per_query))
        except SearchError as error:
            logger.warning("检索词 %r 检索失败，已跳过：%s", query, error)
            last_error = error

    if last_error is not None and not result_lists:
        raise SearchError(f"全部 {len(queries)} 组检索词都检索失败") from last_error
    return result_lists


def _keep_grounded(findings: list[Finding], candidates: dict[str, Paper]) -> list[Finding]:
    """只保留有出处、且出处全部来自候选论文的发现。

    模型写的 ID 先做归一化（前缀大小写、版本号等），
    避免因为写法不同而把有效的发现误判为无出处。
    """
    grounded: list[Finding] = []
    for finding in findings:
        paper_ids = list(dict.fromkeys(normalize_paper_id(raw_id) for raw_id in finding.paper_ids))
        if paper_ids and all(paper_id in candidates for paper_id in paper_ids):
            grounded.append(Finding(statement=finding.statement, paper_ids=paper_ids))
    return grounded


def _conclusion_by_id(conclusions: list[PaperConclusion]) -> dict[str, str]:
    by_id: dict[str, str] = {}
    for item in conclusions:
        if item.conclusion.strip():
            by_id.setdefault(normalize_paper_id(item.paper_id), item.conclusion.strip())
    return by_id


def _cited_papers(
    findings: list[Finding], candidates: dict[str, Paper], conclusions: dict[str, str]
) -> dict[str, Paper]:
    """只有被发现引用过的论文才进入证据池，保持证据池精简；同时记下模型概括的主要结论。"""
    cited_ids = {paper_id for finding in findings for paper_id in finding.paper_ids}
    return {
        paper_id: candidates[paper_id].model_copy(
            update={"main_conclusion": conclusions.get(paper_id)}
        )
        for paper_id in sorted(cited_ids)
    }
