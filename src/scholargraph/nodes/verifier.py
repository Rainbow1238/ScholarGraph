"""引用校验：逐条核对报告里带引用的论断。

分两步，先便宜后贵：
1. 确定性检查（代码）：
   - 引用的 ID 不在证据池里，就是虚构引用。
   - 核心论断（概述里的句子、带“总体而言”这类概括性措辞的句子）引用的论文全部是弱证据，
     就是证据不足：综述的结论不能只靠被引 0 次的论文或学位论文撑起来。
2. 语义检查（LLM）：ID 存在时，判断论断是否真的被那篇论文的摘要支撑，
   以及论文研究的对象是否在调研范围之内。

语义检查分批进行：一次交给模型的论断太多时，它容易漏判或判得粗糙。
模型没有给出判断的论断不会被当作通过，而是再问一次，仍然没有结果的如实记为"未能校验"。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial

from scholargraph.citations import CitedClaim, core_claims, split_cited_claims
from scholargraph.config import Settings
from scholargraph.llm import LLM
from scholargraph.prompts import VERIFIER_SYSTEM, verifier_prompt
from scholargraph.quality import weakness_reasons
from scholargraph.schemas import CitationIssue, Paper, SupportVerdicts, VerificationRound
from scholargraph.state import ResearchState

MAX_PARALLEL_BATCHES = 4


@dataclass
class _SupportJudgement:
    """语义检查的结果：每条论断恰好落入四类之一。"""

    supported: list[CitedClaim] = field(default_factory=list)
    unsupported: list[CitationIssue] = field(default_factory=list)
    out_of_scope: list[CitationIssue] = field(default_factory=list)
    unjudged: list[CitedClaim] = field(default_factory=list)


def verify_citations(state: ResearchState, *, llm: LLM, settings: Settings) -> dict:
    evidence = state.get("evidence", {})
    claims = split_cited_claims(state["draft"])  # pyright: ignore[reportTypedDictNotRequiredAccess]
    fabricated, checkable = _split_by_citation_validity(claims, evidence)
    weak_support, checkable = _split_by_evidence_strength(
        checkable, core_claims(state["draft"]), evidence # type: ignore
    )

    # 修订只改动有问题的句子，其余句子与上一轮一字不差，校验结论仍然有效，不必重复花钱
    already_verified = set(state.get("verified_claims", []))
    still_valid = [claim for claim in checkable if claim.text in already_verified]
    to_judge = [claim for claim in checkable if claim.text not in already_verified]
    judgement = _judge_support(
        to_judge, evidence, state.get("scope", ""), llm, settings.verify_batch_size
    )

    issues = fabricated + weak_support + judgement.unsupported + judgement.out_of_scope
    has_budget = state.get("revisions", 0) < settings.max_revisions
    return {
        "issues": issues,
        "verified_claims": [claim.text for claim in still_valid + judgement.supported],
        "unverified_claims": [claim.text for claim in judgement.unjudged],
        "needs_revision": bool(issues) and has_budget,
        "verification_log": [
            VerificationRound(
                checked=len(claims),
                fabricated=len(fabricated),
                unsupported=len(judgement.unsupported),
                unverified=len(judgement.unjudged),
                out_of_scope=len(judgement.out_of_scope),
                weak_support=len(weak_support),
                issues=issues,
            )
        ],
    }


def _split_by_citation_validity(
    claims: list[CitedClaim],
    evidence: dict[str, Paper],
) -> tuple[list[CitationIssue], list[CitedClaim]]:
    """把论断分成两组：引用了未知 ID 的（直接记为问题），和引用全部有效的（待语义核查）。"""
    fabricated: list[CitationIssue] = []
    checkable: list[CitedClaim] = []
    for claim in claims:
        unknown_ids = [paper_id for paper_id in claim.paper_ids if paper_id not in evidence]
        if unknown_ids:
            fabricated.append(
                CitationIssue(
                    kind="fabricated",
                    claim=claim.text,
                    paper_ids=unknown_ids,
                    reason=f"证据池中不存在这些论文 ID：{', '.join(unknown_ids)}",
                )
            )
        else:
            checkable.append(claim)
    return fabricated, checkable


def _split_by_evidence_strength(
    claims: list[CitedClaim],
    core: list[CitedClaim],
    evidence: dict[str, Paper],
) -> tuple[list[CitationIssue], list[CitedClaim]]:
    """找出只由弱证据支撑的核心论断（直接记为问题），其余论断交给语义核查。

    这些句子反正要修订，就不再花钱做语义核查；修订后的新句子会在下一轮重新核查。
    """
    core_texts = {claim.text for claim in core}
    weak_support: list[CitationIssue] = []
    others: list[CitedClaim] = []
    for claim in claims:
        reasons = {pid: weakness_reasons(evidence[pid]) for pid in claim.paper_ids}
        if claim.text in core_texts and all(reasons.values()):
            details = "；".join(f"{pid}：{'、'.join(why)}" for pid, why in reasons.items())
            weak_support.append(
                CitationIssue(
                    kind="weak_support",
                    claim=claim.text,
                    paper_ids=list(claim.paper_ids),
                    reason=f"这是概述或概括性的论断，但所引论文全部是弱证据（{details}）",
                )
            )
        else:
            others.append(claim)
    return weak_support, others


def _judge_support(
    claims: list[CitedClaim],
    evidence: dict[str, Paper],
    scope: str,
    llm: LLM,
    batch_size: int,
) -> _SupportJudgement:
    """分批核查所有论断；第一遍被模型漏掉的论断再问一遍。"""
    judgement = _SupportJudgement()
    remaining = claims
    for _attempt in range(2):
        if not remaining:
            break
        batches = [remaining[i : i + batch_size] for i in range(0, len(remaining), batch_size)]
        judge_batch = partial(_judge_batch, evidence=evidence, scope=scope, llm=llm)
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as pool:  # 各批互不依赖，并行核查
            batch_results = list(pool.map(judge_batch, batches))

        remaining = []
        for result in batch_results:
            judgement.supported += result.supported
            judgement.unsupported += result.unsupported
            judgement.out_of_scope += result.out_of_scope
            remaining += result.unjudged

    judgement.unjudged = remaining
    return judgement


def _judge_batch(
    claims: list[CitedClaim], evidence: dict[str, Paper], scope: str, llm: LLM
) -> _SupportJudgement:
    """用一次 LLM 调用核查一批论断。只把这一批引用到的论文摘要发给模型。"""
    cited_ids = sorted({paper_id for claim in claims for paper_id in claim.paper_ids})
    cited_papers = [evidence[paper_id] for paper_id in cited_ids]
    reply = llm.complete_structured(
        VERIFIER_SYSTEM, verifier_prompt(claims, cited_papers, scope), SupportVerdicts
    )
    # 编号 -> 判断；越界的编号（模型的笔误）直接忽略
    verdict_by_index = {
        v.claim_index: v for v in reply.verdicts if 0 <= v.claim_index < len(claims)
    }

    result = _SupportJudgement()
    for index, claim in enumerate(claims):
        verdict = verdict_by_index.get(index)
        if verdict is None:
            result.unjudged.append(claim)
        elif verdict.supported and verdict.in_scope:
            result.supported.append(claim)
        elif verdict.supported:
            result.out_of_scope.append(
                CitationIssue(
                    kind="out_of_scope",
                    claim=claim.text,
                    paper_ids=list(claim.paper_ids),
                    reason=f"研究对象不在调研范围内。{verdict.reason}",
                )
            )
        else:
            result.unsupported.append(
                CitationIssue(
                    kind="unsupported",
                    claim=claim.text,
                    paper_ids=list(claim.paper_ids),
                    reason=verdict.reason,
                )
            )
    return result
