"""独立的评委：对任何系统生成的报告用同一套标准打分。

评三件事：
1. 引用的论文是否真实存在（代码判断：在系统的检索结果里，或者联网能查到）。
2. 论断是否被所引论文的摘要支撑（LLM 判断，分批进行）。
3. 报告是否覆盖了人工写的参考要点（LLM 判断）。

注意：评委与完整系统内部的引用校验器用的是同一类方法（同一个模型时尤其如此），
它对完整系统的打分可能偏乐观。这个偏差要靠人工抽检来度量，见 `agreement.py`。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial

from pydantic import BaseModel, Field

from scholargraph.citations import (
    CitedClaim,
    extract_paper_ids,
    split_cited_claims,
    strip_citations,
)
from scholargraph.evaluation.prompts import (
    COVERAGE_JUDGE_SYSTEM,
    SUPPORT_JUDGE_SYSTEM,
    coverage_judge_prompt,
    support_judge_prompt,
)
from scholargraph.evaluation.records import AspectJudgement, ClaimJudgement, ReportEvaluation
from scholargraph.evaluation.resolver import ArxivIdResolver
from scholargraph.llm import LLM
from scholargraph.schemas import Paper, SupportVerdicts

BATCH_SIZE = 15
MAX_PARALLEL_BATCHES = 4
MAX_ATTEMPTS = 2  # 评委漏判的论断再问一次


class AspectVerdict(BaseModel):
    aspect_index: int = Field(description="要点的编号，与输入中的编号一致")
    covered: bool = Field(description="综述是否实质性地讨论了这个要点")
    reason: str = Field(description="一句话理由")


class AspectVerdicts(BaseModel):
    verdicts: list[AspectVerdict]


def evaluate_report(
    report: str,
    evidence: dict[str, Paper],
    aspects: list[str],
    *,
    llm: LLM,
    resolver: ArxivIdResolver | None,
) -> ReportEvaluation:
    """评测一份报告。`evidence` 是系统自己检索到的论文；其余被引用的 ID 交给 `resolver` 核实。"""
    claims = split_cited_claims(report)
    cited_ids = extract_paper_ids(report)
    unknown_ids = [paper_id for paper_id in cited_ids if paper_id not in evidence]

    resolved: dict[str, Paper] = {}
    unreachable: set[str] = set()
    if unknown_ids and resolver is not None:
        resolution = resolver.resolve(unknown_ids)
        resolved, unreachable = resolution.found, resolution.unreachable
    known_papers = {**evidence, **resolved}

    judgements: dict[CitedClaim, ClaimJudgement] = {}
    checkable: list[CitedClaim] = []
    for claim in claims:
        missing = [pid for pid in claim.paper_ids if pid not in known_papers]
        if not missing:
            checkable.append(claim)
        elif any(pid in unreachable for pid in missing):
            judgements[claim] = _judgement(claim, "unjudged", "无法联网核实所引论文是否存在")
        else:
            judgements[claim] = _judgement(claim, "fabricated", f"查无此文：{', '.join(missing)}")
    judgements.update(judge_support(checkable, known_papers, llm))

    return ReportEvaluation(
        claims=[judgements[claim] for claim in dict.fromkeys(claims)],  # 保持在报告中的顺序
        aspects=judge_coverage(report, aspects, llm),
        characters=len(strip_citations(report)),
        cited_papers=len(cited_ids),
        resolved_papers=resolved,
    )


def judge_support(
    claims: list[CitedClaim], papers: dict[str, Paper], llm: LLM
) -> dict[CitedClaim, ClaimJudgement]:
    """分批判断每条论断是否被摘要支撑；评委两次都没给出判断的记为 unjudged。"""
    judgements: dict[CitedClaim, ClaimJudgement] = {}
    remaining = list(dict.fromkeys(claims))

    for _attempt in range(MAX_ATTEMPTS):
        if not remaining:
            break
        batches = [remaining[i : i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
        judge_batch = partial(_judge_support_batch, papers=papers, llm=llm)
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as pool:  # 各批互不依赖，并行判断
            for batch_judgements in pool.map(judge_batch, batches):
                judgements.update(batch_judgements)
        remaining = [claim for claim in remaining if claim not in judgements]

    for claim in remaining:
        judgements[claim] = _judgement(claim, "unjudged", "评委没有给出判断")
    return judgements


def judge_coverage(report: str, aspects: list[str], llm: LLM) -> list[AspectJudgement]:
    """判断报告覆盖了哪些参考要点。评委没有给出判断的要点保守地记为未覆盖。"""
    reply = llm.complete_structured(
        COVERAGE_JUDGE_SYSTEM, coverage_judge_prompt(report, aspects), AspectVerdicts
    )
    verdict_by_index = {verdict.aspect_index: verdict for verdict in reply.verdicts}

    judgements = []
    for index, aspect in enumerate(aspects):
        verdict = verdict_by_index.get(index)
        if verdict is None:
            judgements.append(
                AspectJudgement(aspect=aspect, covered=False, reason="评委没有给出判断")
            )
        else:
            judgements.append(
                AspectJudgement(aspect=aspect, covered=verdict.covered, reason=verdict.reason)
            )
    return judgements


def _judge_support_batch(
    claims: list[CitedClaim], papers: dict[str, Paper], llm: LLM
) -> dict[CitedClaim, ClaimJudgement]:
    """用一次 LLM 调用判断一批论断；只把这一批引用到的摘要发给评委。"""
    cited_ids = sorted({paper_id for claim in claims for paper_id in claim.paper_ids})
    prompt = support_judge_prompt(claims, [papers[paper_id] for paper_id in cited_ids])
    reply = llm.complete_structured(SUPPORT_JUDGE_SYSTEM, prompt, SupportVerdicts)

    judgements = {}
    for verdict in reply.verdicts:
        if 0 <= verdict.claim_index < len(claims):  # 越界的编号是评委的笔误，忽略
            claim = claims[verdict.claim_index]
            label = "supported" if verdict.supported else "unsupported"
            judgements[claim] = _judgement(claim, label, verdict.reason)
    return judgements


def _judgement(claim: CitedClaim, verdict: str, reason: str) -> ClaimJudgement:
    return ClaimJudgement(
        claim=claim.text, paper_ids=list(claim.paper_ids), verdict=verdict, reason=reason
    )
