"""运行指标：把一次调研的关键数字汇总成一个字典，保存为 JSON。

评测阶段要比较不同配置的虚构引用率、引用支撑率、覆盖情况、成本和耗时，
这些数字都从这里取，不需要再去解析终端输出或报告正文。
"""

from __future__ import annotations

from collections import Counter

from scholargraph.citations import extract_paper_ids, strip_citations
from scholargraph.llm import TokenUsage
from scholargraph.quality import is_weak_evidence, publication_label
from scholargraph.state import ResearchState


def build_run_metrics(
    state: ResearchState,
    *,
    model: str,
    usage: TokenUsage,
    source_stats: dict[str, dict],
    elapsed_seconds: float,
) -> dict:
    notes = state.get("notes", [])
    evidence = state.get("evidence", {})
    draft = state.get("draft", "")
    cited_papers = [evidence[pid] for pid in extract_paper_ids(draft) if pid in evidence]

    return {
        "question": state.get("question"),
        "scope": state.get("scope"),
        "model": model,
        "elapsed_seconds": round(elapsed_seconds, 1),  # 包含等待人工审批的时间
        "llm": {
            "calls": usage.calls,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        },
        "research": {
            "rounds": state.get("research_rounds", 0),
            "sub_questions": len(notes),
            "sub_questions_without_findings": sum(1 for note in notes if not note.findings),
            "candidates_read": sum(note.retrieved_count for note in notes),
            "findings": sum(len(note.findings) for note in notes),
            "discarded_findings": sum(note.discarded_findings for note in notes),
            "evidence_papers": len(evidence),
            "open_gaps": state.get("open_gaps", []),
        },
        # 注意：这部分统计保存在内存里，用 --resume 续跑时只包含续跑之后的请求
        "search_sources": source_stats,
        "verification": {
            # 每一轮校验的统计；第 1 轮是修订之前的引用质量
            "rounds": [entry.model_dump() for entry in state.get("verification_log", [])],
            "revisions": state.get("revisions", 0),
            "remaining_issues": [issue.model_dump() for issue in state.get("issues", [])],
            "unverified_claims": len(state.get("unverified_claims", [])),
        },
        "report": {
            "characters": len(draft),
            "characters_without_citations": len(strip_citations(draft)),
            "cited_papers": len(cited_papers),
            "cited_papers_by_source": dict(
                Counter(source for paper in cited_papers for source in paper.sources)
            ),
            "cited_papers_with_venue": sum(1 for paper in cited_papers if paper.venue),
            "cited_papers_by_publication": dict(
                Counter(publication_label(paper) for paper in cited_papers).most_common()
            ),
            "weak_evidence_papers": sum(1 for paper in cited_papers if is_weak_evidence(paper)),
        },
    }
