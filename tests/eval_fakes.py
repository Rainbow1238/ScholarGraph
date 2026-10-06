"""评测测试共用的构造工具。"""

from __future__ import annotations

from fakes import make_paper

from scholargraph.evaluation.records import (
    AspectJudgement,
    ClaimJudgement,
    ReportEvaluation,
    RunRecord,
    SystemOutput,
    Usage,
)

PAPER = make_paper("2401.00001", "Paper A")


def make_evaluation(verdicts: str, covered: tuple[bool, ...] = (True, False)) -> ReportEvaluation:
    """用一串字母描述各条论断的判断：s=支撑，u=不支撑，f=虚构，j=未判定。"""
    names = {"s": "supported", "u": "unsupported", "f": "fabricated", "j": "unjudged"}
    return ReportEvaluation(
        claims=[
            ClaimJudgement(
                claim=f"论断 {n} [arXiv:2401.00001]。",
                paper_ids=[PAPER.paper_id],
                verdict=names[letter],
            )
            for n, letter in enumerate(verdicts)
        ],
        aspects=[
            AspectJudgement(aspect=f"要点 {n}", covered=flag) for n, flag in enumerate(covered)
        ],
        characters=1000,
        cited_papers=4,
    )


def make_record(
    system: str,
    question_id: str = "q1",
    verdicts: str = "ss",
    *,
    covered: tuple[bool, ...] = (True, False),
    first_draft_verdicts: str | None = None,
    tokens: int = 1000,
    seconds: float = 10.0,
) -> RunRecord:
    return RunRecord(
        question_id=question_id,
        question="研究问题",
        system=system,
        output=SystemOutput(report="正文", evidence={PAPER.paper_id: PAPER}),
        usage=Usage(calls=3, prompt_tokens=tokens - 100, completion_tokens=100),
        elapsed_seconds=seconds,
        evaluation=make_evaluation(verdicts, covered),
        first_draft_evaluation=(
            make_evaluation(first_draft_verdicts, covered) if first_draft_verdicts else None
        ),
    )
