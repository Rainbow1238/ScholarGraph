"""独立评委的测试。"""

from __future__ import annotations

from fakes import FakeLLM, make_paper

from scholargraph.citations import strip_citations
from scholargraph.evaluation import judge
from scholargraph.evaluation.judge import AspectVerdict, AspectVerdicts, evaluate_report
from scholargraph.evaluation.resolver import Resolution
from scholargraph.schemas import SupportVerdict, SupportVerdicts

PAPER_A = make_paper("2401.00001", "Paper A")
PAPER_B = make_paper("2402.00002", "Paper B")
EVIDENCE = {PAPER_A.paper_id: PAPER_A, PAPER_B.paper_id: PAPER_B}
ASPECTS = ["检索优化", "评测方法"]
REPORT = "# 综述\n\n甲结论 [arXiv:2401.00001]。乙结论 [arXiv:2402.00002]。没有引用的一句话。"


def support(*flags: bool) -> SupportVerdicts:
    return SupportVerdicts(
        verdicts=[
            SupportVerdict(claim_index=i, supported=flag, reason="有依据" if flag else "摘要未提及")
            for i, flag in enumerate(flags)
        ]
    )


def coverage(*flags: bool) -> AspectVerdicts:
    return AspectVerdicts(
        verdicts=[
            AspectVerdict(aspect_index=i, covered=flag, reason="理由")
            for i, flag in enumerate(flags)
        ]
    )


def make_llm(support_replies, coverage_reply=None) -> FakeLLM:
    return FakeLLM(
        structured={
            SupportVerdicts: list(support_replies),
            AspectVerdicts: [coverage_reply or coverage(True, False)],
        },
        texts=[],
    )


class FakeResolver:
    def __init__(self, found=None, unreachable=()) -> None:
        self._resolution = Resolution(found=dict(found or {}), unreachable=set(unreachable))
        self.asked: list[list[str]] = []

    def resolve(self, paper_ids):
        self.asked.append(list(paper_ids))
        return self._resolution


def verdicts_of(evaluation) -> list[str]:
    return [claim.verdict for claim in evaluation.claims]


# ───────────── 引用支撑 ─────────────


def test_each_cited_claim_is_judged_against_the_abstracts():
    llm = make_llm([support(True, False)])
    evaluation = evaluate_report(REPORT, EVIDENCE, ASPECTS, llm=llm, resolver=None)

    assert verdicts_of(evaluation) == ["supported", "unsupported"]
    assert evaluation.claims[1].reason == "摘要未提及"
    assert (evaluation.count("supported"), evaluation.count("unsupported")) == (1, 1)
    assert evaluation.cited_papers == 2
    prompt = llm.prompts["SupportVerdicts"][0]
    assert "Abstract of Paper A." in prompt and "没有引用的一句话" not in prompt


def test_citation_outside_the_evidence_is_fabricated_when_it_cannot_be_found():
    report = "甲结论 [arXiv:2401.00001]。编造的结论 [arXiv:9999.99999]。"
    resolver = FakeResolver()
    evaluation = evaluate_report(
        report, EVIDENCE, ASPECTS, llm=make_llm([support(True)]), resolver=resolver
    )

    assert verdicts_of(evaluation) == ["supported", "fabricated"]
    assert "9999.99999" in evaluation.claims[1].reason
    assert resolver.asked == [["arXiv:9999.99999"]]  # 只核实系统检索结果之外的 ID


def test_citation_from_memory_that_really_exists_is_judged_normally():
    """直接回答基线没有检索结果：引用的论文联网查到后，照常判断是否支撑。"""
    recalled = make_paper("2005.11401", "RAG original paper")
    resolver = FakeResolver(found={recalled.paper_id: recalled})
    llm = make_llm([support(True)])

    evaluation = evaluate_report(
        "RAG 结合了检索与生成 [arXiv:2005.11401]。", {}, ASPECTS, llm=llm, resolver=resolver
    )

    assert verdicts_of(evaluation) == ["supported"]
    assert "Abstract of RAG original paper." in llm.prompts["SupportVerdicts"][0]
    assert evaluation.resolved_papers == {recalled.paper_id: recalled}  # 留档，人工抽检时要用


def test_citation_that_could_not_be_checked_online_is_unjudged_not_fabricated():
    resolver = FakeResolver(unreachable=["arXiv:2005.11401"])
    evaluation = evaluate_report(
        "某结论 [arXiv:2005.11401]。", {}, ASPECTS, llm=make_llm([support()]), resolver=resolver
    )

    assert verdicts_of(evaluation) == ["unjudged"]


def test_without_a_resolver_unknown_ids_count_as_fabricated():
    evaluation = evaluate_report(
        "某结论 [arXiv:2005.11401]。", {}, ASPECTS, llm=make_llm([support()]), resolver=None
    )

    assert verdicts_of(evaluation) == ["fabricated"]


def test_claim_the_judge_skips_is_asked_again_then_marked_unjudged():
    only_first = SupportVerdicts(
        verdicts=[SupportVerdict(claim_index=0, supported=True, reason="ok")]
    )
    nothing = SupportVerdicts(verdicts=[])
    llm = make_llm([only_first, nothing])

    evaluation = evaluate_report(REPORT, EVIDENCE, ASPECTS, llm=llm, resolver=None)

    assert verdicts_of(evaluation) == ["supported", "unjudged"]
    assert len(llm.prompts["SupportVerdicts"]) == 2
    assert (
        "乙结论" in llm.prompts["SupportVerdicts"][1]
        and "甲结论" not in llm.prompts["SupportVerdicts"][1]
    )


def test_claims_are_judged_in_batches(monkeypatch):
    monkeypatch.setattr(judge, "BATCH_SIZE", 1)
    llm = make_llm([support(True)])

    evaluation = evaluate_report(REPORT, EVIDENCE, ASPECTS, llm=llm, resolver=None)

    assert verdicts_of(evaluation) == ["supported", "supported"]
    assert len(llm.prompts["SupportVerdicts"]) == 2


def test_report_without_citations_needs_no_support_judgement():
    llm = make_llm([support()])
    evaluation = evaluate_report("没有任何引用的报告。", EVIDENCE, ASPECTS, llm=llm, resolver=None)

    assert evaluation.claims == [] and evaluation.cited_papers == 0
    assert "SupportVerdicts" not in llm.prompts


# ───────────── 要点覆盖 ─────────────


def test_coverage_is_judged_per_aspect_on_the_text_without_citation_marks():
    llm = make_llm([support(True, True)], coverage(True, False))
    evaluation = evaluate_report(REPORT, EVIDENCE, ASPECTS, llm=llm, resolver=None)

    assert [(a.aspect, a.covered) for a in evaluation.aspects] == [
        ("检索优化", True),
        ("评测方法", False),
    ]
    assert evaluation.covered_aspects == 1
    prompt = llm.prompts["AspectVerdicts"][0]
    assert "[0] 检索优化" in prompt and "arXiv:" not in prompt
    assert evaluation.characters == len(strip_citations(REPORT)) < len(REPORT)  # 字数不含引用标记


def test_aspect_the_judge_skips_counts_as_not_covered():
    llm = make_llm([support(True, True)], coverage(True))  # 只判断了第一个要点
    evaluation = evaluate_report(REPORT, EVIDENCE, ASPECTS, llm=llm, resolver=None)

    assert [a.covered for a in evaluation.aspects] == [True, False]
    assert evaluation.aspects[1].reason == "评委没有给出判断"
