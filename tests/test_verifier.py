"""引用校验节点的测试：核心论断的证据强度检查、范围检查。"""

from __future__ import annotations

from fakes import FakeLLM, make_paper

from scholargraph.config import Settings
from scholargraph.nodes.verifier import verify_citations
from scholargraph.prompts import revision_prompt
from scholargraph.schemas import SupportVerdict, SupportVerdicts

SCOPE = "间歇性禁食对一般成人的影响；不包括 2 型糖尿病患者"
STRONG = make_paper(
    "OpenAlex:W1",
    "Calorie Restriction with or without Time-Restricted Eating",
    venue="NEJM",
    citation_count=529,
    year=2022,
    main_conclusion="限时进食并不比单纯热量限制更有益",
)
WEAK = make_paper(
    "OpenAlex:W2", "Single-author meta-analysis", venue="J", citation_count=0, year=2020
)
EVIDENCE = {paper.paper_id: paper for paper in (STRONG, WEAK)}


def verify(draft: str, verdicts: list[SupportVerdict]):
    llm = FakeLLM(structured={SupportVerdicts: [SupportVerdicts(verdicts=verdicts)]}, texts=[])
    state = {"draft": draft, "evidence": EVIDENCE, "scope": SCOPE}
    return verify_citations(state, llm=llm, settings=Settings(api_key="k")), llm


def verdict(index: int, *, supported: bool = True, in_scope: bool = True) -> SupportVerdict:
    return SupportVerdict(claim_index=index, supported=supported, in_scope=in_scope, reason="理由")


def test_overview_claim_backed_only_by_weak_evidence_is_flagged_without_asking_the_model():
    draft = "# 综述\n\n## 概述\n\n间歇性禁食显著改善胰岛素敏感性 [OpenAlex:W2]。\n"
    update, llm = verify(draft, [])

    (issue,) = update["issues"]
    assert issue.kind == "weak_support"
    assert "OpenAlex:W2：被引 0 次" in issue.reason
    assert update["verification_log"][0].weak_support == 1
    assert "SupportVerdicts" not in llm.prompts  # 反正要修订，不必再花钱做语义核查


def test_weak_evidence_is_fine_outside_core_claims_or_alongside_strong_evidence():
    draft = (
        "# 综述\n\n## 概述\n\n效果与热量限制相当 [OpenAlex:W1][OpenAlex:W2]。\n\n"
        "## 胰岛素\n\n一项荟萃分析报告 HOMA-IR 下降 [OpenAlex:W2]。\n"
    )
    update, _ = verify(draft, [verdict(0), verdict(1)])

    assert update["issues"] == []


def test_supported_claim_about_an_excluded_population_is_out_of_scope():
    draft = "# 综述\n\n## 依从性\n\n在 2 型糖尿病成人中依从性良好 [OpenAlex:W1]。\n"
    update, llm = verify(draft, [verdict(0, in_scope=False)])

    (issue,) = update["issues"]
    assert issue.kind == "out_of_scope"
    assert update["needs_revision"]
    assert update["verification_log"][0].out_of_scope == 1
    assert SCOPE in llm.prompts["SupportVerdicts"][0]  # 核查员能看到调研范围


def test_reviser_is_offered_strong_alternatives_only_for_weak_support_issues():
    draft = "# 综述\n\n## 概述\n\n间歇性禁食显著改善胰岛素敏感性 [OpenAlex:W2]。\n"
    update, _ = verify(draft, [])
    prompt = revision_prompt(update["issues"], EVIDENCE)

    assert "其他可引用的论文" in prompt
    assert "ID: OpenAlex:W1" in prompt
    assert "主要结论：限时进食并不比单纯热量限制更有益" in prompt
