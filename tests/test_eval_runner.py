"""评测执行流程的测试：断点续跑、失败处理、检索源异常时停止。"""

from __future__ import annotations

import pytest
from fakes import FakeLLM, FakeSearcher, make_paper

from scholargraph.config import Settings
from scholargraph.evaluation import runner
from scholargraph.evaluation.dataset import EvalQuestion
from scholargraph.evaluation.judge import AspectVerdict, AspectVerdicts
from scholargraph.evaluation.records import SystemOutput
from scholargraph.evaluation.runner import (
    EvaluationAborted,
    EvaluationContext,
    load_all_records,
    load_record,
    record_path,
    run_evaluation,
)
from scholargraph.llm import LLMOutputError
from scholargraph.schemas import SupportVerdict, SupportVerdicts
from scholargraph.search import SearchError

PAPER = make_paper("2401.00001", "Paper A")
QUESTIONS = [
    EvalQuestion(id="q1", question="问题一", aspects=["要点甲", "要点乙"]),
    EvalQuestion(id="q2", question="问题二", aspects=["要点丙"]),
]
GOOD_REPORT = "结论 [arXiv:2401.00001]。"


class FakeSystem:
    """记录自己被调用了几次的假系统；可以设定在哪些题目上失败。"""

    def __init__(self, output: SystemOutput | None = None, fail_on: dict | None = None) -> None:
        self._output = output or SystemOutput(report=GOOD_REPORT, evidence={PAPER.paper_id: PAPER})
        self.fail_on = fail_on or {}
        self.calls: list[str] = []

    def __call__(self, question, llm, searcher, settings) -> SystemOutput:
        self.calls.append(question.id)
        if question.id in self.fail_on:
            raise self.fail_on[question.id]
        llm.usage.record(prompt_tokens=900, completion_tokens=100)
        return self._output


def make_judge() -> FakeLLM:
    return FakeLLM(
        structured={
            SupportVerdicts: [
                SupportVerdicts(
                    verdicts=[SupportVerdict(claim_index=0, supported=True, reason="ok")]
                )
            ],
            AspectVerdicts: [
                AspectVerdicts(verdicts=[AspectVerdict(aspect_index=0, covered=True, reason="ok")])
            ],
        },
        texts=[],
    )


def make_context(judge=None, source_stats=None) -> tuple[EvaluationContext, list[str]]:
    progress: list[str] = []
    context = EvaluationContext(
        settings=Settings(api_key="test-key"),
        searcher=FakeSearcher([PAPER]),
        make_llm=lambda: FakeLLM(structured={}, texts=[]),
        judge_llm=judge or make_judge(),
        resolver=None,
        source_stats=source_stats or dict,
        on_progress=progress.append,
    )
    return context, progress


@pytest.fixture
def systems(monkeypatch) -> dict[str, FakeSystem]:
    fakes = {"alpha": FakeSystem(), "beta": FakeSystem()}
    monkeypatch.setattr(runner, "SYSTEMS", {name: (system, name) for name, system in fakes.items()})
    return fakes


def test_every_question_is_run_and_judged_for_every_system(systems, tmp_path):
    context, progress = make_context()

    failures = run_evaluation(QUESTIONS, ["alpha", "beta"], tmp_path, context)

    assert failures == 0
    assert systems["alpha"].calls == systems["beta"].calls == ["q1", "q2"]
    record = load_record(record_path(tmp_path, "alpha", "q1"))
    assert record.evaluation.count("supported") == 1
    assert record.evaluation.covered_aspects == 1  # 评委只判了第一个要点，第二个按未覆盖计
    assert (record.usage.prompt_tokens, record.usage.completion_tokens) == (900, 100)
    assert len(load_all_records(tmp_path)) == 4
    assert progress[0].startswith("✓ [1/4] alpha / q1：1 条论断，支撑 1，虚构 0，覆盖 1/2 个要点")


def test_finished_runs_are_skipped_when_the_evaluation_is_run_again(systems, tmp_path):
    context, _ = make_context()
    run_evaluation(QUESTIONS[:1], ["alpha"], tmp_path, context)

    judge = make_judge()
    context, _ = make_context(judge)
    run_evaluation(QUESTIONS, ["alpha"], tmp_path, context)

    assert systems["alpha"].calls == ["q1", "q2"]  # q1 没有重新生成
    assert len(judge.prompts["AspectVerdicts"]) == 1  # 也没有重新评测


def test_failed_run_is_counted_and_retried_next_time(systems, tmp_path):
    systems["alpha"].fail_on = {"q1": LLMOutputError("模型输出不合法")}
    context, progress = make_context()

    failures = run_evaluation(QUESTIONS, ["alpha"], tmp_path, context)

    assert failures == 1
    assert "✗ [1/2] alpha / q1" in progress[0]
    assert [r.question_id for r in load_all_records(tmp_path)] == ["q2"]  # 其余的照常完成

    systems["alpha"].fail_on = {}
    assert run_evaluation(QUESTIONS, ["alpha"], tmp_path, context) == 0
    assert systems["alpha"].calls == ["q1", "q2", "q1"]  # 只重试了失败的那一次


def test_generated_report_is_kept_when_only_the_judge_fails(systems, tmp_path):
    class BrokenJudge:
        def complete_structured(self, system, user, schema):
            raise LLMOutputError("评委出错")

    context, _ = make_context(BrokenJudge())
    assert run_evaluation(QUESTIONS[:1], ["alpha"], tmp_path, context) == 1

    context, _ = make_context()
    assert run_evaluation(QUESTIONS[:1], ["alpha"], tmp_path, context) == 0
    assert systems["alpha"].calls == ["q1"]  # 报告没有重新生成，只补做了评测


def test_first_draft_is_judged_too_when_the_system_revised_its_report(monkeypatch, tmp_path):
    output = SystemOutput(
        report=GOOD_REPORT,
        evidence={PAPER.paper_id: PAPER},
        first_draft=f"{GOOD_REPORT}编造的结论 [arXiv:9999.99999]。",
    )
    monkeypatch.setattr(runner, "SYSTEMS", {"full": (FakeSystem(output), "完整系统")})
    context, _ = make_context()

    run_evaluation(QUESTIONS[:1], ["full"], tmp_path, context)

    record = load_record(record_path(tmp_path, "full", "q1"))
    assert record.first_draft_evaluation.count("fabricated") == 1
    assert record.evaluation.count("fabricated") == 0


def test_evaluation_stops_when_a_search_source_has_been_disabled(systems, tmp_path):
    """来源被停用后，后面的运行条件就和前面不同了：停下来，并且不保存这一次的结果。"""
    stats = {"OpenAlex": {"disabled_reason": None}}
    context, _ = make_context(source_stats=lambda: stats)

    def system_that_exhausts_the_quota(question, llm, searcher, settings):
        stats["OpenAlex"]["disabled_reason"] = "HTTP 429"
        return SystemOutput(report=GOOD_REPORT, evidence={PAPER.paper_id: PAPER})

    runner.SYSTEMS["alpha"] = (system_that_exhausts_the_quota, "alpha")

    with pytest.raises(EvaluationAborted, match="OpenAlex（HTTP 429）"):
        run_evaluation(QUESTIONS, ["alpha"], tmp_path, context)
    assert load_all_records(tmp_path) == []


def test_search_failure_caused_by_a_disabled_source_also_stops_the_evaluation(systems, tmp_path):
    stats = {"arXiv": {"disabled_reason": "连接超时"}}
    systems["alpha"].fail_on = {"q1": SearchError("所有检索源都已停用")}
    context, _ = make_context(source_stats=lambda: stats)

    with pytest.raises(EvaluationAborted):
        run_evaluation(QUESTIONS, ["alpha"], tmp_path, context)
    assert systems["alpha"].calls == ["q1"]  # 没有继续跑 q2
