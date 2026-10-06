"""评测结果汇总的测试。"""

from __future__ import annotations

import pytest
from eval_fakes import make_record

from scholargraph.evaluation.summary import FULL_BEFORE_REVISION, render_markdown, summarize


def by_system(records) -> dict:
    return {summary.system: summary for summary in summarize(records)}


def test_rates_are_pooled_over_all_claims_of_a_system():
    records = [
        make_record("direct", "q1", "sfff"),  # 4 条论断：1 支撑，3 虚构
        make_record("direct", "q2", "ssssuj"),  # 6 条论断：4 支撑，1 不支撑，1 未判定
    ]
    summary = by_system(records)["direct"]

    assert (summary.runs, summary.claims) == (2, 10)
    assert summary.fabricated_rate == pytest.approx(0.3)
    assert summary.support_rate == pytest.approx(0.5)
    assert summary.unjudged_rate == pytest.approx(0.1)


def test_coverage_cost_and_time_are_averaged_per_question():
    records = [
        make_record("full", "q1", covered=(True, True), tokens=1000, seconds=10),
        make_record("full", "q2", covered=(True, False, False, False), tokens=3000, seconds=30),
    ]
    summary = by_system(records)["full"]

    assert summary.coverage == pytest.approx((1.0 + 0.25) / 2)
    assert (summary.tokens, summary.seconds) == (2000, 20)


def test_first_drafts_of_the_full_system_form_their_own_row():
    records = [
        make_record("full", "q1", "ss", first_draft_verdicts="sf"),  # 修订把虚构引用改掉了
        make_record("full", "q2", "ss"),  # 没有发生修订：初稿就是终稿
    ]
    summaries = by_system(records)

    before, after = summaries[FULL_BEFORE_REVISION], summaries["full"]
    assert (before.runs, after.runs) == (2, 2)  # 两行覆盖的是同一批题目
    assert before.fabricated_rate == pytest.approx(0.25)
    assert after.fabricated_rate == 0


def test_rows_run_from_the_simplest_system_to_the_most_complete():
    records = [make_record(name) for name in ("full", "no_critic", "direct", "single_agent")]

    assert [s.system for s in summarize(records)] == [
        "direct",
        "single_agent",
        "no_critic",
        FULL_BEFORE_REVISION,
        "full",
    ]


def test_system_without_any_cited_claim_has_zero_rates_instead_of_crashing():
    summary = by_system([make_record("direct", verdicts="")])["direct"]

    assert (summary.claims, summary.support_rate, summary.fabricated_rate) == (0, 0.0, 0.0)


def test_markdown_table_has_one_row_per_system_with_readable_names():
    table = render_markdown(summarize([make_record("direct", verdicts="sf"), make_record("full")]))

    assert "| 直接回答（无检索） | 1 | 2 | 50.0% | 50.0% |" in table
    assert "| 完整系统 | 1 | 2 | 0.0% | 100.0% |" in table
    assert "引用支撑率：" in table  # 表格下面附有指标的定义


def test_empty_results_are_stated_plainly():
    assert render_markdown([]) == "还没有已完成的评测结果。"
