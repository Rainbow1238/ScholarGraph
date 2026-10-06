"""运行指标的测试：跑一遍完整流程，检查汇总出来的数字。"""

from __future__ import annotations

import json

from fakes import FakeSearcher
from test_graph import (
    FABRICATED_DRAFT,
    PAPER_A,
    PAPER_B,
    QUESTION,
    SCOPE,
    delete,
    make_graph,
    make_llm,
    run,
    verdicts,
)

from scholargraph.llm import TokenUsage
from scholargraph.metrics import build_run_metrics
from scholargraph.search import FederatedSearcher


def test_metrics_summarise_a_run_with_one_revision():
    searcher = FederatedSearcher(
        {"arXiv": FakeSearcher([PAPER_A]), "OpenAlex": FakeSearcher([PAPER_B])}
    )
    llm = make_llm(
        drafts=[FABRICATED_DRAFT], verdict_replies=[verdicts(True)], corrections=[[delete(0)]]
    )
    state = run(make_graph(llm, searcher), {"configurable": {"thread_id": "t"}})
    usage = TokenUsage(calls=7, prompt_tokens=1000, completion_tokens=200)

    metrics = build_run_metrics(
        state, model="fake-model", usage=usage, source_stats=searcher.stats(), elapsed_seconds=12.34
    )

    assert metrics["question"] == QUESTION and metrics["scope"] == SCOPE
    assert metrics["elapsed_seconds"] == 12.3
    assert metrics["llm"] == {"calls": 7, "prompt_tokens": 1000, "completion_tokens": 200}
    assert metrics["research"]["sub_questions"] == 2
    assert metrics["research"]["sub_questions_without_findings"] == 0
    assert metrics["research"]["findings"] == 4
    assert metrics["search_sources"]["OpenAlex"]["requests"] == 4
    # 第 1 轮：2 条论断，其中 1 条虚构引用；修订后第 2 轮：剩 1 条，没有问题
    first_round, second_round = metrics["verification"]["rounds"]
    assert (first_round["checked"], first_round["fabricated"], first_round["unsupported"]) == (
        2,
        1,
        0,
    )
    assert (
        first_round["issues"][0]["claim"] == "RAG 已被彻底解决 [arXiv:9999.99999]。"
    )  # 问题明细留档
    assert (second_round["checked"], second_round["issues"]) == (1, [])
    assert metrics["verification"]["revisions"] == 1
    assert metrics["report"]["cited_papers"] == 1
    assert metrics["report"]["cited_papers_by_source"] == {"arXiv": 1}
    json.dumps(metrics, ensure_ascii=False)  # 必须能直接存成 JSON
