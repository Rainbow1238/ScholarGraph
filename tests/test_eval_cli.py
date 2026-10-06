"""评测命令行的测试。"""

from __future__ import annotations

import os

import pytest
from eval_fakes import make_record
from fakes import FakeLLM, FakeSearcher, make_paper

from scholargraph.evaluation import __main__ as eval_cli
from scholargraph.evaluation.judge import AspectVerdict, AspectVerdicts
from scholargraph.evaluation.resolver import Resolution
from scholargraph.evaluation.runner import record_path, save_record
from scholargraph.schemas import SupportVerdict, SupportVerdicts
from scholargraph.search import FederatedSearcher

QUESTION_LINE = '{"id": "q1", "question": "RAG 有哪些改进方向？", "aspects": ["检索", "生成"]}\n'


def test_summary_writes_the_comparison_table(tmp_path, capsys):
    save_record(make_record("direct", "q1", "sf"), record_path(tmp_path, "direct", "q1"))
    save_record(make_record("full", "q1", "ss"), record_path(tmp_path, "full", "q1"))

    assert eval_cli.main(["summary", "--out", str(tmp_path)]) == eval_cli.EXIT_OK

    table = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| 直接回答（无检索） | 1 | 2 | 50.0% |" in table
    assert "| 完整系统 |" in capsys.readouterr().out


def test_summary_of_an_empty_directory_does_not_crash(tmp_path, capsys):
    assert eval_cli.main(["summary", "--out", str(tmp_path / "nothing")]) == eval_cli.EXIT_OK
    assert "还没有已完成的评测结果" in capsys.readouterr().out


def test_sample_refuses_to_overwrite_a_sheet_that_may_hold_labels(tmp_path, capsys):
    save_record(make_record("full", "q1", "ssuu"), record_path(tmp_path, "full", "q1"))
    assert eval_cli.main(["sample", "--out", str(tmp_path), "--n", "4"]) == eval_cli.EXIT_OK

    assert eval_cli.main(["sample", "--out", str(tmp_path)]) == eval_cli.EXIT_ERROR
    assert "--force" in capsys.readouterr().out
    assert eval_cli.main(["sample", "--out", str(tmp_path), "--force"]) == eval_cli.EXIT_OK


def test_agreement_asks_for_labels_when_none_are_filled_in(tmp_path, capsys):
    save_record(make_record("full", "q1", "ssuu"), record_path(tmp_path, "full", "q1"))
    eval_cli.main(["sample", "--out", str(tmp_path)])

    assert eval_cli.main(["agreement", "--out", str(tmp_path)]) == eval_cli.EXIT_ERROR
    assert "还没有填写任何标注" in capsys.readouterr().out


@pytest.fixture
def project(tmp_path, monkeypatch):
    """一个可以运行评测的工作目录：有密钥、有评测集，外部依赖都换成假的。"""
    monkeypatch.setattr(os, "environ", {**os.environ, "LLM_API_KEY": "sk-test"})
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "questions.jsonl").write_text(QUESTION_LINE, encoding="utf-8")

    paper = make_paper("2005.11401", "RAG original paper")

    def make_llm(settings) -> FakeLLM:
        return FakeLLM(
            structured={
                SupportVerdicts: [
                    SupportVerdicts(
                        verdicts=[SupportVerdict(claim_index=0, supported=True, reason="ok")]
                    )
                ],
                AspectVerdicts: [
                    AspectVerdicts(
                        verdicts=[AspectVerdict(aspect_index=0, covered=True, reason="ok")]
                    )
                ],
            },
            texts=["RAG 结合了检索与生成 [arXiv:2005.11401]。"],
        )

    class FakeResolver:
        def resolve(self, paper_ids):
            return Resolution(found={paper.paper_id: paper})

    monkeypatch.setattr(eval_cli, "OpenAICompatibleLLM", make_llm)
    monkeypatch.setattr(eval_cli, "ArxivIdResolver", FakeResolver)
    monkeypatch.setattr(
        eval_cli,
        "build_searcher",
        lambda settings: FederatedSearcher({"arXiv": FakeSearcher([paper])}),
    )
    return tmp_path


def test_run_evaluates_and_prints_the_table(project, capsys):
    exit_code = eval_cli.main(["run", "--systems", "direct"])

    output = capsys.readouterr().out
    assert exit_code == eval_cli.EXIT_OK
    assert "评测 1 道题 × 1 个系统" in output
    assert "✓ [1/1] direct / q1：1 条论断，支撑 1，虚构 0，覆盖 1/2 个要点" in output
    assert "| 直接回答（无检索） | 1 | 1 | 0.0% | 100.0% |" in output
    assert (project / "evals" / "results" / "runs" / "direct" / "q1.json").exists()
    assert (project / "evals" / "cache" / "search.sqlite").exists()


def test_run_rejects_an_unknown_system_name(project, capsys):
    assert eval_cli.main(["run", "--systems", "direct,magic"]) == eval_cli.EXIT_ERROR
    assert "magic" in capsys.readouterr().out


def test_run_reports_a_missing_question_file_clearly(project, capsys):
    assert eval_cli.main(["run", "--questions", "nowhere.jsonl"]) == eval_cli.EXIT_ERROR
    assert "nowhere.jsonl" in capsys.readouterr().out
