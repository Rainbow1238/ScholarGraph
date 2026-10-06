"""命令行入口的测试。"""

from __future__ import annotations

import io
import json
import os
import sys

from fakes import FakeSearcher
from test_graph import PAPER_A, PAPER_B, QUESTION, make_llm

from scholargraph import cli
from scholargraph.search import FederatedSearcher


def test_output_streams_are_switched_to_utf8(monkeypatch):
    """模拟 Windows 下输出被重定向的情形：流的默认编码是 GBK，无法表示 ✓。"""
    raw_output = io.BytesIO()
    gbk_stream = io.TextIOWrapper(raw_output, encoding="gbk")
    monkeypatch.setattr(sys, "stdout", gbk_stream)
    monkeypatch.setattr(sys, "stderr", gbk_stream)

    cli._use_utf8_streams()
    print("✓ 完成", file=sys.stdout, flush=True)

    assert raw_output.getvalue().decode("utf-8").strip() == "✓ 完成"


def test_missing_configuration_exits_with_an_error_instead_of_a_traceback(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if k != "LLM_API_KEY"})
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["一个研究问题"])

    assert exit_code == cli.EXIT_ERROR
    assert "LLM_API_KEY" in capsys.readouterr().out


def test_full_run_writes_the_report_and_the_metrics_file(tmp_path, monkeypatch, capsys):
    """用假模型和假检索源把 main() 从头跑到尾：检查装配是否正确、两个输出文件是否写出。"""
    monkeypatch.setattr(os, "environ", {**os.environ, "LLM_API_KEY": "sk-test"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "OpenAICompatibleLLM", lambda settings: make_llm())
    monkeypatch.setattr(
        cli,
        "build_searcher",
        lambda settings: FederatedSearcher(
            {"arXiv": FakeSearcher([PAPER_A]), "OpenAlex": FakeSearcher([PAPER_B])}
        ),
    )

    exit_code = cli.main([QUESTION, "--auto-approve"])

    assert exit_code == cli.EXIT_OK
    [report_path] = (tmp_path / "outputs").glob("*.md")
    [metrics_path] = (tmp_path / "outputs").glob("*.metrics.json")
    assert "## 文献来源构成" in report_path.read_text(encoding="utf-8")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["question"] == QUESTION
    assert set(metrics["search_sources"]) == {"arXiv", "OpenAlex"}
    output = capsys.readouterr().out
    assert "检索源 arXiv：请求 4 次，失败 0 次" in output
    assert (tmp_path / "checkpoints" / "scholargraph-v3.sqlite").exists()


def test_unknown_search_source_is_a_configuration_error(tmp_path, monkeypatch, capsys):
    environ = {**os.environ, "LLM_API_KEY": "sk-test", "SEARCH_SOURCES": "arxiv,google"}
    monkeypatch.setattr(os, "environ", environ)
    monkeypatch.chdir(tmp_path)

    assert cli.main([QUESTION]) == cli.EXIT_ERROR
    assert "google" in capsys.readouterr().out
