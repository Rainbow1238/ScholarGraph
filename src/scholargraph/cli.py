"""命令行入口：解析参数、装配各个组件、保存结果。

用法：
    python -m scholargraph "你的研究问题"
    python -m scholargraph --resume 20261005-213000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAIError
from rich.console import Console

from scholargraph.checkpoint import open_checkpointer
from scholargraph.config import ConfigError, Settings
from scholargraph.graph import build_graph
from scholargraph.llm import LLMOutputError, OpenAICompatibleLLM, TokenUsage
from scholargraph.metrics import build_run_metrics
from scholargraph.search import SearchError, build_searcher
from scholargraph.session import ResearchSession, UnknownThreadError

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130  # 被 Ctrl+C 中断时的惯例退出码


def main(argv: list[str] | None = None) -> int:
    _use_utf8_streams()
    args = _parse_args(argv)
    console = Console()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    try:
        settings = Settings.from_env()
        searcher = build_searcher(settings)
    except ConfigError as error:
        console.print(f"[red]配置错误：{error}[/red]")
        return EXIT_ERROR

    thread_id = args.resume or datetime.now().strftime("%Y%m%d-%H%M%S")
    llm = OpenAICompatibleLLM(settings)
    console.print(
        f"[bold]会话 ID：{thread_id}[/bold]"
        f"（模型：{settings.model}；检索源：{'、'.join(settings.search_sources)}）"
    )

    started_at = time.monotonic()
    try:
        with open_checkpointer(settings.checkpoint_path) as checkpointer:
            graph = build_graph(
                llm=llm, searcher=searcher, settings=settings, checkpointer=checkpointer
            )
            session = ResearchSession(graph, thread_id, console, auto_approve=args.auto_approve)
            final_state = session.resume() if args.resume else session.start(args.question)
    except UnknownThreadError:
        console.print(f"[red]找不到会话 {thread_id}，无法恢复。[/red]")
        return EXIT_ERROR
    except (OpenAIError, LLMOutputError, SearchError) as error:
        console.print(f"[red]运行失败：{error}[/red]")
        _print_resume_hint(console, thread_id)
        return EXIT_ERROR
    except KeyboardInterrupt:
        console.print("\n[yellow]已中断。[/yellow]")
        _print_resume_hint(console, thread_id)
        return EXIT_INTERRUPTED

    source_stats = searcher.stats()
    metrics = build_run_metrics(
        final_state,
        model=settings.model,
        usage=llm.usage,
        source_stats=source_stats,
        elapsed_seconds=time.monotonic() - started_at,
    )
    report_path, metrics_path = _save_outputs(
        final_state["report"], metrics, settings.output_dir, thread_id # pyright: ignore[reportTypedDictNotRequiredAccess]
    )

    console.print(f"\n[bold green]报告已保存：{report_path}[/bold green]")
    console.print(f"运行指标已保存：{metrics_path}")
    _print_source_stats(console, source_stats)
    _print_unused_source_hint(console, source_stats, metrics["report"]["cited_papers_by_source"])
    _print_usage(console, llm.usage, metrics["elapsed_seconds"])
    return EXIT_OK


def _use_utf8_streams() -> None:
    """让标准输出和标准错误使用 UTF-8。

    Windows 下输出被重定向或由其他程序捕获时，Python 默认使用系统的 GBK 编码，
    遇到 GBK 里没有的字符（例如进度前的 ✓）会直接抛出 UnicodeEncodeError。
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace") # type: ignore


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scholargraph",
        description="多智能体学术文献调研：输入研究问题，输出带引用的综述报告。",
    )
    parser.add_argument("question", nargs="?", help="研究问题")
    parser.add_argument("--resume", metavar="会话ID", help="从 checkpoint 继续一次中断的调研")
    parser.add_argument("--auto-approve", action="store_true", help="跳过人工审批，自动通过计划")
    args = parser.parse_args(argv)

    if bool(args.question) == bool(args.resume):
        parser.error("请提供研究问题，或用 --resume 指定要恢复的会话（二选一）。")
    return args


def _save_outputs(
    report: str, metrics: dict, output_dir: Path, thread_id: str
) -> tuple[Path, Path]:
    """保存报告（Markdown）和运行指标（JSON），返回两个文件的路径。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{thread_id}.md"
    metrics_path = output_dir / f"{thread_id}.metrics.json"
    report_path.write_text(report, encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path, metrics_path


def _print_resume_hint(console: Console, thread_id: str) -> None:
    console.print(
        f"已完成的步骤保存在 checkpoint 中，可用以下命令继续：\n"
        f"  python -m scholargraph --resume {thread_id}"
    )


def _print_source_stats(console: Console, source_stats: dict[str, dict]) -> None:
    """逐个检索源报告本次运行的情况；被停用的来源用醒目的颜色提示。"""
    for name, stats in source_stats.items():
        line = f"检索源 {name}：请求 {stats['requests']} 次，失败 {stats['failures']} 次，返回 {stats['papers']} 篇"
        if stats["disabled_reason"]:
            console.print(
                f"[yellow]{line}；因连续失败已停用（{stats['disabled_reason']}）[/yellow]"
            )
        else:
            console.print(f"[dim]{line}[/dim]")


def _print_unused_source_hint(
    console: Console, source_stats: dict[str, dict], cited_by_source: dict[str, int]
) -> None:
    """某个来源检索到了论文、却没有一篇被引用：多半是它不收录这个领域，提示可以关掉它省时间。"""
    for name, stats in source_stats.items():
        if stats["papers"] and not cited_by_source.get(name) and len(source_stats) > 1:
            console.print(
                f"[yellow]提示：检索源 {name} 返回了 {stats['papers']} 篇论文，但最终一篇也没有被引用，"
                f"可能是它很少收录这个领域的文献。同类题目可以在 .env 的 SEARCH_SOURCES 里去掉它，"
                f"节省检索时间。[/yellow]"
            )


def _print_usage(console: Console, usage: TokenUsage, elapsed_seconds: float) -> None:
    console.print(
        f"[dim]LLM 调用 {usage.calls} 次，"
        f"输入 {usage.prompt_tokens:,} tokens，输出 {usage.completion_tokens:,} tokens；"
        f"耗时 {elapsed_seconds:.0f} 秒[/dim]"
    )
