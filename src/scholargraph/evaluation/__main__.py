"""评测的命令行入口，见包的说明（`scholargraph/evaluation/__init__.py`）。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

from rich.console import Console

from scholargraph.cli import use_utf8_streams
from scholargraph.config import ConfigError, Settings
from scholargraph.evaluation.agreement import compute_agreement, export_sample
from scholargraph.evaluation.dataset import load_questions
from scholargraph.evaluation.resolver import ArxivIdResolver
from scholargraph.evaluation.runner import (
    EvaluationAborted,
    EvaluationContext,
    load_all_records,
    run_evaluation,
)
from scholargraph.evaluation.summary import render_markdown, summarize
from scholargraph.evaluation.systems import SYSTEMS
from scholargraph.llm import OpenAICompatibleLLM
from scholargraph.search import build_searcher
from scholargraph.search.cache import CachedSearcher

EXIT_OK = 0
EXIT_ERROR = 1
DEFAULT_QUESTIONS = Path("evals/questions.jsonl")
DEFAULT_OUT_DIR = Path("evals/results")
SEARCH_CACHE_PATH = Path("evals/cache/search.sqlite")


def main(argv: list[str] | None = None) -> int:
    use_utf8_streams()
    args = _parse_args(argv)
    console = Console()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    def say(text: str) -> None:
        # markup=False：进度里的 [1/80] 这类方括号不是 rich 的样式标记，原样输出。
        # soft_wrap=True：不在窗口宽度处强行折行，否则 Markdown 表格的一行会被拆成几行
        console.print(text, markup=False, highlight=False, soft_wrap=True)

    try:
        return _COMMANDS[args.command](args, say)
    except (ConfigError, ValueError, FileNotFoundError) as error:
        say(f"错误：{error}")
        return EXIT_ERROR
    except EvaluationAborted as error:
        say(f"评测已停止：{error}")
        return EXIT_ERROR
    except KeyboardInterrupt:
        say("\n已中断。已完成的结果都已保存，重新运行即可从断点继续。")
        return EXIT_ERROR


def _run(args: argparse.Namespace, say) -> int:
    settings = Settings.from_env()
    questions = load_questions(args.questions)[: args.limit]
    system_names = _parse_system_names(args.systems)

    federated = build_searcher(settings)
    searcher = CachedSearcher(
        federated, SEARCH_CACHE_PATH, namespace=",".join(settings.search_sources)
    )
    # 评委可以换一个模型（EVAL_JUDGE_MODEL），温度固定为 0 让判断尽量稳定
    judge_model = os.getenv("EVAL_JUDGE_MODEL", "").strip() or settings.model
    judge_settings = replace(settings, model=judge_model, temperature=0.0)
    context = EvaluationContext(
        settings=settings,
        searcher=searcher,
        make_llm=lambda: OpenAICompatibleLLM(settings),
        judge_llm=OpenAICompatibleLLM(judge_settings),
        resolver=ArxivIdResolver(),
        source_stats=federated.stats,
        on_progress=say,
    )

    say(
        f"评测 {len(questions)} 道题 × {len(system_names)} 个系统"
        f"（模型：{settings.model}；评委：{judge_model}；检索源：{'、'.join(settings.search_sources)}）"
    )
    try:
        failures = run_evaluation(questions, system_names, args.out, context)
    finally:
        searcher.close()

    if failures:
        say(f"\n有 {failures} 次运行失败，重新执行同一条命令会重试它们。")
    return _summary(args, say)


def _summary(args: argparse.Namespace, say) -> int:
    table = render_markdown(summarize(load_all_records(args.out)))
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.md"
    summary_path.write_text(table + "\n", encoding="utf-8")
    say(f"\n{table}\n\n对比表已保存：{summary_path}")
    return EXIT_OK


def _sample(args: argparse.Namespace, say) -> int:
    sheet_path, key_path = _labeling_paths(args.out)
    if sheet_path.exists() and not args.force:
        raise ValueError(
            f"{sheet_path} 已存在（里面可能有你已经填好的标注）。确认要覆盖请加 --force。"
        )
    count = export_sample(load_all_records(args.out), args.n, args.seed, sheet_path, key_path)
    say(
        f"已抽取 {count} 条论断：{sheet_path}\n"
        "用 Excel 打开，在最后一列逐条填写 1（摘要支撑论断）或 0（不支撑），保存后运行：\n"
        "  python -m scholargraph.evaluation agreement"
    )
    return EXIT_OK


def _agreement(args: argparse.Namespace, say) -> int:
    sheet_path, key_path = _labeling_paths(args.out)
    result = compute_agreement(sheet_path, key_path)
    if not result.labeled:
        raise ValueError(f"{sheet_path} 里还没有填写任何标注。")
    say(
        f"已标注 {result.labeled} 条\n"
        f"评委与人工一致：{result.raw_agreement:.1%}（Cohen's kappa = {result.cohens_kappa:.2f}）\n"
        f"  都认为支撑：{result.both_supported}；都认为不支撑：{result.both_unsupported}\n"
        f"  评委认为支撑、人工认为不支撑：{result.only_judge_supported}\n"
        f"  评委认为不支撑、人工认为支撑：{result.only_human_supported}"
    )
    return EXIT_OK


_COMMANDS = {"run": _run, "summary": _summary, "sample": _sample, "agreement": _agreement}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scholargraph.evaluation", description="评测不同系统配置生成的综述。"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="运行评测（可中断，再次运行会续跑）")
    run.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="评测集文件")
    run.add_argument("--systems", default=",".join(SYSTEMS), help="要评测的系统，逗号分隔")
    run.add_argument("--limit", type=int, default=None, help="只评测前 N 道题（试跑时用）")

    commands.add_parser("summary", help="汇总已有结果，生成对比表")

    sample = commands.add_parser("sample", help="抽样导出论断，供人工标注")
    sample.add_argument("--n", type=int, default=30, help="抽取的条数")
    sample.add_argument("--seed", type=int, default=0, help="随机种子")
    sample.add_argument("--force", action="store_true", help="覆盖已有的标注表")

    commands.add_parser("agreement", help="计算评委与人工标注的一致性")

    for command in commands.choices.values():
        command.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="结果目录")
    return parser.parse_args(argv)


def _parse_system_names(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in names if name not in SYSTEMS]
    if unknown or not names:
        raise ValueError(
            f"不认识的系统：{', '.join(unknown) or '（空）'}。可选：{', '.join(SYSTEMS)}"
        )
    return names


def _labeling_paths(out_dir: Path) -> tuple[Path, Path]:
    return out_dir / "labeling.csv", out_dir / "labeling_key.json"


if __name__ == "__main__":
    sys.exit(main())
