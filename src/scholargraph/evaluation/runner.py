"""评测的执行：把每道题交给每个系统运行，再交给评委打分，逐次保存结果。

每次运行保存为一个文件（`runs/<系统>/<题目ID>.json`），所以评测可以随时中断：
再次运行时已经完成的会被跳过，只做缺的部分。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAIError

from scholargraph.config import Settings
from scholargraph.evaluation.dataset import EvalQuestion
from scholargraph.evaluation.judge import evaluate_report
from scholargraph.evaluation.records import RunRecord, Usage
from scholargraph.evaluation.resolver import ArxivIdResolver
from scholargraph.evaluation.systems import SYSTEMS
from scholargraph.llm import LLM, LLMOutputError
from scholargraph.search import PaperSearcher, SearchError

logger = logging.getLogger(__name__)


class EvaluationAborted(RuntimeError):
    """评测必须停下来：继续跑出来的结果不具有可比性。"""


@dataclass
class EvaluationContext:
    """评测所需的外部资源。全部从外面传入，测试时可以换成假的实现。"""

    settings: Settings
    searcher: PaperSearcher
    make_llm: Callable[[], LLM]  # 每次运行新建一个客户端，这样 token 用量按运行单独统计
    judge_llm: LLM
    resolver: ArxivIdResolver | None
    # 返回各检索源的状态；任何一个来源被停用时评测就停止（见 `_check_sources`）
    source_stats: Callable[[], dict[str, dict]] = dict
    on_progress: Callable[[str], None] = print


def run_evaluation(
    questions: list[EvalQuestion],
    system_names: list[str],
    out_dir: Path,
    context: EvaluationContext,
) -> int:
    """运行评测，返回失败（没有产出结果）的运行次数。失败的运行下次会重试。"""
    failures = 0
    total = len(questions) * len(system_names)
    done = 0

    for question in questions:
        for system_name in system_names:
            done += 1
            label = f"[{done}/{total}] {system_name} / {question.id}"
            path = record_path(out_dir, system_name, question.id)
            try:
                record = load_record(path) or _generate(question, system_name, context)
                save_record(record, path)  # 先保存生成结果：评委出错时不必重新生成
                if record.evaluation is None:
                    _evaluate(record, question, context)
                    save_record(record, path)
            except (OpenAIError, LLMOutputError, SearchError) as error:
                _check_sources(context)  # 如果是检索源出了问题，直接停止而不是继续失败下去
                failures += 1
                logger.warning("%s 失败，下次运行时会重试：%s", label, error)
                context.on_progress(f"✗ {label}：失败（{type(error).__name__}）")
                continue

            evaluation = record.evaluation
            context.on_progress(
                f"✓ {label}：{len(evaluation.claims)} 条论断，"
                f"支撑 {evaluation.count('supported')}，虚构 {evaluation.count('fabricated')}，"
                f"覆盖 {evaluation.covered_aspects}/{len(evaluation.aspects)} 个要点"
            )
    return failures


def record_path(out_dir: Path, system_name: str, question_id: str) -> Path:
    return out_dir / "runs" / system_name / f"{question_id}.json"


def load_record(path: Path) -> RunRecord | None:
    if not path.exists():
        return None
    return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))


def save_record(record: RunRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")


def load_all_records(out_dir: Path) -> list[RunRecord]:
    """读取结果目录下所有已完成评测的运行记录。"""
    records = [load_record(path) for path in sorted((out_dir / "runs").glob("*/*.json"))]
    return [record for record in records if record is not None and record.evaluation is not None]


def _generate(question: EvalQuestion, system_name: str, context: EvaluationContext) -> RunRecord:
    """让一个系统回答一道题，并记录它的耗时和 token 用量。"""
    system, _description = SYSTEMS[system_name]
    llm = context.make_llm()
    started_at = time.monotonic()
    output = system(question, llm, context.searcher, context.settings)
    elapsed = time.monotonic() - started_at

    _check_sources(context)  # 检索源中途被停用：这次的结果是在残缺的条件下得到的，不保存
    usage = llm.usage
    return RunRecord(
        question_id=question.id,
        question=question.question,
        system=system_name,
        output=output,
        usage=Usage(
            calls=usage.calls,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        ),
        elapsed_seconds=round(elapsed, 1),
    )


def _evaluate(record: RunRecord, question: EvalQuestion, context: EvaluationContext) -> None:
    """让评委给报告打分。完整系统发生过修订时，初稿也评一遍，用来量化引用校验的作用。"""
    judge = {"llm": context.judge_llm, "resolver": context.resolver}
    output = record.output
    if output.first_draft is not None:
        record.first_draft_evaluation = evaluate_report(
            output.first_draft, output.evidence, question.aspects, **judge
        )
    record.evaluation = evaluate_report(output.report, output.evidence, question.aspects, **judge)


def _check_sources(context: EvaluationContext) -> None:
    """有检索源被停用（多半是当天的免费额度用完了）时停止评测。

    否则后面的运行只能用剩下的来源，和前面的运行条件不同，结果没法放在一起比较。
    """
    disabled = {
        name: stats["disabled_reason"]
        for name, stats in context.source_stats().items()
        if stats.get("disabled_reason")
    }
    if disabled:
        details = "；".join(f"{name}（{reason}）" for name, reason in disabled.items())
        raise EvaluationAborted(
            f"检索源已停用：{details}。已完成的结果都已保存，问题解决后重新运行即可从断点继续。"
        )
