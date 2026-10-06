"""评测结果的汇总：把所有运行记录聚合成一张对比表。"""

from __future__ import annotations

from dataclasses import dataclass

from scholargraph.evaluation.records import ReportEvaluation, RunRecord
from scholargraph.evaluation.systems import SYSTEMS

FULL_BEFORE_REVISION = "full_before_revision"
_DESCRIPTIONS = {name: description for name, (_system, description) in SYSTEMS.items()}
_DESCRIPTIONS[FULL_BEFORE_REVISION] = "完整系统（引用校验之前的初稿）"
_ROW_ORDER = ["direct", "single_agent", "no_critic", FULL_BEFORE_REVISION, "full"]


@dataclass
class SystemSummary:
    """一个系统在整个评测集上的汇总指标。"""

    system: str
    runs: int
    claims: int  # 带引用的论断总数
    fabricated: int
    supported: int
    unjudged: int
    coverage: float  # 各题"覆盖的要点数 / 要点总数"的平均值
    cited_papers: float  # 以下四项都是每题平均
    characters: float
    tokens: float
    seconds: float

    @property
    def fabricated_rate(self) -> float:
        return _ratio(self.fabricated, self.claims)

    @property
    def support_rate(self) -> float:
        """引用支撑率：所引论文存在且摘要支撑的论断，占全部带引用论断的比例。"""
        return _ratio(self.supported, self.claims)

    @property
    def unjudged_rate(self) -> float:
        return _ratio(self.unjudged, self.claims)


def summarize(records: list[RunRecord]) -> list[SystemSummary]:
    """按系统汇总。虚构引用率和支撑率按论断汇总（论断多的题权重大），覆盖率按题平均。"""
    by_system: dict[str, list[tuple[RunRecord, ReportEvaluation]]] = {}
    for record in records:
        if record.evaluation is None:
            continue
        by_system.setdefault(record.system, []).append((record, record.evaluation))
        if record.system == "full":
            # 没有发生修订的运行，初稿就是终稿：这样"校验之前"这一行覆盖的是同一批题目
            before = record.first_draft_evaluation or record.evaluation
            by_system.setdefault(FULL_BEFORE_REVISION, []).append((record, before))

    ordered = [name for name in _ROW_ORDER if name in by_system]
    ordered += sorted(name for name in by_system if name not in _ROW_ORDER)
    return [_summarize_system(name, by_system[name]) for name in ordered]


def render_markdown(summaries: list[SystemSummary]) -> str:
    """把汇总结果渲染成 Markdown 表格。"""
    if not summaries:
        return "还没有已完成的评测结果。"

    lines = [
        "| 系统 | 题数 | 论断数 | 虚构引用率 | 引用支撑率 | 未判定率 | 要点覆盖率 | 引用论文数/题 | 字数/题 | token/题 | 耗时/题 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in summaries:
        lines.append(
            f"| {_DESCRIPTIONS.get(item.system, item.system)} | {item.runs} | {item.claims} "
            f"| {item.fabricated_rate:.1%} | {item.support_rate:.1%} | {item.unjudged_rate:.1%} "
            f"| {item.coverage:.1%} | {item.cited_papers:.1f} | {item.characters:.0f} "
            f"| {item.tokens:,.0f} | {item.seconds:.0f} 秒 |"
        )
    lines += [
        "",
        "- 虚构引用率：所引论文 ID 查无此文的论断占比。",
        "- 引用支撑率：所引论文真实存在、且摘要支撑论断的论断占比（由 LLM 评委判断）。",
        "- 未判定率：评委没有给出结论，或无法联网核实论文是否存在的论断占比。",
        "- 要点覆盖率：报告覆盖的人工参考要点占比，各题取平均。",
        "- token 和耗时只统计被评测系统本身，不含评委。"
        "“引用校验之前的初稿”与完整系统来自同一次运行，token 和耗时相同。",
    ]
    return "\n".join(lines)


def _summarize_system(
    system: str, items: list[tuple[RunRecord, ReportEvaluation]]
) -> SystemSummary:
    runs = len(items)
    evaluations = [evaluation for _record, evaluation in items]
    return SystemSummary(
        system=system,
        runs=runs,
        claims=sum(len(e.claims) for e in evaluations),
        fabricated=sum(e.count("fabricated") for e in evaluations),
        supported=sum(e.count("supported") for e in evaluations),
        unjudged=sum(e.count("unjudged") for e in evaluations),
        coverage=sum(_ratio(e.covered_aspects, len(e.aspects)) for e in evaluations) / runs,
        cited_papers=sum(e.cited_papers for e in evaluations) / runs,
        characters=sum(e.characters for e in evaluations) / runs,
        tokens=sum(record.usage.total_tokens for record, _ in items) / runs,
        seconds=sum(record.elapsed_seconds for record, _ in items) / runs,
    )


def _ratio(part: int, whole: int) -> float:
    return part / whole if whole else 0.0
