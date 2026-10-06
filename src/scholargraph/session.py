"""一次交互式调研会话：驱动图运行、展示进度、在暂停处向人提问。"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from rich.console import Console
from rich.table import Table

from scholargraph.nodes import names
from scholargraph.nodes.human_review import ReviewDecision, approve, request_changes
from scholargraph.state import ResearchState

_WAITING_MESSAGE = "运行中（文献检索有限速，检索阶段可能需要几分钟）…"


class UnknownThreadError(LookupError):
    """要恢复的会话在 checkpoint 库里不存在。"""


class ResearchSession:
    """把"运行 -> 暂停等待审批 -> 恢复"的循环封装起来。

    `thread_id` 是一次调研在 checkpoint 库里的名字：
    同一个 thread_id 的多次调用会接着上次的状态继续。
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        thread_id: str,
        console: Console,
        *,
        auto_approve: bool = False,
    ) -> None:
        self._graph = graph
        self._config = {"configurable": {"thread_id": thread_id}}
        self._console = console
        self._auto_approve = auto_approve

    def start(self, question: str) -> ResearchState:
        """开始一次新的调研，返回最终状态（其中 `report` 是最终报告）。"""
        return self._run_to_completion({"question": question})

    def resume(self) -> ResearchState:
        """从 checkpoint 继续一次中断的调研，返回最终状态。"""
        if not self._graph.get_state(self._config).values: # type: ignore
            raise UnknownThreadError(self._config["configurable"]["thread_id"])
        return self._run_to_completion(None)  # 输入为 None 表示"从上次停下的地方继续"

    def _run_to_completion(self, graph_input: dict | Command | None) -> ResearchState:
        while True:
            # 在真正的终端里显示一个转动的等待提示；输出被重定向时它自动不显示。
            # spinner="line" 只用 ASCII 字符，在旧版 Windows 控制台里也能正常显示
            with self._console.status(_WAITING_MESSAGE, spinner="line"):
                for update in self._graph.stream(graph_input, self._config, stream_mode="updates"): # type: ignore
                    self._show_progress(update)

            snapshot = self._graph.get_state(self._config) # type: ignore
            if not snapshot.next:  # 没有待执行的节点：运行结束
                return snapshot.values # type: ignore

            # 图停在了 interrupt 处：取出它带出来的数据，请人审批，再把结果送回去
            decision = self._ask_for_review(snapshot.interrupts[0].value)
            graph_input = Command(resume=decision)

    def _show_progress(self, update: dict) -> None:
        """`update` 的形状是 {节点名: 该节点返回的状态更新}。"""
        for node_name, node_output in update.items():
            describe = _PROGRESS_MESSAGES.get(node_name)
            if describe is not None:
                self._console.print(f"[green]✓[/green] {describe(node_output)}")

    def _ask_for_review(self, request: dict) -> ReviewDecision:
        self._console.print(f"[bold]调研范围：[/bold]{request['scope']}")
        self._console.print(_plan_table(request["plan"]))
        if self._auto_approve:
            self._console.print("[dim]已自动通过计划（--auto-approve）[/dim]")
            return approve()

        feedback = self._console.input(
            "直接回车通过；或输入对范围、子问题的修改意见后回车："
        ).strip()
        return request_changes(feedback) if feedback else approve()


def _plan_table(plan: list[dict]) -> Table:
    table = Table(title="调研计划（待审批）", show_lines=True)
    table.add_column("#", justify="right")
    # overflow="fold"：窗口较窄时换行显示，而不是把放不下的内容截断成省略号
    table.add_column("子问题", overflow="fold", ratio=3)
    table.add_column("检索词", overflow="fold", ratio=2)
    for index, sub_question in enumerate(plan, start=1):
        table.add_row(
            str(index), sub_question["question"], "\n".join(sub_question["search_queries"])
        )
    return table


def _describe_researcher(output: dict) -> str:
    note = output["notes"][0]
    summary = (
        f"Researcher 完成「{note.sub_question}」："
        f"阅读 {note.retrieved_count} 篇候选论文，提炼出 {len(note.findings)} 条发现"
    )
    if note.discarded_findings:
        summary += f"（另有 {note.discarded_findings} 条因出处无效被丢弃）"
    return summary


def _describe_critic(output: dict) -> str:
    round_label = f"第 {output['research_rounds']} 轮检索评估"
    if output["pending"]:
        return f"Critic {round_label}：发现缺口，补充检索 {len(output['pending'])} 个子问题"
    if output["open_gaps"]:
        return f"Critic {round_label}：不再补充检索，遗留 {len(output['open_gaps'])} 个缺口"
    return f"Critic {round_label}：覆盖充分"


def _describe_verifier(output: dict) -> str:
    stats = output["verification_log"][-1]
    summary = f"引用校验：共 {stats.checked} 条带引用的论断"
    if not output["issues"] and not stats.unverified:
        return f"{summary}，全部通过"

    if output["issues"]:
        counts = [
            ("虚构引用", stats.fabricated),
            ("摘要不支撑", stats.unsupported),
            ("超出调研范围", stats.out_of_scope),
            ("核心论断仅有弱证据", stats.weak_support),
        ]
        problems = "、".join(f"{label} {n} 条" for label, n in counts if n)
        action = (
            "交给 Reviser 修订" if output["needs_revision"] else "修订预算已用尽，将在报告中注明"
        )
        summary += f"，{problems}，{action}"
    if stats.unverified:
        summary += f"，另有 {stats.unverified} 条未能完成校验（将在报告中注明）"
    return summary


_PROGRESS_MESSAGES: dict[str, Callable[[dict], str]] = {
    names.PLANNER: lambda output: f"Planner 界定范围并生成计划：{len(output['plan'])} 个子问题",
    names.RESEARCHER: _describe_researcher,
    names.CRITIC: _describe_critic,
    names.WRITER: lambda output: f"Writer 完成初稿（{len(output['draft'])} 字）",
    names.VERIFIER: _describe_verifier,
    names.REVISER: lambda output: f"Reviser 完成第 {output['revisions']} 次修订",
    names.FINALIZER: lambda output: "报告已定稿",
}
