"""评测结果的数据模型。每次运行保存为一个 JSON 文件，汇总时再读回来。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from scholargraph.schemas import Paper

Verdict = Literal["supported", "unsupported", "fabricated", "unjudged"]


class ClaimJudgement(BaseModel):
    """评委对一条带引用的论断的判断。

    - supported：所引论文真实存在，且摘要支撑论断
    - unsupported：论文存在，但摘要不支撑论断
    - fabricated：所引的论文 ID 查无此文
    - unjudged：无法判断（评委没有给出结论，或无法联网核实论文是否存在）
    """

    claim: str
    paper_ids: list[str]
    verdict: Verdict
    reason: str = ""


class AspectJudgement(BaseModel):
    """评委对"报告是否覆盖了某个参考要点"的判断。"""

    aspect: str
    covered: bool
    reason: str = ""


class ReportEvaluation(BaseModel):
    """一份报告的评测结果。"""

    claims: list[ClaimJudgement]
    aspects: list[AspectJudgement]
    characters: int  # 正文字数，不含引用标记
    cited_papers: int  # 正文引用的不同论文数
    resolved_papers: dict[str, Paper] = Field(default_factory=dict)  # 评测时联网核实到的论文

    def count(self, verdict: Verdict) -> int:
        return sum(1 for claim in self.claims if claim.verdict == verdict)

    @property
    def covered_aspects(self) -> int:
        return sum(1 for aspect in self.aspects if aspect.covered)


class Usage(BaseModel):
    """被评测系统消耗的 LLM 资源（不含评委的消耗）。"""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class SystemOutput(BaseModel):
    """一个系统对一道题的输出。"""

    report: str  # 综述正文（不含程序自动附加的参考文献等章节）
    evidence: dict[str, Paper] = Field(default_factory=dict)  # 系统检索到并可以引用的论文
    first_draft: str | None = None  # 仅完整系统且发生过修订时：引用校验之前的初稿
    details: dict = Field(default_factory=dict)  # 系统自己记录的过程信息


class RunRecord(BaseModel):
    """一次运行的完整记录：哪个系统、哪道题、输出了什么、花了多少、评得怎样。"""

    question_id: str
    question: str
    system: str
    output: SystemOutput
    usage: Usage
    elapsed_seconds: float
    evaluation: ReportEvaluation | None = None
    first_draft_evaluation: ReportEvaluation | None = None
