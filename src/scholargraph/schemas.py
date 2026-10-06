"""数据模型。

分成两组：
- 领域对象：在图的状态里流转（SubQuestion / Paper / Note / CitationIssue / VerificationRound）。
- LLM 输出契约：只描述"模型这一次必须返回什么形状的 JSON"
  （ResearchPlan / FindingList / Critique / SupportVerdicts / CorrectionList）。

字段的 description 会随 JSON Schema 一起发给模型，所以它既是文档也是提示词。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ───────────────────────── 领域对象 ─────────────────────────


class SubQuestion(BaseModel):
    """一个可以独立检索的子问题。"""

    question: str = Field(description="子问题，使用与用户问题相同的语言")
    search_queries: list[str] = Field(
        min_length=1,
        description=(
            "2 到 3 组用于文献检索的英文检索词。每组 2 到 4 个词；"
            "各组使用不同的术语（缩写、全称、近义的方法名各一组）；"
            "不要写成长句，不要带引号或布尔运算符"
        ),
    )


class Paper(BaseModel):
    """证据池中的一篇论文。

    `paper_id` 带有来源前缀，是论文在全系统内的唯一标识，也是正文里的引用写法：
    `arXiv:2401.12345`、`OpenAlex:W4385245566`、`S2:215416146`。
    同一篇论文被多个检索源找到时会合并成一条记录（见 `search/fusion.py`）。
    """

    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    abstract: str
    url: str
    doi: str | None = None
    venue: str | None = None  # 正式发表的期刊或会议；没有正式出处时为 None
    # 文献类型：preprint / dissertation / article / review 等（取 OpenAlex 的分类）；不知道时为 None
    work_type: str | None = None
    citation_count: int | None = None  # 被引次数；检索源不提供时为 None
    sources: list[str] = Field(default_factory=list)  # 哪些检索源返回了这篇论文
    # Researcher 读摘要后概括的主要结论。论文可能是因为某个次要结果被检索到的，
    # 把主要结论交给 Writer，才不会只引用它的细枝末节、却漏掉（甚至违背）它真正的结论
    main_conclusion: str | None = None


class Finding(BaseModel):
    """一条有出处的发现。"""

    statement: str = Field(description="一句话陈述，必须能被所引论文的摘要直接支撑")
    paper_ids: list[str] = Field(description="支撑该陈述的论文 ID，只能取自给定的论文列表")


class Note(BaseModel):
    """一个 Researcher 针对一个子问题交回的笔记。

    除了发现本身，还记录检索过程的统计，用来回答"为什么这个子问题没有结果"：
    是没检索到论文，还是检索到了但不相关，还是模型给出的发现没有出处。
    """

    sub_question: str
    search_queries: list[str]  # 实际使用的检索词
    retrieved_count: int  # 交给模型阅读的候选论文数（多源合并去重并截断之后）
    discarded_findings: int  # 因没有出处或出处无效而被丢弃的发现数
    findings: list[Finding]


class CitationIssue(BaseModel):
    """引用校验发现的一个问题。"""

    # fabricated：引用了证据池之外的 ID；unsupported：摘要不支撑论断；
    # out_of_scope：论断依据的研究对象不在调研范围内；weak_support：核心论断只由弱证据支撑
    kind: Literal["fabricated", "unsupported", "out_of_scope", "weak_support"]
    claim: str
    paper_ids: list[str]
    reason: str


class VerificationRound(BaseModel):
    """一轮引用校验的统计。第一轮的数字就是"修订之前"的引用质量，评测时会用到。"""

    checked: int  # 本轮正文中带引用的论断总数
    fabricated: int  # 引用了证据池之外的 ID
    unsupported: int  # ID 有效，但摘要不支撑论断
    unverified: int  # 模型没有给出判断，未能完成校验
    out_of_scope: int = 0  # 摘要支撑，但研究对象在调研范围之外
    weak_support: int = 0  # 核心论断（概述或概括性的句子）只由弱证据支撑
    issues: list[CitationIssue]  # 本轮发现的问题的明细（修订后正文里就看不到它们了，在这里留档）


# ─────────────────────── LLM 输出契约 ───────────────────────


class ResearchPlan(BaseModel):
    """Planner 的输出。"""

    scope: str = Field(
        description=(
            "一句话界定调研范围：研究对象是什么，哪些相邻但不同的对象不在范围内。"
            "例如问题问的是纯文本大语言模型，就写明不包括只研究视觉语言模型的工作。"
            "排除项要写成明确的列表，不要附加“除非……”之类的例外条款"
        )
    )
    sub_questions: list[SubQuestion] = Field(
        min_length=1, description="互不重叠、合起来能覆盖原问题的子问题"
    )


class PaperConclusion(BaseModel):
    """一篇论文的主要结论。"""

    paper_id: str = Field(description="论文 ID，原样照抄候选列表中的完整字符串")
    conclusion: str = Field(
        description="一句话概括摘要中作者最想说明的主要结果，不论它是否与你负责的子问题相关"
    )


class FindingList(BaseModel):
    """Researcher 的输出。字段顺序就是希望模型思考的顺序：先读懂每篇论文的主要结论，再提炼发现。"""

    paper_conclusions: list[PaperConclusion] = Field(
        default_factory=list, description="你在 findings 中引用的每一篇论文的主要结论，每篇一条"
    )
    findings: list[Finding] = Field(
        description="按重要性从高到低排列的发现；没有相关内容时返回空列表"
    )


class Critique(BaseModel):
    """Critic 的输出。字段顺序就是希望模型思考的顺序：先列应有的方面，再对照找缺口。"""

    expected_aspects: list[str] = Field(
        description="回答研究问题应当覆盖的主要方面或方法类别（凭领域常识列出，仅用于找缺口）"
    )
    is_sufficient: bool = Field(description="现有发现是否已经覆盖了上述主要方面")
    gaps: list[str] = Field(
        description=(
            "本次检索尚未覆盖的方面，相近的合并为一条；足够时为空列表。"
            "写成“本次检索未找到关于……的文献”，不要断言该领域缺乏研究"
        )
    )
    follow_ups: list[SubQuestion] = Field(
        description="为填补缺口需要补充检索的子问题；足够时为空列表"
    )


class SupportVerdict(BaseModel):
    """对一条论断的支撑性判断。"""

    claim_index: int = Field(description="论断的编号，与输入中的编号一致")
    supported: bool = Field(description="所引论文的摘要是否支撑该论断")
    in_scope: bool = Field(
        default=True,
        description="论断所依据的研究对象（人群、模型、场景）是否在调研范围之内",
    )
    reason: str = Field(description="一句话理由")


class SupportVerdicts(BaseModel):
    """引用校验的输出。"""

    verdicts: list[SupportVerdict]


class Correction(BaseModel):
    """对一个有问题的句子的修改。"""

    issue_index: int = Field(description="问题的编号，与输入中的编号一致")
    action: Literal["rewrite", "delete"] = Field(
        description="rewrite：改写成与摘要一致的表述；delete：无法改写时删除该句"
    )
    new_text: str = Field(
        description="action 为 rewrite 时给出完整的新句子（含引用和句末标点）；delete 时留空"
    )


class CorrectionList(BaseModel):
    """Reviser 的输出。"""

    corrections: list[Correction]
