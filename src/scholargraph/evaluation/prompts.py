"""评测用的提示词：两个基线系统，以及独立的评委。

评委的提示词与被评测系统内部的校验器分开写、分开维护：
评委要对所有系统一视同仁，不应该随着系统内部的调整而变化。
"""

from __future__ import annotations

from scholargraph.citations import CitedClaim, format_citation, strip_citations
from scholargraph.schemas import Paper

_CITATION_RULES = f"""引用规则：
- 每个事实性论断都要标注出处，写法是把论文 ID 放进方括号，例如 {format_citation("arXiv:2401.00001")}，写在句末标点之前。
- 一句话引用多篇论文时连续书写。
- 绝不能编造论文 ID。"""

_WRITING_RULES = """写作要求：
- Markdown 格式：一级标题、概述、按主题组织的正文小节、局限与展望。
- 概述要直接回答研究问题；正文不超过 2500 个汉字（英文不超过 1500 词）。
- 不要写参考文献列表。
- 使用与研究问题相同的语言，直接输出报告正文。"""

# ───────────── 基线一：直接回答（没有检索） ─────────────

DIRECT_SYSTEM = f"""你是一名学术综述的作者。你没有检索工具，只能凭自己的知识撰写一份文献综述。

{_CITATION_RULES}
- 只引用你确信存在、并且记得 arXiv 编号的论文；没有把握的论断不要写。

{_WRITING_RULES}"""


def direct_prompt(question: str) -> str:
    return f"研究问题：{question}"


# ───────────── 基线二：单智能体（边检索边决定下一步） ─────────────

AGENT_STEP_SYSTEM = """你是一名独自完成文献调研的研究员，可以反复使用文献检索工具。
你会看到研究问题和目前为止的检索记录，任务是决定下一步：继续检索，还是已经可以动笔。

要求：
- 文献库是关键词匹配：检索词用 2 到 4 个英文词，使用论文标题里常见的术语。
- 每次检索换一个角度，不要重复用过的检索词。
- 已有的论文足以覆盖研究问题的主要方面时，就结束检索。"""


def agent_step_prompt(
    question: str, search_log: list[tuple[str, list[Paper]]], remaining: int
) -> str:
    if search_log:
        history = "\n\n".join(
            f'检索词 "{query}" 的结果：\n'
            + ("\n".join(f"- {paper.title}" for paper in papers) or "- （没有结果）")
            for query, papers in search_log
        )
    else:
        history = "（还没有检索过）"
    return f"研究问题：{question}\n\n检索记录：\n{history}\n\n还可以检索 {remaining} 次。"


AGENT_WRITER_SYSTEM = f"""你是一名学术综述的作者。你会拿到一个研究问题和检索到的论文摘要，任务是写一份文献综述。

{_CITATION_RULES}
- 只能引用下面给出的论文，ID 必须原样照抄"ID:"后面的完整字符串（包括前缀）。
- 只写摘要里有依据的内容；与研究问题无关的论文直接忽略。

{_WRITING_RULES}"""


def agent_writer_prompt(question: str, papers: list[Paper]) -> str:
    return f"研究问题：{question}\n\n检索到的论文：\n{_render_papers(papers)}"


# ───────────── 评委：引用是否被摘要支撑 ─────────────

SUPPORT_JUDGE_SYSTEM = """你是一名严格的引用核查员，负责评测文献综述的引用质量。
你会拿到若干条带引用的论断，以及被引用论文的摘要，任务是逐条判断论断是否被所引摘要支撑。

判断标准：
- 支撑：摘要明确包含该论断，或可以由摘要直接推出。引用多篇论文时，它们合起来能支撑即可。
- 不支撑：摘要没有提到、与摘要矛盾，或论断比摘要说得更绝对、更夸大、适用范围更广。
- 只依据给出的摘要判断，不要使用你自己的背景知识。
- 每一条论断都必须给出判断，编号与输入一致。"""


def support_judge_prompt(claims: list[CitedClaim], papers: list[Paper]) -> str:
    numbered = "\n".join(
        f"[{index}] {claim.text}（引用：{', '.join(claim.paper_ids)}）"
        for index, claim in enumerate(claims)
    )
    return f"待核查的论断：\n{numbered}\n\n被引用的论文：\n{_render_papers(papers)}"


# ───────────── 评委：报告是否覆盖了参考要点 ─────────────

COVERAGE_JUDGE_SYSTEM = """你是一名文献综述的评审人，负责评测综述的内容覆盖情况。
你会拿到一份综述和若干参考要点，任务是逐个判断综述是否覆盖了该要点。

判断标准：
- 覆盖：综述用至少一两句话实质性地讨论了这个要点（不要求使用相同的措辞或举相同的例子）。
- 未覆盖：完全没有提到，或只是一笔带过的名词罗列。
- 只看综述写了什么，不评价写得对不对。
- 每个要点都必须给出判断，编号与输入一致。"""


def coverage_judge_prompt(report: str, aspects: list[str]) -> str:
    numbered = "\n".join(f"[{index}] {aspect}" for index, aspect in enumerate(aspects))
    return f"参考要点：\n{numbered}\n\n<综述>\n{strip_citations(report)}\n</综述>"


def _render_papers(papers: list[Paper]) -> str:
    blocks = [
        f"- ID: {paper.paper_id} | {paper.title}\n  摘要：{paper.abstract}" for paper in papers
    ]
    return "\n".join(blocks) or "（无）"
