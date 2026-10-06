"""论文的发表状态与证据强度。

全部是纯函数：只看论文的元数据（出处、类型、DOI、被引次数），不调用模型、不访问网络。
这些判断会展示给 Researcher 和 Writer，也用于候选论文的排序、核心论断的确定性检查和来源构成统计。

"弱证据"是一个保守的信号，不是对论文质量的判决：它只表示这篇论文缺少常见的可信度背书
（没有被引用过、是学位论文、DOI 来自任何人都能上传的自存档平台），
因此不应单独支撑综述的核心结论。
"""

from __future__ import annotations

import re
from datetime import date

from scholargraph.schemas import Paper

# 自存档平台签发的 DOI：任何人都可以上传，DOI 本身不代表经过同行评审
SELF_ARCHIVE_DOI_PREFIXES = {
    "10.5281": "Zenodo",
    "10.6084": "figshare",
    "10.17605": "OSF",
    "10.31219": "OSF Preprints",
    "10.21203": "Research Square",
    "10.20944": "Preprints.org",
}
_DOI_PREFIX = re.compile(r"^(10\.\d{4,9})/")


def publication_label(paper: Paper) -> str:
    """一篇论文"发表在哪里"的简短说明：正式出处的名称，或者它属于哪类未正式发表的文献。"""
    if paper.venue:
        return paper.venue
    if paper.work_type == "dissertation":
        return "学位论文"
    if paper.work_type == "preprint":
        return "预印本"
    return "出处不详"


def weakness_reasons(paper: Paper, *, current_year: int | None = None) -> list[str]:
    """这篇论文缺少哪些可信度背书；空列表表示没有发现问题。

    被引 0 次只对发表满一年的论文算问题：当年发表的论文还来不及被引用。
    检索源不提供被引次数（例如只被 arXiv 找到的论文）时不做判断。
    """
    this_year = current_year or date.today().year
    reasons = []
    if paper.work_type == "dissertation":
        reasons.append("学位论文")
    if platform := self_archive_platform(paper.doi):
        reasons.append(f"DOI 来自自存档平台 {platform}")
    is_new = paper.year is not None and paper.year >= this_year
    if paper.citation_count == 0 and not is_new:
        reasons.append("被引 0 次")
    return reasons


def is_weak_evidence(paper: Paper, *, current_year: int | None = None) -> bool:
    return bool(weakness_reasons(paper, current_year=current_year))


def self_archive_platform(doi: str | None) -> str | None:
    match = _DOI_PREFIX.match(doi or "")
    return SELF_ARCHIVE_DOI_PREFIXES.get(match.group(1)) if match else None


def demote_weak_evidence(papers: list[Paper], positions: int) -> list[Paper]:
    """把弱证据论文在排序里后移 `positions` 位（稳定排序，其余论文的相对顺序不变）。

    只是降权而不是剔除：某个方向只有弱证据时，它们仍然会排进候选列表。
    """

    def sort_key(indexed: tuple[int, Paper]) -> float:
        index, paper = indexed
        # 加 0.5：排在原本位于目标位置的那篇论文之后，恰好后移 `positions` 位
        return index + positions + 0.5 if is_weak_evidence(paper) else index

    return [paper for _, paper in sorted(enumerate(papers), key=sort_key)]
