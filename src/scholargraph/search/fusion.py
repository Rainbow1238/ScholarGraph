"""多路检索结果的融合：识别同一篇论文并合并，再用 RRF 排序。

同一篇论文可能被不同的检索源、不同的检索词各返回一次，
而且各源给出的信息互补：arXiv 有最新的摘要，OpenAlex 有被引次数和发表出处。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scholargraph.schemas import Paper

RRF_K = 60  # RRF 的平滑常数，原论文的推荐值
_ID_PRIORITY = ("arXiv:", "OpenAlex:", "S2:")  # 合并时优先保留哪种 ID
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_MIN_TITLE_KEY_LENGTH = 15  # 太短的标题容易撞车，不用来判断是否同一篇


@dataclass
class _Entry:
    paper: Paper
    score: float
    first_seen: int


def fuse(ranked_lists: list[list[Paper]], limit: int | None = None) -> list[Paper]:
    """把多个按相关度排好序的列表融合成一个列表。

    排序使用倒数排名融合（Reciprocal Rank Fusion）：一篇论文在某个列表中排第 r 名，
    就得到 1 / (RRF_K + r) 分，各列表的得分相加。被多个来源、多个检索词同时排在前面的论文
    得分最高。RRF 只用名次不用原始分数，所以不同检索源之间不需要做分数归一化。
    得分相同时，被引次数多的排在前面。
    """
    entries: list[_Entry] = []
    entry_by_key: dict[str, _Entry] = {}

    for papers in ranked_lists:
        for rank, paper in enumerate(papers, start=1):
            entry = _find_entry(paper, entry_by_key)
            if entry is None:
                entry = _Entry(paper=paper, score=0.0, first_seen=len(entries))
                entries.append(entry)
            else:
                entry.paper = merge_papers(entry.paper, paper)
            entry.score += 1.0 / (RRF_K + rank)
            for key in _identity_keys(entry.paper):
                entry_by_key.setdefault(key, entry)

    entries.sort(key=lambda e: (-e.score, -(e.paper.citation_count or 0), e.first_seen))
    fused = [entry.paper for entry in entries]
    return fused if limit is None else fused[:limit]


def merge_papers(first: Paper, second: Paper) -> Paper:
    """把同一篇论文的两条记录合并成一条，各字段取信息更完整的一方。"""
    primary, secondary = sorted((first, second), key=_id_rank)  # 稳定排序：同级时保持先后顺序
    known_years = [year for year in (primary.year, secondary.year) if year is not None]
    known_counts = [
        count for count in (primary.citation_count, secondary.citation_count) if count is not None
    ]
    return Paper(
        paper_id=primary.paper_id,
        title=primary.title,
        authors=primary.authors or secondary.authors,
        year=min(known_years) if known_years else None,  # 取最早公开的年份
        abstract=max(primary.abstract, secondary.abstract, key=len),
        url=primary.url,
        doi=primary.doi or secondary.doi,
        venue=primary.venue or secondary.venue,
        work_type=_more_formal_type(primary.work_type, secondary.work_type),
        citation_count=max(known_counts) if known_counts else None,
        sources=list(dict.fromkeys(first.sources + second.sources)),
        main_conclusion=primary.main_conclusion or secondary.main_conclusion,
    )


def _more_formal_type(first: str | None, second: str | None) -> str | None:
    """arXiv 只知道"这是预印本"；另一来源给出了更具体的类型（例如期刊论文）时以后者为准。"""
    known = [work_type for work_type in (first, second) if work_type]
    formal = [work_type for work_type in known if work_type != "preprint"]
    return (formal or known or [None])[0]


def title_key(title: str) -> str:
    """标题的比较键：只保留小写字母和数字，忽略大小写、标点和空白的差异。"""
    return _NON_ALPHANUMERIC.sub("", title.lower())


def _identity_keys(paper: Paper) -> list[str]:
    """能够认定"是同一篇论文"的键，按可靠程度从高到低：ID、DOI、标题。"""
    keys = [f"id:{paper.paper_id}"]
    if paper.doi:
        keys.append(f"doi:{paper.doi.lower()}")
    normalized_title = title_key(paper.title)
    if len(normalized_title) >= _MIN_TITLE_KEY_LENGTH:
        keys.append(f"title:{normalized_title}")
    return keys


def _find_entry(paper: Paper, entry_by_key: dict[str, _Entry]) -> _Entry | None:
    for key in _identity_keys(paper):
        if key in entry_by_key:
            return entry_by_key[key]
    return None


def _id_rank(paper: Paper) -> int:
    for rank, prefix in enumerate(_ID_PRIORITY):
        if paper.paper_id.startswith(prefix):
            return rank
    return len(_ID_PRIORITY)
