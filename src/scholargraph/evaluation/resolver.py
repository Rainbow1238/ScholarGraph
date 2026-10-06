"""联网核实论文是否存在：按 arXiv 编号查询 arXiv。

"直接回答"基线凭记忆引用论文，评测时必须核实这些编号是不是真的。
对所有系统都用同一条规则：引用的 ID 不在系统自己的检索结果里时，一律来这里核实。
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

import arxiv
import requests

from scholargraph.schemas import Paper
from scholargraph.search.arxiv_search import ARXIV_MIN_INTERVAL_SECONDS, enforce_timeout, to_paper

_ARXIV_PREFIX = "arXiv:"
# 只查询格式合法的编号：格式不对的编号会让 arXiv 整个请求报错，而它们本来也不可能存在
_VALID_ARXIV_NUMBER = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})$")
_BATCH_SIZE = 40


@dataclass
class Resolution:
    """核实的结果。没有出现在这两个集合里的 ID，就是查无此文。"""

    found: dict[str, Paper] = field(default_factory=dict)
    unreachable: set[str] = field(default_factory=set)  # 因为网络问题没能核实，不能算作虚构


class ArxivIdResolver:
    """按编号向 arXiv 查询论文。查过的结果记在内存里，同一个编号不会查两次。"""

    def __init__(self, min_interval_seconds: float = ARXIV_MIN_INTERVAL_SECONDS) -> None:
        self._client = arxiv.Client(
            page_size=_BATCH_SIZE, delay_seconds=min_interval_seconds, num_retries=3
        )
        enforce_timeout(self._client)
        self._lock = threading.Lock()
        self._known: dict[str, Paper | None] = {}  # 论文 ID -> 论文；None 表示已确认不存在

    def resolve(self, paper_ids: list[str]) -> Resolution:
        resolution = Resolution()
        with self._lock:
            to_look_up = []
            for paper_id in dict.fromkeys(paper_ids):
                if not self._is_valid_arxiv_id(paper_id):
                    self._known[paper_id] = None
                elif paper_id not in self._known:
                    to_look_up.append(paper_id)

            for start in range(0, len(to_look_up), _BATCH_SIZE):
                batch = to_look_up[start : start + _BATCH_SIZE]
                try:
                    found = self._fetch(batch)
                except (arxiv.ArxivError, requests.RequestException):
                    resolution.unreachable.update(batch)
                else:
                    for paper_id in batch:
                        self._known[paper_id] = found.get(paper_id)

            for paper_id in paper_ids:
                paper = self._known.get(paper_id)
                if paper is not None:
                    resolution.found[paper_id] = paper
        return resolution

    def _fetch(self, paper_ids: list[str]) -> dict[str, Paper]:
        numbers = [paper_id.removeprefix(_ARXIV_PREFIX) for paper_id in paper_ids]
        results = self._client.results(arxiv.Search(id_list=numbers, max_results=len(numbers)))
        papers = [to_paper(result) for result in results]
        return {paper.paper_id: paper for paper in papers}

    @staticmethod
    def _is_valid_arxiv_id(paper_id: str) -> bool:
        if not paper_id.startswith(_ARXIV_PREFIX):
            return False  # 其他来源的 ID 无法在这里核实；系统检索结果之外的这类 ID 视为不存在
        return _VALID_ARXIV_NUMBER.match(paper_id.removeprefix(_ARXIV_PREFIX)) is not None
