"""基于 Semantic Scholar Academic Graph API 的检索器（可选）。

不带密钥时与所有匿名用户共享额度，容易被限流（HTTP 429）；
有密钥时每秒 1 次请求。默认不启用，在 `.env` 的 SEARCH_SOURCES 里加上 semantic_scholar 即可。
"""

from __future__ import annotations

import requests

from scholargraph.citations import normalize_paper_id
from scholargraph.schemas import Paper
from scholargraph.search.base import SearchError
from scholargraph.search.http import MinIntervalLimiter, get_json, int_or_none

SOURCE_NAME = "Semantic Scholar"
SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
PAGE_SIZE = 25
REQUESTED_FIELDS = "title,abstract,year,authors,venue,citationCount,externalIds,url"
_MAX_AUTHORS = 20


class SemanticScholarSearcher:
    def __init__(self, api_key: str | None = None, min_interval_seconds: float = 1.1) -> None:
        self._headers = {"x-api-key": api_key} if api_key else {}
        self._session = requests.Session()
        self._limiter = MinIntervalLimiter(min_interval_seconds)

    def search(self, query: str, max_results: int) -> list[Paper]:
        params: dict[str, str | int] = {
            "query": query,
            "limit": PAGE_SIZE,
            "fields": REQUESTED_FIELDS,
        }
        payload = get_json(
            self._session, SEARCH_URL, params=params, headers=self._headers, limiter=self._limiter
        )
        records = payload.get("data", [])  # 没有结果时可能不带 data 字段
        if not isinstance(records, list):
            raise SearchError("Semantic Scholar 的响应里 data 不是列表")

        papers = (to_paper(record) for record in records if isinstance(record, dict))
        return [paper for paper in papers if paper is not None][:max_results]


def to_paper(record: dict) -> Paper | None:
    """把 Semantic Scholar 的一条记录转换成 `Paper`；没有标题或摘要的记录返回 None。"""
    title, abstract = record.get("title"), record.get("abstract")
    external_ids = record.get("externalIds") if isinstance(record.get("externalIds"), dict) else {}
    assert external_ids is not None
    arxiv_id, corpus_id = external_ids.get("ArXiv"), external_ids.get("CorpusId")
    if not title or not abstract or not (arxiv_id or corpus_id):
        return None

    doi = str(external_ids["DOI"]) if external_ids.get("DOI") else None  # type: ignore
    if arxiv_id:
        paper_id = normalize_paper_id(f"arXiv:{arxiv_id}")
        url = f"https://arxiv.org/abs/{paper_id.split(':', 1)[1]}"
    else:
        paper_id = normalize_paper_id(f"S2:{corpus_id}")
        url = str(record.get("url") or f"https://api.semanticscholar.org/CorpusID:{corpus_id}")

    return Paper(
        paper_id=paper_id,
        title=" ".join(str(title).split()),
        authors=_author_names(record),
        year=int_or_none(record.get("year")),
        abstract=" ".join(str(abstract).split()),
        url=url,
        doi=doi,
        venue=_venue(record),
        work_type="preprint" if arxiv_id and not _venue(record) else None,
        citation_count=int_or_none(record.get("citationCount")),
        sources=[SOURCE_NAME],
    )


def _venue(record: dict) -> str | None:
    venue = str(record.get("venue") or "").strip()
    return venue if venue and "arxiv" not in venue.lower() else None


def _author_names(record: dict) -> list[str]:
    authors = record.get("authors") if isinstance(record.get("authors"), list) else []
    names = [str(a["name"]) for a in authors if isinstance(a, dict) and a.get("name")]  # type: ignore
    return names[:_MAX_AUTHORS]
