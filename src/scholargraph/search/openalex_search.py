"""基于 OpenAlex API 的检索器。

OpenAlex 收录期刊、会议论文和预印本，并提供被引次数和发表出处，
用来弥补 arXiv 的两个不足：只有预印本，以及排序里没有影响力信号。

不填密钥也能用，但每天的免费额度较小；在 openalex.org 免费注册一个密钥后额度更高。
"""

from __future__ import annotations

import re

import requests

from scholargraph.citations import normalize_paper_id
from scholargraph.schemas import Paper
from scholargraph.search.base import SearchError
from scholargraph.search.http import MinIntervalLimiter, get_json, int_or_none

SOURCE_NAME = "OpenAlex"
WORKS_URL = "https://api.openalex.org/works"
PAGE_SIZE = 25  # 一次请求的费用与返回条数无关，多取一些以便跳过没有摘要的记录
# 只请求用得到的字段，减小响应体积
SELECTED_FIELDS = (
    "id,doi,display_name,publication_year,cited_by_count,type,is_retracted,"
    "authorships,abstract_inverted_index,primary_location,locations"
)
_SKIPPED_TYPES = frozenset({"paratext", "dataset", "erratum", "retraction"})
_MAX_AUTHORS = 20
_ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(.+)$", re.IGNORECASE)
_ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/([^?#\s]+?)(?:\.pdf)?$", re.IGNORECASE)


class OpenAlexSearcher:
    def __init__(self, api_key: str | None = None, min_interval_seconds: float = 0.2) -> None:
        # 密钥放在请求头里而不是 URL 参数里：出错信息会带上 URL，这样密钥不会出现在日志中
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._session = requests.Session()
        self._limiter = MinIntervalLimiter(min_interval_seconds)

    def search(self, query: str, max_results: int) -> list[Paper]:
        params: dict[str, str | int] = {
            "search": query,
            "per_page": PAGE_SIZE,
            "select": SELECTED_FIELDS,
        }
        payload = get_json(
            self._session, WORKS_URL, params=params, headers=self._headers, limiter=self._limiter
        )
        works = payload.get("results")
        if not isinstance(works, list):
            raise SearchError("OpenAlex 的响应里没有 results 列表")

        papers = (to_paper(work) for work in works if isinstance(work, dict))
        return [paper for paper in papers if paper is not None][:max_results]


def to_paper(work: dict) -> Paper | None:
    """把 OpenAlex 的一条 work 记录转换成 `Paper`；不可用的记录返回 None。

    没有摘要的记录无法用于提炼发现和校验引用，直接跳过。
    """
    title = work.get("display_name")
    abstract = rebuild_abstract(work.get("abstract_inverted_index"))
    openalex_id = str(work.get("id") or "").rsplit("/", 1)[-1]
    is_unusable = work.get("is_retracted") or work.get("type") in _SKIPPED_TYPES
    if not title or not abstract or not openalex_id or is_unusable:
        return None

    doi = _strip_doi_prefix(work.get("doi"))
    arxiv_id = _find_arxiv_id(work, doi)
    if arxiv_id:
        # 这篇论文在 arXiv 上也有：使用 arXiv 的 ID，这样能与 arXiv 检索到的同一篇论文合并
        paper_id = normalize_paper_id(f"arXiv:{arxiv_id}")
        url = f"https://arxiv.org/abs/{paper_id.split(':', 1)[1]}"
    else:
        paper_id = normalize_paper_id(f"OpenAlex:{openalex_id}")
        url = f"https://doi.org/{doi}" if doi else f"https://openalex.org/{openalex_id}"

    return Paper(
        paper_id=paper_id,
        title=" ".join(str(title).split()),
        authors=_author_names(work),
        year=int_or_none(work.get("publication_year")),
        abstract=abstract,
        url=url,
        doi=doi,
        venue=_venue(work),
        work_type=_work_type(work),
        citation_count=int_or_none(work.get("cited_by_count")),
        sources=[SOURCE_NAME],
    )


def rebuild_abstract(inverted_index: object) -> str:
    """OpenAlex 出于版权原因只提供摘要的倒排索引（词 -> 出现位置），这里还原成正文。"""
    if not isinstance(inverted_index, dict):
        return ""
    positioned_words = [
        (position, word)
        for word, positions in inverted_index.items()
        if isinstance(positions, list)
        for position in positions
        if isinstance(position, int)
    ]
    return " ".join(word for _, word in sorted(positioned_words))


def _locations(work: dict) -> list[dict]:
    """主要出处在前，其后是其余出处。"""
    candidates = [work.get("primary_location"), *(work.get("locations") or [])]
    return [location for location in candidates if isinstance(location, dict)]


def _find_arxiv_id(work: dict, doi: str | None) -> str | None:
    """从 DOI（arXiv 自己签发的 DOI 形如 10.48550/arXiv.2401.12345）或各个出处的链接里找 arXiv 编号。"""
    if doi and (match := _ARXIV_DOI.search(doi)):
        return match.group(1)
    for location in _locations(work):
        for key in ("landing_page_url", "pdf_url"):
            if match := _ARXIV_URL.search(str(location.get(key) or "")):
                return match.group(1)
    return None


def _venue(work: dict) -> str | None:
    """正式发表的期刊或会议名；所有出处都是存储库时返回 None。

    主要出处是存储库（PubMed Central、arXiv、机构库等）不代表论文没有正式发表：
    OpenAlex 常把开放获取的 PMC 副本列为主要出处，期刊本身排在其余出处里，所以要逐个查看。
    """
    for location in _locations(work):
        source = location.get("source")
        if not isinstance(source, dict):
            continue
        name = source.get("display_name")
        is_repository = source.get("type") == "repository" or "arxiv" in str(name).lower()
        if name and not is_repository:
            return str(name)
    return None


def _work_type(work: dict) -> str | None:
    """OpenAlex 的文献类型（article / review / preprint / dissertation 等）。"""
    work_type = work.get("type")
    return str(work_type) if work_type else None


def _author_names(work: dict) -> list[str]:
    names = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") if isinstance(authorship, dict) else None
        if isinstance(author, dict) and author.get("display_name"):
            names.append(str(author["display_name"]))
    return names[:_MAX_AUTHORS]


def _strip_doi_prefix(doi: object) -> str | None:
    """`https://doi.org/10.1234/abc` -> `10.1234/abc`。"""
    if not isinstance(doi, str) or not doi:
        return None
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
