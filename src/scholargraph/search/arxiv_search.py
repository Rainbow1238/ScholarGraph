"""基于 arXiv 官方 API 的检索器（免费，无需密钥）。"""

from __future__ import annotations

import re
import threading
from typing import Literal

import arxiv
import requests
from requests.adapters import HTTPAdapter

from scholargraph.citations import normalize_paper_id
from scholargraph.schemas import Paper
from scholargraph.search.base import SearchError

SOURCE_NAME = "arXiv"
ARXIV_MIN_INTERVAL_SECONDS = 3.0  # arXiv 使用条款：同一客户端两次请求至少间隔 3 秒
REQUEST_TIMEOUT_SECONDS = (10.0, 30.0)  # (建立连接, 等待响应)
MAX_QUERY_TERMS = 6
_WORD = re.compile(r"[A-Za-z0-9]+")
_WHITESPACE = re.compile(r"\s+")
# 不携带检索信息的虚词；其中 and / or / not 还与 arXiv 的布尔运算符同形，必须去掉
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
        "or",
        "not",
        "andnot",
        "with",
        "by",
        "via",
        "from",
        "at",
        "as",
        "is",
        "are",
        "vs",
    ]
)


class ArxivSearcher:
    """线程安全的 arXiv 检索器，采用"先严后宽"的两段式检索。

    arXiv 是关键词匹配，不是语义检索：检索词之间用 AND 连接时结果精确但可能为空，
    用 OR 连接时结果多但混杂。所以先用 AND 检索，一篇都没有时再放宽为 OR。
    两种写法都显式给出运算符，不依赖 arXiv 对"空格分隔的多个词"的默认解释。

    多个 Researcher 会在不同线程里同时调用 `search`。arXiv 要求限速，
    而 `arxiv.Client` 的限速计时不是线程安全的，所以这里用一把锁让检索请求排队：
    检索是串行的，耗时更长的 LLM 调用仍然是并行的。
    """

    def __init__(self, min_interval_seconds: float = ARXIV_MIN_INTERVAL_SECONDS) -> None:
        self._client = arxiv.Client(page_size=20, delay_seconds=min_interval_seconds, num_retries=3)
        _enforce_timeout(self._client)
        self._lock = threading.Lock()

    def search(self, query: str, max_results: int) -> list[Paper]:
        terms = extract_terms(query)
        if not terms:
            return []

        papers = self._fetch(build_arxiv_query(terms, "AND"), max_results)
        if not papers and len(terms) > 1:
            papers = self._fetch(build_arxiv_query(terms, "OR"), max_results)
        return papers

    def _fetch(self, arxiv_query: str, max_results: int) -> list[Paper]:
        request = arxiv.Search(
            query=arxiv_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        try:
            with self._lock:
                results = list(self._client.results(request))
        except (arxiv.ArxivError, requests.RequestException) as error:
            raise SearchError(f"arXiv 检索失败（{arxiv_query}）：{error}") from error
        return [to_paper(result) for result in results]


class _TimeoutAdapter(HTTPAdapter):
    """给每个 HTTP 请求加上超时。"""

    def send(self, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = REQUEST_TIMEOUT_SECONDS
        return super().send(request, **kwargs)


def _enforce_timeout(client: arxiv.Client) -> None:
    """`arxiv.Client` 发请求时不设超时：网络卡住时会一直等下去，还占着检索锁。

    这里给它内部的 HTTP 会话装上带超时的适配器，超时后抛出的异常会被转换成 `SearchError`。
    """
    session = getattr(client, "_session", None)
    if isinstance(session, requests.Session):
        session.mount("http://", _TimeoutAdapter())
        session.mount("https://", _TimeoutAdapter())


def extract_terms(query: str) -> list[str]:
    """从检索词中提取用于匹配的词：只保留英文单词和数字，去掉虚词和重复词。

    连字符会把词拆开（`multi-hop` -> `multi`、`hop`），与 arXiv 自身的分词方式一致。
    """
    terms: dict[str, None] = {}  # 用 dict 保序去重
    for word in _WORD.findall(query):
        lowered = word.lower()
        if len(lowered) > 1 and lowered not in _STOPWORDS:
            terms.setdefault(lowered)
    return list(terms)[:MAX_QUERY_TERMS]


def build_arxiv_query(terms: list[str], operator: Literal["AND", "OR"]) -> str:
    """生成 arXiv API 的查询语句，例如 `all:graph AND all:rag`。

    `all:` 表示在标题、摘要、作者等所有字段中匹配。
    """
    return f" {operator} ".join(f"all:{term}" for term in terms)


def to_paper(result: arxiv.Result) -> Paper:
    """把 arXiv 的返回结果转换成系统内部的 `Paper`。"""
    paper_id = normalize_paper_id(f"arXiv:{result.get_short_id()}")
    arxiv_number = paper_id.split(":", 1)[1]
    return Paper(
        paper_id=paper_id,
        title=_collapse_whitespace(result.title),
        authors=[author.name for author in result.authors],
        year=result.published.year,
        abstract=_collapse_whitespace(result.summary),
        url=f"https://arxiv.org/abs/{arxiv_number}",  # 不带版本号，始终指向最新版本
        doi=result.doi or None,
        work_type="preprint",  # 正式发表的出处由 OpenAlex 等来源在融合时补上
        sources=[SOURCE_NAME],
    )


def _collapse_whitespace(text: str) -> str:
    """arXiv 的标题和摘要里带有排版用的换行，统一压成单个空格。"""
    return _WHITESPACE.sub(" ", text).strip()
