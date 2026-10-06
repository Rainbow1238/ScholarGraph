"""论文检索层：节点只依赖 `PaperSearcher` 协议。"""

from __future__ import annotations

from scholargraph.config import ConfigError, Settings
from scholargraph.search.arxiv_search import ArxivSearcher
from scholargraph.search.base import PaperSearcher, SearchError
from scholargraph.search.federated import FederatedSearcher
from scholargraph.search.openalex_search import OpenAlexSearcher
from scholargraph.search.semantic_scholar_search import SemanticScholarSearcher

__all__ = [
    "ArxivSearcher",
    "FederatedSearcher",
    "OpenAlexSearcher",
    "PaperSearcher",
    "SearchError",
    "SemanticScholarSearcher",
    "build_searcher",
]


def build_searcher(settings: Settings) -> FederatedSearcher:
    """按配置里的 SEARCH_SOURCES 创建检索源，并包装成一个联邦检索器。"""
    factories = {
        "arxiv": lambda: ("arXiv", ArxivSearcher()),
        "openalex": lambda: ("OpenAlex", OpenAlexSearcher(settings.openalex_api_key)),
        "semantic_scholar": lambda: (
            "Semantic Scholar",
            SemanticScholarSearcher(settings.semantic_scholar_api_key),
        ),
    }
    unknown = [name for name in settings.search_sources if name not in factories]
    if unknown:
        raise ConfigError(
            f"SEARCH_SOURCES 里有不认识的检索源：{', '.join(unknown)}。可选：{', '.join(factories)}"
        )
    return FederatedSearcher(dict(factories[name]() for name in settings.search_sources))
