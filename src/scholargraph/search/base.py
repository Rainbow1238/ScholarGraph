"""检索器的抽象接口。

想换成 Semantic Scholar、OpenAlex 或本地知识库时，
只需再写一个实现了 `search` 方法的类，图和节点都不用改。
"""

from __future__ import annotations

from typing import Protocol

from scholargraph.schemas import Paper


class SearchError(RuntimeError):
    """检索失败（网络错误、服务不可用等）。Researcher 节点会针对它自动重试。"""


class PaperSearcher(Protocol):
    def search(self, query: str, max_results: int) -> list[Paper]:
        """按相关度返回至多 `max_results` 篇论文；没有结果时返回空列表。"""
        ...
