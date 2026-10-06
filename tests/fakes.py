"""测试替身：不联网、不花钱的假模型和假检索器。"""

from __future__ import annotations

import threading
from collections import defaultdict

from pydantic import BaseModel

from scholargraph.llm import TokenUsage
from scholargraph.schemas import Paper
from scholargraph.search import SearchError


class FakeLLM:
    """按预设脚本作答的假模型。

    每种输出类型各有一个应答队列，按调用顺序依次取用；队列只剩最后一个时重复使用它。
    所有调用都会被记录下来，供测试断言。
    """

    def __init__(
        self, structured: dict[type[BaseModel], list[BaseModel]], texts: list[str]
    ) -> None:
        self._structured = {schema: list(replies) for schema, replies in structured.items()}
        self._texts = list(texts)
        self._lock = threading.Lock()
        self.prompts: dict[str, list[str]] = defaultdict(list)  # 输出类型名 -> 收到的用户提示词
        self.usage = TokenUsage()  # 与真实的 LLM 客户端保持同样的属性

    def complete(self, system: str, user: str) -> str:
        with self._lock:
            self.prompts["text"].append(user)
            return self._next(self._texts)

    def complete_structured(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        with self._lock:
            self.prompts[schema.__name__].append(user)
            return self._next(self._structured[schema])

    @staticmethod
    def _next(queue: list):
        return queue.pop(0) if len(queue) > 1 else queue[0]


class FakeSearcher:
    """假检索器。

    默认对任何检索词都返回 `papers`；可以用 `by_query` 为个别检索词指定结果，
    用 `failing` 指定哪些检索词会抛出 `SearchError`。
    """

    def __init__(
        self,
        papers: list[Paper],
        *,
        by_query: dict[str, list[Paper]] | None = None,
        failing: tuple[str, ...] = (),
    ) -> None:
        self._papers = papers
        self._by_query = by_query or {}
        self._failing = set(failing)
        self._lock = threading.Lock()
        self.queries: list[str] = []

    def search(self, query: str, max_results: int) -> list[Paper]:
        with self._lock:
            self.queries.append(query)
        if query in self._failing:
            raise SearchError(f"模拟的检索失败：{query}")
        return self._by_query.get(query, self._papers)[:max_results]


def make_paper(paper_id: str, title: str = "A Paper", **overrides) -> Paper:
    """构造一篇测试用的论文。`paper_id` 省略前缀时按 arXiv 处理。

    默认是一篇 arXiv 预印本；传入 venue 时类型默认改为期刊论文。
    """
    full_id = paper_id if ":" in paper_id else f"arXiv:{paper_id}"
    fields = {
        "paper_id": full_id,
        "title": title,
        "authors": ["Ada Lovelace", "Alan Turing"],
        "year": 2024,
        "abstract": f"Abstract of {title}.",
        "url": f"https://example.org/{full_id}",
        "work_type": "article" if overrides.get("venue") else "preprint",
        "sources": ["arXiv"],
    }
    return Paper(**{**fields, **overrides})
