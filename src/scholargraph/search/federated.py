"""联邦检索：把同一个检索词发给多个检索源，融合结果。

解决"所有论文都来自同一个来源"的问题，同时保证任何一个来源出故障都不会拖垮整次运行。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass

from scholargraph.schemas import Paper
from scholargraph.search.base import PaperSearcher, SearchError
from scholargraph.search.fusion import fuse

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 2


@dataclass
class SourceStats:
    """一个检索源在本次运行中的表现。"""

    requests: int = 0
    failures: int = 0
    papers: int = 0
    disabled_reason: str | None = None  # 非空表示该来源已被熔断，本次运行不再使用

    @property
    def is_active(self) -> bool:
        return self.disabled_reason is None


class FederatedSearcher:
    """把多个检索源包装成一个检索器（它自己也满足 `PaperSearcher` 协议）。

    - 容错：单个来源失败时使用其余来源的结果；只有全部来源都失败才抛出 `SearchError`。
    - 熔断：一个来源连续失败 2 次就停用到本次运行结束，避免每个检索词都白等一次超时。
    - 缓存：相同的检索词只检索一次（不同子问题常常会用到同一个检索词）。
    - 线程安全：整个检索过程持有一把锁。多个 Researcher 的检索因此是串行的，
      这本来就是限速所要求的；耗时更长的 LLM 调用仍然并行。
    """

    def __init__(self, sources: dict[str, PaperSearcher]) -> None:
        if not sources:
            raise ValueError("至少需要一个检索源")
        self._sources = sources
        self._stats = {name: SourceStats() for name in sources}
        self._consecutive_failures = dict.fromkeys(sources, 0)
        self._cache: dict[tuple[str, int], list[Paper]] = {}
        self._lock = threading.Lock()

    def search(self, query: str, max_results: int) -> list[Paper]:
        """向每个可用的来源各取至多 `max_results` 篇，返回融合去重后的列表。"""
        cache_key = (" ".join(query.lower().split()), max_results)
        with self._lock:
            if cache_key not in self._cache:
                self._cache[cache_key] = self._search_all_sources(query, max_results)
            return list(self._cache[cache_key])  # 返回副本，调用方改动列表不会污染缓存

    def stats(self) -> dict[str, dict]:
        """各检索源的统计，用于终端输出和指标文件。"""
        with self._lock:
            return {name: asdict(stats) for name, stats in self._stats.items()}

    def _search_all_sources(self, query: str, max_results: int) -> list[Paper]:
        active_sources = [name for name, stats in self._stats.items() if stats.is_active]
        if not active_sources:
            raise SearchError("所有检索源都已停用：" + self._describe_disabled_sources())

        result_lists: list[list[Paper]] = []
        errors: list[SearchError] = []
        for name in active_sources:
            try:
                papers = self._sources[name].search(query, max_results)
            except SearchError as error:
                errors.append(error)
                self._record_failure(name, error)
            else:
                self._record_success(name, len(papers))
                result_lists.append(papers)

        if not result_lists:
            raise SearchError(f"检索词 {query!r} 在所有检索源上都失败了") from errors[-1]
        return fuse(result_lists)

    def _record_success(self, name: str, paper_count: int) -> None:
        stats = self._stats[name]
        stats.requests += 1
        stats.papers += paper_count
        self._consecutive_failures[name] = 0

    def _record_failure(self, name: str, error: SearchError) -> None:
        stats = self._stats[name]
        stats.requests += 1
        stats.failures += 1
        self._consecutive_failures[name] += 1
        logger.warning("检索源 %s 请求失败：%s", name, error)

        if self._consecutive_failures[name] >= MAX_CONSECUTIVE_FAILURES:
            stats.disabled_reason = str(error)
            logger.warning(
                "检索源 %s 连续失败 %d 次，本次运行不再使用", name, MAX_CONSECUTIVE_FAILURES
            )

    def _describe_disabled_sources(self) -> str:
        return "；".join(
            f"{name}（{stats.disabled_reason}）"
            for name, stats in self._stats.items()
            if not stats.is_active
        )
