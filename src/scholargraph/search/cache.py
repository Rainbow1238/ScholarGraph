"""检索结果的磁盘缓存。

评测时同一个检索词会被不同的系统配置反复用到。把结果缓存到磁盘有三个好处：
各配置面对的是同一批检索结果，对比才公平；中断后续跑不必重新请求；节省检索源的每日额度。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from scholargraph.schemas import Paper
from scholargraph.search.base import PaperSearcher


class CachedSearcher:
    """给任意检索器加上磁盘缓存（它自己也满足 `PaperSearcher` 协议）。

    `namespace` 用来区分不同的检索配置：换了检索源组合之后，旧的缓存不应该被复用。
    只缓存成功的结果；检索失败会原样抛出，下次再试。
    """

    def __init__(self, inner: PaperSearcher, path: Path, namespace: str = "") -> None:
        self._inner = inner
        self._namespace = namespace
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        # 多个 Researcher 在不同线程里检索，所以关闭同线程检查，并用锁保证读写串行
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS search_cache (key TEXT PRIMARY KEY, papers TEXT NOT NULL)"
        )
        self._connection.commit()
        self.hits = 0
        self.misses = 0

    def search(self, query: str, max_results: int) -> list[Paper]:
        key = f"{self._namespace}|{' '.join(query.lower().split())}|{max_results}"
        with self._lock:
            row = self._connection.execute(
                "SELECT papers FROM search_cache WHERE key = ?", (key,)
            ).fetchone()
            if row is not None:
                self.hits += 1
                return [Paper.model_validate(item) for item in json.loads(row[0])]

            papers = self._inner.search(query, max_results)
            serialized = json.dumps([paper.model_dump() for paper in papers], ensure_ascii=False)
            self._connection.execute(
                "INSERT OR REPLACE INTO search_cache (key, papers) VALUES (?, ?)", (key, serialized)
            )
            self._connection.commit()
            self.misses += 1
            return papers

    def close(self) -> None:
        with self._lock:
            self._connection.close()
