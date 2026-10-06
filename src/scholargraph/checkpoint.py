"""Checkpoint 持久化：把每一步之后的状态存进 SQLite。

有了它，图才能在人工审批处暂停、在进程重启后从断点继续，
中途失败时也不必把已经花钱跑完的步骤重跑一遍。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from scholargraph.schemas import (
    CitationIssue,
    Finding,
    Note,
    Paper,
    SubQuestion,
    VerificationRound,
)

# 状态里出现的自定义类型。反序列化 checkpoint 时只允许还原这份白名单里的类型，
# 这是 LangGraph 的安全要求：防止被篡改的数据库在加载时构造出任意对象。
CHECKPOINTED_TYPES = (SubQuestion, Paper, Finding, Note, CitationIssue, VerificationRound)


@contextmanager
def open_checkpointer(path: Path) -> Iterator[SqliteSaver]:
    """打开（必要时创建）SQLite checkpoint 库，退出 `with` 块时自动关闭连接。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    # 并行节点会在不同线程里写 checkpoint，所以关闭 sqlite3 的同线程检查；
    # SqliteSaver 内部有锁，保证写入是串行的。
    connection = sqlite3.connect(path, check_same_thread=False)
    try:
        yield SqliteSaver(connection, serde=_build_serializer())
    finally:
        connection.close()


def _build_serializer() -> JsonPlusSerializer:
    allowed_types = [(cls.__module__, cls.__name__) for cls in CHECKPOINTED_TYPES]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed_types)
