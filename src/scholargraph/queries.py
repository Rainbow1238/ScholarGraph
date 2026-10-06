"""检索词的整理：对模型给出的子问题做确定性的清洗。

模型可能给出重复的检索词、过多的检索词，或者把已经失败过的检索词原样再提一遍。
这些都在这里用代码处理，而不是寄希望于提示词。
"""

from __future__ import annotations

from collections.abc import Collection, Iterable

from scholargraph.schemas import Note, SubQuestion


def query_key(query: str) -> str:
    """检索词的比较键：忽略大小写和多余空白，`Graph  RAG` 与 `graph rag` 视为相同。"""
    return " ".join(query.lower().split())


def tried_query_keys(notes: Iterable[Note]) -> set[str]:
    """已经检索过的全部检索词（以比较键表示）。"""
    return {query_key(query) for note in notes for query in note.search_queries}


def tidy_sub_questions(
    sub_questions: Iterable[SubQuestion],
    *,
    max_sub_questions: int,
    max_queries: int,
    already_tried: Collection[str] = frozenset(),
) -> list[SubQuestion]:
    """清洗子问题列表。

    对每个子问题：去掉空的、重复的、以前的轮次已经检索过的检索词，并限制检索词数量。
    清洗后一个检索词都不剩的子问题会被丢弃：重复同样的检索不会带来新的论文。
    （同一批里的不同子问题可以共用检索词，检索器有缓存，不会重复请求。）
    """
    tidied: list[SubQuestion] = []

    for sub_question in sub_questions:
        fresh_queries: dict[str, str] = {}  # 比较键 -> 检索词；用 dict 保序去重
        for query in sub_question.search_queries:
            key = query_key(query)
            if key and key not in already_tried:
                fresh_queries.setdefault(key, " ".join(query.split()))

        if fresh_queries:
            kept_queries = list(fresh_queries.values())[:max_queries]
            tidied.append(SubQuestion(question=sub_question.question, search_queries=kept_queries))

    return tidied[:max_sub_questions]
