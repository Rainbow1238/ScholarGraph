"""检索词清洗（纯函数）的测试。"""

from scholargraph.queries import query_key, tidy_sub_questions, tried_query_keys
from scholargraph.schemas import Note, SubQuestion


def sub_question(*queries: str, question: str = "子问题") -> SubQuestion:
    return SubQuestion(question=question, search_queries=list(queries))


def tidy(
    *sub_questions: SubQuestion, max_sub_questions=4, max_queries=3, already_tried=frozenset()
):
    return tidy_sub_questions(
        sub_questions,
        max_sub_questions=max_sub_questions,
        max_queries=max_queries,
        already_tried=already_tried,
    )


def test_query_key_ignores_case_and_extra_whitespace():
    assert query_key("  Graph   RAG ") == query_key("graph rag")


def test_duplicate_and_blank_queries_are_removed():
    [result] = tidy(sub_question("graph rag", "Graph  RAG", "   ", "knowledge graph"))
    assert result.search_queries == ["graph rag", "knowledge graph"]


def test_queries_are_capped_per_sub_question():
    [result] = tidy(sub_question("a b", "c d", "e f", "g h"), max_queries=2)
    assert result.search_queries == ["a b", "c d"]


def test_queries_tried_in_earlier_rounds_are_dropped():
    [result] = tidy(sub_question("graph rag", "lightrag"), already_tried={query_key("Graph RAG")})
    assert result.search_queries == ["lightrag"]


def test_sub_question_with_no_fresh_query_is_dropped():
    assert tidy(sub_question("graph rag"), already_tried={"graph rag"}) == []


def test_different_sub_questions_may_share_a_query_within_one_round():
    results = tidy(
        sub_question("graph rag", question="甲"), sub_question("graph rag", question="乙")
    )
    assert [r.search_queries for r in results] == [["graph rag"], ["graph rag"]]


def test_sub_questions_are_capped():
    results = tidy(
        sub_question("a b"), sub_question("c d"), sub_question("e f"), max_sub_questions=2
    )
    assert len(results) == 2


def test_tried_query_keys_collects_queries_from_all_notes():
    notes = [
        Note(
            sub_question="甲",
            search_queries=["Graph RAG"],
            retrieved_count=0,
            discarded_findings=0,
            findings=[],
        ),
        Note(
            sub_question="乙",
            search_queries=["lightrag"],
            retrieved_count=0,
            discarded_findings=0,
            findings=[],
        ),
    ]
    assert tried_query_keys(notes) == {"graph rag", "lightrag"}
