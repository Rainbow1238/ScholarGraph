"""评测集读取的测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from scholargraph.evaluation.dataset import load_questions


def write(tmp_path, text: str, encoding: str = "utf-8") -> Path:
    path = tmp_path / "questions.jsonl"
    path.write_text(text, encoding=encoding)
    return path


def test_loads_questions_skipping_comments_and_blank_lines(tmp_path):
    path = write(
        tmp_path,
        '# 说明\n\n{"id": "q1", "question": "问题一", "aspects": ["甲", "乙"]}\n'
        '{"id": "q2", "question": "问题二", "aspects": ["丙"]}\n',
        encoding="utf-8-sig",  # Windows 记事本保存的文件带 BOM
    )
    questions = load_questions(path)

    assert [q.id for q in questions] == ["q1", "q2"]
    assert questions[0].aspects == ["甲", "乙"]


@pytest.mark.parametrize(
    "line",
    [
        '{"id": "q1", "question": "缺少要点"}',
        '{"id": "q1", "question": "要点为空", "aspects": []}',
        "不是 JSON",
    ],
)
def test_invalid_line_is_reported_with_its_line_number(tmp_path, line):
    with pytest.raises(ValueError, match="第 2 行"):
        load_questions(write(tmp_path, f"# 说明\n{line}\n"))


def test_duplicate_ids_are_rejected(tmp_path):
    line = '{"id": "q1", "question": "问题", "aspects": ["甲"]}\n'
    with pytest.raises(ValueError, match="重复"):
        load_questions(write(tmp_path, line * 2))


def test_the_shipped_question_set_is_valid():
    questions = load_questions(Path(__file__).parent.parent / "evals" / "questions.jsonl")

    assert len(questions) == 20
    assert all(len(q.aspects) >= 4 for q in questions)
