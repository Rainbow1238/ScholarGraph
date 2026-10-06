"""评测集：研究问题，以及回答每个问题应当覆盖的要点。"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvalQuestion(BaseModel):
    """一道评测题。

    `aspects` 是人工写的"参考要点"：一份合格的综述应当覆盖这些方面。
    它只用来衡量覆盖率，不会给被评测的系统看到。
    """

    id: str
    question: str
    aspects: list[str] = Field(min_length=1)


def load_questions(path: Path) -> list[EvalQuestion]:
    """读取 JSON Lines 格式的评测集：每行一道题；空行和以 # 开头的行会被忽略。"""
    questions = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            questions.append(EvalQuestion.model_validate(json.loads(line)))
        except ValueError as error:
            raise ValueError(f"{path} 第 {line_number} 行不是合法的评测题：{error}") from error

    duplicated = {q.id for q in questions if sum(1 for other in questions if other.id == q.id) > 1}
    if duplicated:
        raise ValueError(f"{path} 里有重复的题目 ID：{', '.join(sorted(duplicated))}")
    return questions
