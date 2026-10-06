"""Reviser：按引用校验的问题清单逐句修订。

模型只负责给出"这一句改成什么"，替换由代码完成。
这样没有问题的句子在修订前后保证一字不差（上一轮的校验结论仍然有效），
也不必为了改几句话而让模型把整篇综述重写一遍。
"""

from __future__ import annotations

import re

from scholargraph.citations import canonicalize_citations
from scholargraph.llm import LLM
from scholargraph.prompts import REVISER_SYSTEM, revision_prompt
from scholargraph.schemas import CitationIssue, Correction, CorrectionList
from scholargraph.state import ResearchState

_REPEATED_SPACES = re.compile(r"(?<=\S)[ \t]{2,}")
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")


def revise_report(state: ResearchState, *, llm: LLM) -> dict:
    issues = state["issues"] # type: ignore
    prompt = revision_prompt(issues, state.get("evidence", {}))
    corrections = llm.complete_structured(REVISER_SYSTEM, prompt, CorrectionList).corrections

    draft = apply_corrections(state["draft"], issues, corrections) # type: ignore
    return {"draft": canonicalize_citations(draft), "revisions": state.get("revisions", 0) + 1}


def apply_corrections(
    draft: str, issues: list[CitationIssue], corrections: list[Correction]
) -> str:
    """把每个有问题的句子替换成修改后的句子。

    模型没有给出修改、或者选择删除的句子，一律从正文中删去：
    对一份强调引用可信的综述来说，宁可少一句话，也不保留一句没有依据的话。
    """
    correction_by_issue = {correction.issue_index: correction for correction in corrections}

    for index, issue in enumerate(issues):
        correction = correction_by_issue.get(index)
        is_rewrite = correction is not None and correction.action == "rewrite"
        replacement = correction.new_text.strip() if correction is not None and is_rewrite else ""
        draft = draft.replace(issue.claim, replacement, 1)

    return _tidy_whitespace(draft)


def _tidy_whitespace(text: str) -> str:
    """删除句子后可能留下连续的空格或多余的空行，整理干净。"""
    text = _REPEATED_SPACES.sub(" ", text)  # 行首的缩进不动（Markdown 列表靠它表示层级）
    return _EXTRA_BLANK_LINES.sub("\n\n", text)
