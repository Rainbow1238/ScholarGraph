"""Writer：根据笔记撰写综述初稿。"""

from __future__ import annotations

from scholargraph.citations import canonicalize_citations
from scholargraph.llm import LLM
from scholargraph.prompts import WRITER_SYSTEM, writer_prompt
from scholargraph.state import ResearchState


def write_report(state: ResearchState, *, llm: LLM) -> dict:
    prompt = writer_prompt(
        question=state["question"], # pyright: ignore[reportTypedDictNotRequiredAccess]
        scope=state["scope"], # type: ignore
        notes=state.get("notes", []),
        evidence=state.get("evidence", {}),
        open_gaps=state.get("open_gaps", []),
    )
    draft = llm.complete(WRITER_SYSTEM, prompt)
    return {"draft": canonicalize_citations(draft)}
