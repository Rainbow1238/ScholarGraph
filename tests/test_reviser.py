"""逐句修订（纯函数部分）的测试。"""

from __future__ import annotations

from scholargraph.nodes.reviser import apply_corrections
from scholargraph.schemas import CitationIssue, Correction

GOOD = "第一句没有问题 [arXiv:2401.00001]。"
BAD = "第二句夸大了结论 [arXiv:2402.00002]。"
ALSO_BAD = "第三句引用了不存在的论文 [arXiv:9999.99999]。"
DRAFT = f"# 标题\n\n{GOOD}{BAD}\n\n- 列表项\n  - 缩进的子项：{ALSO_BAD}\n"


def issue(claim: str) -> CitationIssue:
    return CitationIssue(kind="unsupported", claim=claim, paper_ids=[], reason="摘要未提及")


def rewrite(index: int, text: str) -> Correction:
    return Correction(issue_index=index, action="rewrite", new_text=text)


def test_only_the_flagged_sentence_changes():
    fixed = "第二句与摘要一致 [arXiv:2402.00002]。"
    revised = apply_corrections(DRAFT, [issue(BAD)], [rewrite(0, fixed)])

    assert revised == DRAFT.replace(BAD, fixed)


def test_each_correction_is_matched_to_its_issue_by_index_not_by_order():
    corrections = [
        rewrite(1, "改写后的第三句 [arXiv:2401.00001]。"),
        rewrite(0, "改写后的第二句 [arXiv:2402.00002]。"),
    ]
    revised = apply_corrections(DRAFT, [issue(BAD), issue(ALSO_BAD)], corrections)

    assert "改写后的第二句" in revised and "改写后的第三句" in revised
    assert BAD not in revised and ALSO_BAD not in revised
    assert revised.index("改写后的第二句") < revised.index("改写后的第三句")


def test_delete_removes_the_sentence_and_keeps_the_layout_tidy():
    corrections = [Correction(issue_index=0, action="delete", new_text="")]
    revised = apply_corrections(DRAFT, [issue(BAD)], corrections)

    assert BAD not in revised
    assert GOOD in revised
    assert "\n\n\n" not in revised
    assert "  - 缩进的子项" in revised  # 行首缩进不受影响


def test_issue_without_a_correction_is_deleted():
    revised = apply_corrections(
        DRAFT, [issue(BAD), issue(ALSO_BAD)], [rewrite(0, "改好的 [arXiv:2402.00002]。")]
    )

    assert ALSO_BAD not in revised
    assert "改好的" in revised


def test_correction_for_an_unknown_issue_index_is_ignored():
    revised = apply_corrections(
        DRAFT, [issue(BAD)], [rewrite(7, "不相干的句子。"), rewrite(0, "改好的。")]
    )

    assert "不相干的句子" not in revised
    assert "改好的。" in revised


def test_rewrite_with_empty_text_is_treated_as_deletion():
    revised = apply_corrections(DRAFT, [issue(BAD)], [rewrite(0, "   ")])

    assert BAD not in revised
