"""Finalizer：给正文附上参考文献、来源构成和校验说明，得到最终报告。不调用模型。"""

from __future__ import annotations

from collections import Counter

from scholargraph.citations import extract_paper_ids, format_references
from scholargraph.quality import is_weak_evidence, publication_label
from scholargraph.schemas import CitationIssue, Paper
from scholargraph.state import ResearchState


def finalize_report(state: ResearchState) -> dict:
    draft = state["draft"].strip()  # type: ignore
    evidence = state.get("evidence", {})
    cited_papers = [evidence[pid] for pid in extract_paper_ids(draft) if pid in evidence]

    references = format_references(draft, evidence)
    report = (
        f"{draft}\n\n"
        f"## 参考文献\n\n{references or '（正文未引用任何论文）'}\n\n"
        f"## 文献来源构成\n\n{render_source_composition(cited_papers)}\n\n"
        f"## 引用校验说明\n\n{_render_verification(draft, state)}\n"
    )
    return {"report": report}


# 一个出处占到这个比例以上，就提示来源集中；论文太少时比例没有意义，不做提示
CONCENTRATION_SHARE = 1 / 3
CONCENTRATION_MIN_PAPERS = 6
_UNPUBLISHED_LABELS = ("预印本", "学位论文", "出处不详")


def render_source_composition(papers: list[Paper]) -> str:
    """统计被引用论文的出处和证据强度，让读者一眼看出文献是否过于集中、是否依赖弱证据。

    集中度按期刊/会议统计，而不是按检索接口：OpenAlex、Semantic Scholar 是索引库，
    所有论文都经 OpenAlex 检索到，并不说明它们来自同一个出版方。
    """
    if not papers:
        return "正文没有引用任何论文。"

    total = len(papers)
    by_label = Counter(publication_label(paper) for paper in papers)
    by_venue = Counter(
        {label: count for label, count in by_label.items() if label not in _UNPUBLISHED_LABELS}
    )
    published = sum(by_venue.values())
    by_source = Counter(source for paper in papers for source in paper.sources)
    weak = sum(1 for paper in papers if is_weak_evidence(paper))

    status = [f"已在期刊或会议正式发表 {published} 篇"]
    status += [f"{label} {by_label[label]} 篇" for label in _UNPUBLISHED_LABELS if by_label[label]]
    lines = [f"- 共引用 {total} 篇论文。", "- 按发表情况：" + "，".join(status) + "。"]
    if by_venue:
        top = "、".join(f"{venue} {count} 篇" for venue, count in by_venue.most_common(3))
        lines.append(f"- 按出处：来自 {len(by_venue)} 种期刊或会议，最多的是 {top}。")
    lines.append(f"- 弱证据（发表满一年仍被引 0 次、学位论文或自存档平台）：{weak} 篇。")
    lines.append(
        "- 按检索接口（同一篇论文可能被多个接口找到；接口是索引库，不代表出版方）："
        + "、".join(f"{source} {count} 篇" for source, count in by_source.most_common())
        + "。"
    )

    if by_venue and total >= CONCENTRATION_MIN_PAPERS:
        venue, count = by_venue.most_common(1)[0]
        if count / total > CONCENTRATION_SHARE:
            lines.append(
                f"- 注意：{count} 篇（{count / total:.0%}）来自 {venue}，出处较为集中，"
                "结论可能受该出处的选题和审稿倾向影响。"
            )
    if (total - published) * 2 > total:
        lines.append(
            f"- 注意：一半以上的论文（{total - published} 篇）没有正式发表的出处，"
            "未经同行评审，请谨慎采信。"
        )
    return "\n".join(lines)


def _render_verification(draft: str, state: ResearchState) -> str:
    """如实说明校验结果：没有引用就说没有引用，仍有问题或没能校验的都列出来，而不是悄悄略过。"""
    if not extract_paper_ids(draft):
        return "正文没有引用任何论文，因此没有可校验的论断。"

    issues: list[CitationIssue] = state.get("issues", [])
    unverified: list[str] = state.get("unverified_claims", [])
    if not issues and not unverified:
        return (
            "正文中所有带引用的论断均已通过自动校验："
            "引用的论文真实存在于检索结果中，且论断与摘要一致。"
        )

    sections = []
    if issues:
        lines = ["以下论断未通过自动校验，请谨慎采信："]
        lines += [f"- {issue.claim}\n  - 原因：{issue.reason}" for issue in issues]
        sections.append("\n".join(lines))
    if unverified:
        lines = ["以下论断引用的论文真实存在，但未能完成论断与摘要的一致性校验："]
        lines += [f"- {claim}" for claim in unverified]
        sections.append("\n".join(lines))
    return "\n\n".join(sections)
