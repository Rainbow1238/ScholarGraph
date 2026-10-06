"""论文 ID 与引用的解析、格式化。

全部是不调用模型、不访问网络的纯函数：输入文本，输出结构。
"虚构引用能被确定性地检出"这件事，靠的就是这个文件。

论文 ID 带有来源前缀：`arXiv:2401.12345`、`OpenAlex:W4385245566`、`S2:215416146`。
正文里的引用就是把 ID 放进方括号：`[arXiv:2401.12345]`，写在所支撑句子的句末标点之前。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scholargraph.quality import publication_label, weakness_reasons
from scholargraph.schemas import Paper

# 前缀的规范写法（匹配时不区分大小写）
_CANONICAL_PREFIX = {"arxiv": "arXiv", "openalex": "OpenAlex", "s2": "S2"}
_PREFIX_PATTERN = "arXiv|OpenAlex|S2"

_PREFIXED_ID = re.compile(rf"^({_PREFIX_PATTERN})\s*[:：]\s*(.+)$", re.IGNORECASE)
# 省略了前缀、但形状可以认出来的 ID：新式/旧式 arXiv 编号，OpenAlex 的 W 编号
_BARE_ARXIV_ID = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?$", re.IGNORECASE
)
_BARE_OPENALEX_ID = re.compile(r"^W\d+$", re.IGNORECASE)
_VERSION_SUFFIX = re.compile(r"v\d+$")

_CITATION = re.compile(rf"\[\s*((?:{_PREFIX_PATTERN})\s*:[^\]]+)\]", re.IGNORECASE)
_SPACE_AFTER_COLON = re.compile(r":\s+")
# 中文语境下模型偶尔会用全角括号和全角冒号写引用，例如【arXiv：2401.12345】
_FULLWIDTH_CITATION = re.compile(
    rf"[【［]\s*((?:{_PREFIX_PATTERN})\s*[:：][^】］\]]+)[】］]", re.IGNORECASE
)
_ID_SEPARATOR = re.compile(r"[;,；，\s]+")
# 写在行尾句末标点之后的引用，例如"……效果更好。[arXiv:2401.12345]"
_CITATION_AFTER_FINAL_PUNCTUATION = re.compile(
    rf"([。！？.!?])[ \t]*((?:\[(?:{_PREFIX_PATTERN}):[^\]]+\])+)[ \t]*(?=\r?$)",
    re.IGNORECASE | re.MULTILINE,
)
# 句子边界：中文句末标点之后，或英文句号后跟空白（避免把 2401.12345 里的点当成句号）
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?])\s*|(?<=\.)\s+")
_TRAILING_PUNCTUATION = re.compile(r"[\s.。]+$")

# 核心论断：概述里的句子，以及带有概括性措辞的句子。它们代表综述的结论，不能只靠弱证据支撑
_OVERVIEW_HEADING = re.compile(
    r"^#{2,}\s*(概述|摘要|总结|overview|summary|abstract)", re.IGNORECASE
)
_GENERALIZING_WORDS = re.compile(
    r"总体而言|总体上|总的来说|整体而言|综上|总之|普遍|一致(?:地|显示|表明|认为|发现)|"
    r"多项研究|大量研究|现有证据|\boverall\b|\bin summary\b|\bconsistently\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CitedClaim:
    """报告中一句带引用的论断。"""

    text: str
    paper_ids: tuple[str, ...]


# ───────────────────────── 论文 ID ─────────────────────────


def normalize_paper_id(raw_id: str) -> str:
    """把各种写法的论文 ID 统一成证据池使用的形式。

    模型有时会写成 `arxiv: 2401.12345v2`、`[arXiv:2401.12345]`，或者干脆省略前缀，
    它们指的都是同一篇论文，统一为 `arXiv:2401.12345`。认不出来的写法原样返回。
    """
    text = raw_id.strip().strip("[]").strip()

    prefixed = _PREFIXED_ID.match(text)
    if prefixed:
        prefix = _CANONICAL_PREFIX[prefixed.group(1).lower()]
        local_id = prefixed.group(2).strip()
    elif _BARE_ARXIV_ID.match(text):
        prefix, local_id = "arXiv", text
    elif _BARE_OPENALEX_ID.match(text):
        prefix, local_id = "OpenAlex", text
    else:
        return text

    if prefix == "arXiv":
        local_id = _VERSION_SUFFIX.sub("", local_id)  # 同一论文的不同版本视为同一条证据
    elif prefix == "OpenAlex":
        local_id = local_id.upper()
    return f"{prefix}:{local_id}"


def has_known_prefix(paper_id: str) -> bool:
    return _PREFIXED_ID.match(paper_id) is not None


# ───────────────────────── 引用 ─────────────────────────


def format_citation(paper_id: str) -> str:
    return f"[{paper_id}]"


def extract_paper_ids(text: str) -> list[str]:
    """按首次出现的顺序返回文本中引用过的论文 ID（去重，已归一化）。

    同时兼容 `[arXiv:A][arXiv:B]` 和 `[arXiv:A; arXiv:B]` 两种写法。
    """
    ordered_ids: dict[str, None] = {}  # 用 dict 保序去重
    for match in _CITATION.finditer(text):
        inside_brackets = _SPACE_AFTER_COLON.sub(":", match.group(1))
        for part in _ID_SEPARATOR.split(inside_brackets):
            paper_id = normalize_paper_id(part)
            if has_known_prefix(paper_id):
                ordered_ids.setdefault(paper_id)
    return list(ordered_ids)


def canonicalize_citations(text: str) -> str:
    """把正文里各种写法的引用统一成规范格式。

    1. ID 归一化并拆成一个方括号一个 ID：
       `[arxiv:2401.12345v2; arXiv:2402.00002]` -> `[arXiv:2401.12345][arXiv:2402.00002]`
    2. 行尾写在句末标点之后的引用挪到标点之前：
       `……效果更好。[arXiv:2401.12345]` -> `……效果更好 [arXiv:2401.12345]。`

    这样正文中的引用与参考文献列表、证据池里的 ID 完全一致，
    每句话和它的引用也始终在同一个句子里，便于逐句校验和逐句修订。
    写成全角括号、全角冒号的引用（`【arXiv：2401.12345】`）也会先换成半角。
    """
    text = _FULLWIDTH_CITATION.sub(lambda match: f"[{match.group(1).replace('：', ':')}]", text)
    text = _CITATION.sub(
        lambda match: "".join(format_citation(pid) for pid in extract_paper_ids(match.group(0))),
        text,
    )
    return _CITATION_AFTER_FINAL_PUNCTUATION.sub(r" \2\1", text)


def strip_citations(text: str) -> str:
    """去掉正文里的引用标记，用于统计正文篇幅。"""
    return _CITATION.sub("", text)


def split_cited_claims(report: str) -> list[CitedClaim]:
    """把报告拆成句子，只保留带引用的句子。"""
    claims: list[CitedClaim] = []
    for line in report.splitlines():
        if line.lstrip().startswith("#"):  # 跳过 Markdown 标题
            continue
        claims.extend(_cited_claims_in_line(line))
    return claims


def core_claims(report: str) -> list[CitedClaim]:
    """报告里的核心论断：概述一节（或第一个二级标题之前）的句子，以及带概括性措辞的句子。"""
    claims: list[CitedClaim] = []
    in_overview = True  # 一级标题与第一个二级标题之间的导语也算概述
    for line in report.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if not stripped.startswith("# "):  # 二级及以下标题：进入一个新的小节
                in_overview = _OVERVIEW_HEADING.match(stripped) is not None
            continue
        claims.extend(
            claim
            for claim in _cited_claims_in_line(line)
            if in_overview or _GENERALIZING_WORDS.search(claim.text)
        )
    return claims


def format_references(report: str, evidence: dict[str, Paper]) -> str:
    """根据正文中实际出现的引用生成参考文献列表（Markdown）。

    由代码生成而不是让模型去写，所以标题、作者、出处、链接不会出错；
    证据池里没有的 ID 不会出现在列表中。
    """
    lines = []
    for paper_id in extract_paper_ids(report):
        paper = evidence.get(paper_id)
        if paper is not None:
            lines.append(_format_reference(paper))
    return "\n".join(lines)


def _format_reference(paper: Paper) -> str:
    published_in = publication_label(paper)
    if paper.year is not None:
        published_in += f", {paper.year}"
    # 标题自带的句号去掉，否则会和后面统一添加的句号连成 `review.*.`
    title = _TRAILING_PUNCTUATION.sub("", paper.title)
    reference = (
        f"- {format_citation(paper.paper_id)} {_format_authors(paper.authors)}. "
        f"*{title}*. {published_in}."
    )
    if paper.citation_count is not None:
        reference += f" 被引 {paper.citation_count} 次."
    if reasons := weakness_reasons(paper):
        reference += f" 弱证据（{'、'.join(reasons)}）."
    return f"{reference} {paper.url}"


def _cited_claims_in_line(line: str) -> list[CitedClaim]:
    sentences: list[tuple[str, list[str]]] = []  # (句子文本, 该句引用的论文 ID)
    for raw_sentence in _SENTENCE_BOUNDARY.split(line):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        paper_ids = extract_paper_ids(sentence)
        is_citation_only = not _CITATION.sub("", sentence).strip()
        if is_citation_only and sentences:
            # 引用被写到了句号后面：把它归还给紧邻的前一句
            sentences[-1][1].extend(paper_ids)
        else:
            sentences.append((sentence, paper_ids))

    return [
        CitedClaim(text, tuple(dict.fromkeys(paper_ids)))
        for text, paper_ids in sentences
        if paper_ids
    ]


def _format_authors(authors: list[str], max_shown: int = 3) -> str:
    """返回不带句末标点的作者串，句号由调用方统一添加（否则会出现 `et al..`）。"""
    if not authors:
        return "作者不详"
    if len(authors) <= max_shown:
        return ", ".join(authors)
    return ", ".join(authors[:max_shown]) + " et al"
