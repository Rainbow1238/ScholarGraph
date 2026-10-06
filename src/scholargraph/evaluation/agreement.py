"""人工抽检：度量 LLM 评委的判断有多可信。

流程：
1. `export_sample` 从评测结果里抽取若干条"论断 + 摘要"，导出成表格（不显示评委的判断，避免先入为主）。
2. 人工在表格里逐条填写：摘要是否支撑论断（1 = 支撑，0 = 不支撑）。
3. `compute_agreement` 把人工标注与评委的判断对比，给出一致率和 Cohen's kappa。
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from scholargraph.evaluation.records import RunRecord

LABEL_COLUMN = "人工判断（1=支撑，0=不支撑）"
_COLUMNS = ["编号", "论断", "所引论文", "摘要", LABEL_COLUMN]
_LABELS = {"1": True, "0": False}


@dataclass
class Agreement:
    """评委与人工标注的一致性。"""

    labeled: int
    both_supported: int
    both_unsupported: int
    only_judge_supported: int  # 评委认为支撑、人工认为不支撑：评委偏乐观
    only_human_supported: int

    @property
    def raw_agreement(self) -> float:
        return (self.both_supported + self.both_unsupported) / self.labeled if self.labeled else 0.0

    @property
    def cohens_kappa(self) -> float:
        """Cohen's kappa：扣除"碰巧一致"之后的一致程度。1 为完全一致，0 相当于随机。"""
        if not self.labeled:
            return 0.0
        judge_yes = (self.both_supported + self.only_judge_supported) / self.labeled
        human_yes = (self.both_supported + self.only_human_supported) / self.labeled
        expected = judge_yes * human_yes + (1 - judge_yes) * (1 - human_yes)
        if expected == 1.0:  # 双方的判断全部相同且只有一种取值，kappa 无定义，按完全一致处理
            return 1.0
        return (self.raw_agreement - expected) / (1 - expected)


def export_sample(
    records: list[RunRecord], sample_size: int, seed: int, sheet_path: Path, key_path: Path
) -> int:
    """抽样并导出标注表，返回实际抽到的条数。

    尽量让评委判为"支撑"和"不支撑"的各占一半：不支撑的论断通常很少，
    完全随机抽样的话几乎抽不到，也就无从知道评委在这类论断上准不准。
    """
    candidates = []
    for record in records:
        evaluation = record.evaluation
        if evaluation is None:
            continue
        papers = {**record.output.evidence, **evaluation.resolved_papers}
        for claim in evaluation.claims:
            if claim.verdict in ("supported", "unsupported"):
                cited = [papers[pid] for pid in claim.paper_ids if pid in papers]
                candidates.append((record, claim, cited))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    unsupported = [item for item in candidates if item[1].verdict == "unsupported"]
    supported = [item for item in candidates if item[1].verdict == "supported"]
    half = sample_size // 2
    chosen = unsupported[:half] + supported[: sample_size - min(half, len(unsupported))]
    chosen += unsupported[half : half + sample_size - len(chosen)]  # 支撑的不够时用不支撑的补足
    rng.shuffle(chosen)

    key = {}
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig：让 Windows 上的 Excel 正确识别中文
    with sheet_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(_COLUMNS)
        for number, (record, claim, cited) in enumerate(chosen, start=1):
            sample_id = f"S{number:03d}"
            writer.writerow(
                [
                    sample_id,
                    claim.claim,
                    "\n".join(f"{paper.paper_id} {paper.title}" for paper in cited),
                    "\n\n".join(f"[{paper.paper_id}] {paper.abstract}" for paper in cited),
                    "",
                ]
            )
            key[sample_id] = {
                "judge_supported": claim.verdict == "supported",
                "system": record.system,
                "question_id": record.question_id,
            }
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(chosen)


def compute_agreement(sheet_path: Path, key_path: Path) -> Agreement:
    """读取标注表，与评委的判断对比。没有填写或填写不合法的行会被跳过。"""
    key = json.loads(key_path.read_text(encoding="utf-8"))
    counts = {(True, True): 0, (False, False): 0, (True, False): 0, (False, True): 0}

    with sheet_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            human = _LABELS.get((row.get(LABEL_COLUMN) or "").strip())
            entry = key.get((row.get("编号") or "").strip())
            if human is not None and entry is not None:
                counts[(entry["judge_supported"], human)] += 1

    return Agreement(
        labeled=sum(counts.values()),
        both_supported=counts[(True, True)],
        both_unsupported=counts[(False, False)],
        only_judge_supported=counts[(True, False)],
        only_human_supported=counts[(False, True)],
    )
