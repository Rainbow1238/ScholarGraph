"""人工抽检（抽样导出与一致性计算）的测试。"""

from __future__ import annotations

import csv
import json

import pytest
from eval_fakes import make_record

from scholargraph.evaluation.agreement import (
    LABEL_COLUMN,
    Agreement,
    compute_agreement,
    export_sample,
)


def export(tmp_path, records, n=6, seed=0):
    sheet, key = tmp_path / "labeling.csv", tmp_path / "key.json"
    count = export_sample(records, n, seed, sheet, key)
    with sheet.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return count, rows, json.loads(key.read_text(encoding="utf-8")), sheet, key


def fill_labels(sheet, labels: dict[str, str]) -> None:
    with sheet.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row[LABEL_COLUMN] = labels.get(row["编号"], "")
    with sheet.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ───────────── 抽样导出 ─────────────


def test_sample_is_balanced_between_supported_and_unsupported_claims(tmp_path):
    records = [make_record("full", "q1", "s" * 20 + "u" * 5)]
    count, rows, key, *_ = export(tmp_path, records, n=6)

    assert count == len(rows) == 6
    assert sum(1 for entry in key.values() if not entry["judge_supported"]) == 3


def test_sample_is_filled_up_when_one_kind_is_scarce(tmp_path):
    count, _, key, *_ = export(tmp_path, [make_record("full", "q1", "s" * 20 + "u")], n=6)

    assert count == 6
    assert sum(1 for entry in key.values() if not entry["judge_supported"]) == 1


def test_fabricated_and_unjudged_claims_are_never_sampled(tmp_path):
    count, *_ = export(tmp_path, [make_record("direct", "q1", "ffjj")], n=6)

    assert count == 0  # 这两类没有"摘要是否支撑"可标


def test_sheet_shows_the_abstract_but_hides_the_judge_verdict(tmp_path):
    _, rows, key, *_ = export(tmp_path, [make_record("full", "q1", "su")], n=2)

    row = rows[0]
    assert "Abstract of Paper A." in row["摘要"] and "Paper A" in row["所引论文"]
    assert row[LABEL_COLUMN] == ""
    assert not any("supported" in str(value) for value in row.values())  # 避免标注者先入为主
    assert key[row["编号"]]["system"] == "full"


def test_same_seed_gives_the_same_sample(tmp_path):
    records = [make_record("full", "q1", "s" * 30 + "u" * 30)]
    _, first, *_ = export(tmp_path, records, seed=7)
    _, second, *_ = export(tmp_path, records, seed=7)

    assert first == second


# ───────────── 一致性 ─────────────


def test_agreement_compares_human_labels_with_the_judge(tmp_path):
    _, _, key, sheet, key_path = export(tmp_path, [make_record("full", "q1", "ssssuuuu")], n=8)
    # 人工与评委完全一致，除了一条：评委说支撑，人工说不支撑
    labels = {
        sample_id: "1" if entry["judge_supported"] else "0" for sample_id, entry in key.items()
    }
    flipped = next(sample_id for sample_id, entry in key.items() if entry["judge_supported"])
    labels[flipped] = "0"
    fill_labels(sheet, labels)

    result = compute_agreement(sheet, key_path)

    assert result.labeled == 8
    assert (result.both_supported, result.both_unsupported) == (3, 4)
    assert (result.only_judge_supported, result.only_human_supported) == (1, 0)
    assert result.raw_agreement == pytest.approx(7 / 8)


def test_blank_or_invalid_labels_are_skipped(tmp_path):
    _, _, key, sheet, key_path = export(tmp_path, [make_record("full", "q1", "ssuu")], n=4)
    ids = list(key)
    fill_labels(sheet, {ids[0]: "1", ids[1]: "也许", ids[2]: ""})

    assert compute_agreement(sheet, key_path).labeled == 1


@pytest.mark.parametrize(
    ("counts", "kappa"),
    [
        ((10, 10, 0, 0), 1.0),  # 完全一致
        ((5, 5, 5, 5), 0.0),  # 和随机一样
        ((0, 0, 10, 10), -1.0),  # 完全相反
        ((20, 0, 0, 0), 1.0),  # 双方全都判为支撑：kappa 无定义，按完全一致处理
        ((40, 40, 10, 10), 0.6),
    ],
)
def test_cohens_kappa(counts, kappa):
    both_yes, both_no, only_judge, only_human = counts
    agreement = Agreement(sum(counts), both_yes, both_no, only_judge, only_human)

    assert agreement.cohens_kappa == pytest.approx(kappa)


def test_no_labels_means_zero_agreement_instead_of_crashing():
    agreement = Agreement(0, 0, 0, 0, 0)
    assert (agreement.raw_agreement, agreement.cohens_kappa) == (0.0, 0.0)
