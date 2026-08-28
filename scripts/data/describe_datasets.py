# ****************************************
# MIT License
# Copyright (c) 2026 Jiayu Lu
#
# author(s): Jiayu Lu, Boston University Chobanian & Avedisian School of Medicine
# date: 2026-8-15
# ver: 1.0
#
# This code was written to support model evaluation for the 2026 paper published
# in JMIR Medical Informatics.
# The code is for research use only, and is provided as it is.
#
# See LICENSE in the project root for license terms.

#!/usr/bin/env python
"""Create long-form descriptive rows for manuscript Tables 1 and 2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = RELEASE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drug_spelling.io import read_labeled_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Labeled .txt or metadata .csv; repeat as needed.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--term-column", default="term")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--type-column", default="term_type")
    parser.add_argument("--frequency-column", default="frequency")
    parser.add_argument("--category-column", default="category")
    return parser.parse_args()


def normalize_category(value: object) -> str:
    if pd.isna(value):
        return ""
    text = " ".join(str(value).strip().lower().split())
    aliases = {
        "mispelled drug name": "misspelled drug name",
        "mispelled non-drug name": "misspelled non-drug name",
        "not sure": "not sure or non-standard",
        "nonstandard": "not sure or non-standard",
        "non-standard": "not sure or non-standard",
    }
    return aliases.get(text, text)


def load_frame(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    if path.suffix.lower() == ".txt":
        examples = read_labeled_file(path)
        return pd.DataFrame(
            {
                "term": [item.term for item in examples],
                "target": [item.target for item in examples],
            }
        )
    frame = pd.read_csv(path)
    rename = {
        args.term_column: "term",
        args.target_column: "target",
        args.split_column: "split",
        args.type_column: "term_type",
        args.frequency_column: "frequency",
        args.category_column: "category",
    }
    frame = frame.rename(columns={key: value for key, value in rename.items() if key in frame.columns})
    if "category" in frame.columns:
        frame["category"] = frame["category"].map(normalize_category)
    return frame


def grouping_specs(frame: pd.DataFrame) -> list[tuple[str, ...]]:
    specs = [
        (column,)
        for column in ["split", "target", "term_type", "frequency", "category"]
        if column in frame.columns
    ]
    for candidate in [("split", "term_type"), ("frequency", "category")]:
        if all(column in frame.columns for column in candidate):
            specs.append(candidate)
    return specs


def group_items(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> Iterable[tuple[tuple[object, ...], pd.DataFrame]]:
    grouping = columns[0] if len(columns) == 1 else list(columns)
    for key, subset in frame.groupby(grouping, dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        yield values, subset


def parent_size(frame: pd.DataFrame, columns: Sequence[str], values: Sequence[object]) -> int:
    if len(columns) == 1:
        return len(frame)
    mask = pd.Series(True, index=frame.index)
    for column, value in zip(columns[:-1], values[:-1]):
        mask &= frame[column].isna() if pd.isna(value) else frame[column].eq(value)
    return int(mask.sum())


def summary_row(
    source: str,
    grouping: str,
    group: str,
    subset: pd.DataFrame,
    source_n: int,
    parent_n: int,
) -> dict:
    return {
        "source": source,
        "grouping": grouping,
        "group": group,
        "n": len(subset),
        "percent_of_source": 100.0 * len(subset) / source_n,
        "percent_within_parent": 100.0 * len(subset) / parent_n,
        "word_count_mean": subset["word_count"].mean(),
        "word_count_sd": subset["word_count"].std(ddof=1),
        "character_count_mean": subset["character_count"].mean(),
        "character_count_sd": subset["character_count"].std(ddof=1),
    }


def summarize(path: Path, frame: pd.DataFrame) -> list[dict]:
    if "term" not in frame.columns:
        raise ValueError(f"{path} has no term column")
    frame = frame.copy()
    frame["term"] = frame["term"].fillna("").astype(str)
    frame["word_count"] = frame["term"].str.split().str.len()
    frame["character_count"] = frame["term"].str.len()
    rows = [summary_row(path.name, "all", "all", frame, len(frame), len(frame))]
    for columns in grouping_specs(frame):
        grouping = "|".join(columns)
        for values, subset in group_items(frame, columns):
            labels = [f"{column}={value}" for column, value in zip(columns, values)]
            rows.append(
                summary_row(
                    path.name,
                    grouping,
                    "|".join(labels),
                    subset,
                    len(frame),
                    parent_size(frame, columns, values),
                )
            )
    return rows


def main() -> int:
    args = parse_args()
    output_rows = []
    for path in args.input:
        output_rows.extend(summarize(path, load_frame(path, args)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(args.output, index=False)
    print(f"Wrote {len(output_rows)} summary rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
