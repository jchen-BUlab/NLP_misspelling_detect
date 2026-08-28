#****************************************
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
"""Reproduce the cleaned-LTCDC easy/difficult and model-disagreement analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = RELEASE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drug_spelling.io import parse_binary_target
from drug_spelling.metrics import MODEL_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--subset",
        default="cleaned",
        help="cleaned (oov + non-oov_cleaned), all, or an exact data_type value.",
    )
    return parser.parse_args()


def subset_frame(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset.lower() == "all" or "data_type" not in frame.columns:
        return frame.copy()
    values = frame["data_type"].astype(str).str.strip().str.lower()
    mask = values.isin({"oov", "non-oov_cleaned"}) if subset.lower() == "cleaned" else values == subset.lower()
    output = frame[mask].copy()
    if output.empty:
        raise ValueError(f"No rows matched subset {subset!r}")
    return output


def append_group_counts(
    rows: List[dict],
    frame: pd.DataFrame,
    analysis: str,
    column: str,
) -> None:
    """Append counts and within-frame percentages for one grouping column."""

    denominator = len(frame)
    if denominator == 0 or column not in frame.columns:
        return
    values = frame[column].fillna("missing").astype(str).replace("", "missing")
    for value, count in values.value_counts(dropna=False).items():
        rows.append(
            {
                "analysis": analysis,
                "group": value,
                "count": int(count),
                "percent": 100.0 * int(count) / denominator,
                "denominator": denominator,
            }
        )


def main() -> int:
    args = parse_args()
    frame = subset_frame(pd.read_csv(args.input), args.subset)
    frame["target_binary"] = frame["target"].map(parse_binary_target)
    correctness_columns: List[str] = []
    for model, spec in MODEL_SPECS.items():
        score_column = str(spec["score"])
        if score_column not in frame.columns:
            raise ValueError(f"{args.input} is missing {score_column}")
        scores = pd.to_numeric(frame[score_column], errors="raise")
        if model == "SpellChecker":
            prediction_column = str(spec["prediction"])
            predictions = (
                pd.to_numeric(frame[prediction_column], errors="raise").astype(int)
                if prediction_column in frame.columns
                else (scores > 0).astype(int)
            )
        else:
            predictions = (scores >= args.threshold).astype(int)
        safe_name = model.replace("ML", "").replace(" ", "_")
        frame[f"{safe_name}_prediction"] = predictions
        correctness_column = f"{safe_name}_correct"
        frame[correctness_column] = predictions.eq(frame["target_binary"])
        correctness_columns.append(correctness_column)

    frame["case_class"] = "mixed"
    frame.loc[frame[correctness_columns].all(axis=1), "case_class"] = "easy"
    frame.loc[(~frame[correctness_columns]).all(axis=1), "case_class"] = "difficult"
    spell_correct = frame["SpellChecker_correct"]
    transformer_correct = frame[["CharBERTDrug_correct", "BERTDrug_correct"]]
    frame["disagreement_class"] = "other"
    frame.loc[spell_correct & (~transformer_correct).all(axis=1), "disagreement_class"] = "spell_correct_transformers_wrong"
    frame.loc[(~spell_correct) & transformer_correct.all(axis=1), "disagreement_class"] = "spell_wrong_transformers_correct"

    summary_rows: List[dict] = []
    total = len(frame)
    for column in ["case_class", "disagreement_class"]:
        append_group_counts(summary_rows, frame, column, column)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "error_analysis_rows.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "error_analysis_summary.csv", index=False)
    payload = {
        "subset": args.subset,
        "n": total,
        "easy": int((frame["case_class"] == "easy").sum()),
        "difficult": int((frame["case_class"] == "difficult").sum()),
        "spell_correct_transformers_wrong": int((frame["disagreement_class"] == "spell_correct_transformers_wrong").sum()),
        "spell_wrong_transformers_correct": int((frame["disagreement_class"] == "spell_wrong_transformers_correct").sum()),
    }
    (args.output_dir / "error_analysis_counts.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
