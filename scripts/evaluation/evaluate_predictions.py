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
"""Compute manuscript metrics from a canonical sample-level prediction CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = RELEASE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drug_spelling.io import parse_binary_target
from drug_spelling.metrics import binary_metrics


MODEL_COLUMNS = {
    "CharBERTDrug": (["CharBERTDrug_probability", "CharBERTDrug__score"], None),
    "BERTDrug": (["BERTDrug_probability", "BERTDrug__score"], None),
    "SpellChecker": (
        ["SpellChecker_edit_distance", "SpellChecker__score", "SpellChecker__Score"],
        ["SpellChecker_prediction", "SpellChecker__pred"],
    ),
    "fastTextML": (["fasttext+xgboost_probability", "fasttext+xgboost__score"], None),
    "BioWordVecML": (["BioWordVec_probability", "BioWordVec__score"], None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--subset-column", default="data_type")
    parser.add_argument("--subset", action="append", default=[], help="Subset value; repeat. By default evaluates all and every observed value.")
    parser.add_argument("--stratify", action="append", default=[], help="Metadata column such as term_type or frequency; repeat.")
    return parser.parse_args()


def choose_column(frame: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"No {label} column found; checked {list(candidates)}")


def groups(frame: pd.DataFrame, args: argparse.Namespace) -> Iterable[Tuple[str, pd.DataFrame]]:
    yield "all", frame
    if args.subset_column in frame.columns:
        normalized = frame[args.subset_column].astype(str).str.strip().str.lower()
        if not args.subset and {"oov", "non-oov_cleaned"}.intersection(set(normalized)):
            cleaned_mask = normalized.isin({"oov", "non-oov_cleaned"})
            if cleaned_mask.any():
                yield "cleaned", frame[cleaned_mask]
        values = args.subset or sorted(frame[args.subset_column].dropna().astype(str).unique())
        for value in values:
            mask = normalized == value.lower()
            if mask.any():
                yield f"{args.subset_column}={value}", frame[mask]
    for column in args.stratify:
        if column not in frame.columns:
            raise ValueError(f"Cannot stratify by missing column {column!r}")
        for value, subset in frame.groupby(column, dropna=False):
            yield f"{column}={value}", subset


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.input)
    if "target" not in frame.columns:
        raise ValueError(f"{args.input} has no target column")
    frame["_target"] = frame["target"].map(parse_binary_target)
    model_specs = {}
    for model, (score_candidates, prediction_candidates) in MODEL_COLUMNS.items():
        score_column = choose_column(frame, score_candidates, f"{model} score")
        prediction_column = None
        if prediction_candidates:
            prediction_column = next((column for column in prediction_candidates if column in frame.columns), None)
        model_specs[model] = (score_column, prediction_column)

    output_rows: List[dict] = []
    for group_name, subset in groups(frame, args):
        targets = subset["_target"].to_numpy(dtype=np.int64)
        for model, (score_column, prediction_column) in model_specs.items():
            scores = pd.to_numeric(subset[score_column], errors="raise").to_numpy(dtype=np.float64)
            if model == "SpellChecker":
                predictions = (
                    pd.to_numeric(subset[prediction_column], errors="raise").to_numpy(dtype=np.int64)
                    if prediction_column
                    else (scores > 0).astype(np.int64)
                )
                metrics = binary_metrics(targets, predictions, scores, include_brier=False)
            else:
                predictions = (scores >= args.threshold).astype(np.int64)
                metrics = binary_metrics(targets, predictions, scores)
            output_rows.append({"group": group_name, "model": model, "threshold": None if model == "SpellChecker" else args.threshold, **metrics})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(output_rows)
    result.to_csv(args.output_dir / "metrics_long.csv", index=False)
    metric_columns = [
        "precision",
        "recall",
        "f1",
        "accuracy",
        "specificity",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]
    for group_name, group_frame in result.groupby("group", sort=False):
        safe_name = "".join(character if character.isalnum() else "_" for character in group_name).strip("_")
        group_frame.set_index("model").reindex(columns=metric_columns).T.to_csv(args.output_dir / f"table_{safe_name}.csv")
    manifest = {
        "input": str(args.input),
        "threshold": args.threshold,
        "groups": list(result["group"].drop_duplicates()),
        "model_score_columns": {model: score for model, (score, _prediction) in model_specs.items()},
    }
    (args.output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(result)} model/group rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
