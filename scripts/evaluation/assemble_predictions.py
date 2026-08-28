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
"""Merge sample metadata and five model outputs into the canonical analysis CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


MODEL_ARGUMENTS = {
    "charbert": ("CharBERTDrug_probability", False),
    "bert": ("BERTDrug_probability", False),
    "fasttext": ("fasttext+xgboost_probability", False),
    "biowordvec": ("BioWordVec_probability", False),
    "spellchecker": ("SpellChecker_edit_distance", True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True, help="CSV containing a unique key, term, target, and optional metadata.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-column", default="sample_key")
    parser.add_argument("--prediction-key-column", default=None, help="Defaults to --key-column; use index for training-script outputs.")
    for argument in MODEL_ARGUMENTS:
        parser.add_argument(f"--{argument}", type=Path, required=True)
    return parser.parse_args()


def first_column(frame: pd.DataFrame, candidates: Iterable[str], path: Path) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"{path} has none of the expected columns: {list(candidates)}")


def load_model_output(
    path: Path,
    source_key: str,
    destination_key: str,
    score_name: str,
    spellchecker: bool,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if source_key not in frame.columns:
        raise ValueError(f"{path} has no key column {source_key!r}")
    if frame[source_key].duplicated().any():
        raise ValueError(f"{path} contains duplicate prediction keys")
    if spellchecker:
        score_column = first_column(
            frame,
            ["ranking_score", "edit_distance", "SpellChecker_edit_distance", "score", "probability"],
            path,
        )
        prediction_column: Optional[str] = next(
            (column for column in ["prediction", "SpellChecker_prediction", "pred"] if column in frame.columns),
            None,
        )
    else:
        score_column = first_column(frame, ["probability", "positive_probability", "score", score_name], path)
        prediction_column = None
    output = frame[[source_key, score_column]].rename(columns={source_key: destination_key, score_column: score_name})
    output[score_name] = pd.to_numeric(output[score_name], errors="raise")
    if spellchecker:
        if (output[score_name] < 0).any():
            raise ValueError(f"{path} contains a negative SpellChecker ranking score")
    elif ((output[score_name] < 0) | (output[score_name] > 1)).any():
        raise ValueError(
            f"{path} contains scores outside [0, 1]; the canonical analysis requires positive-class probabilities"
        )
    if "term" in frame.columns:
        output["_prediction_term"] = frame["term"]
    if "target" in frame.columns:
        output["_prediction_target"] = frame["target"]
    if spellchecker:
        if prediction_column:
            output["SpellChecker_prediction"] = pd.to_numeric(frame[prediction_column], errors="raise").astype(int)
        else:
            output["SpellChecker_prediction"] = (output[score_name] > 0).astype(int)
    return output


def normalized_terms(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.lower()


def normalized_targets(values: pd.Series) -> pd.Series:
    aliases = {
        "1": 1,
        "1.0": 1,
        "true": 1,
        "positive": 1,
        "__label__positive": 1,
        "0": 0,
        "0.0": 0,
        "false": 0,
        "negative": 0,
        "__label__negative": 0,
    }
    normalized = values.astype(str).str.strip().str.lower().map(aliases)
    if normalized.isna().any():
        invalid = sorted(values[normalized.isna()].astype(str).unique())
        raise ValueError(f"Could not normalize prediction targets: {invalid}")
    return normalized.astype(int)


def main() -> int:
    args = parse_args()
    samples = pd.read_csv(args.samples)
    required = {args.key_column, "term", "target"}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"{args.samples} is missing required columns: {missing}")
    if samples[args.key_column].duplicated().any():
        raise ValueError(f"{args.samples} contains duplicate values in {args.key_column}")
    prediction_key = args.prediction_key_column or args.key_column
    merged = samples.copy()
    for argument, (score_name, is_spellchecker) in MODEL_ARGUMENTS.items():
        model_frame = load_model_output(
            getattr(args, argument),
            prediction_key,
            args.key_column,
            score_name,
            is_spellchecker,
        )
        missing_keys = pd.Index(samples[args.key_column]).difference(model_frame[args.key_column])
        extra_keys = pd.Index(model_frame[args.key_column]).difference(samples[args.key_column])
        if len(missing_keys) or len(extra_keys):
            raise ValueError(
                f"{argument} prediction keys differ from the sample table: "
                f"{len(missing_keys)} missing, {len(extra_keys)} extra"
            )
        merged = merged.merge(model_frame, on=args.key_column, how="left", validate="one_to_one")
        if merged[score_name].isna().any():
            missing_count = int(merged[score_name].isna().sum())
            raise ValueError(f"{missing_count} samples have no {argument} prediction")
        if "_prediction_term" in merged.columns:
            mismatch = normalized_terms(merged["term"]) != normalized_terms(merged["_prediction_term"])
            if mismatch.any():
                raise ValueError(f"{int(mismatch.sum())} {argument} prediction terms do not match the sample table")
            merged = merged.drop(columns="_prediction_term")
        if "_prediction_target" in merged.columns:
            mismatch = normalized_targets(merged["target"]) != normalized_targets(merged["_prediction_target"])
            if mismatch.any():
                raise ValueError(f"{int(mismatch.sum())} {argument} prediction targets do not match the sample table")
            merged = merged.drop(columns="_prediction_target")
    if args.key_column != "sample_key":
        if "sample_key" in merged.columns:
            merged = merged.rename(columns={args.key_column: "prediction_key"})
        else:
            merged = merged.rename(columns={args.key_column: "sample_key"})
    preferred = [
        "sample_key",
        "term",
        "target",
        "target_label",
        "term_type",
        "data_type",
        "frequency",
        "CharBERTDrug_probability",
        "BERTDrug_probability",
        "SpellChecker_edit_distance",
        "fasttext+xgboost_probability",
        "BioWordVec_probability",
        "SpellChecker_prediction",
    ]
    columns = [column for column in preferred if column in merged.columns]
    columns.extend(column for column in merged.columns if column not in columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged[columns].to_csv(args.output, index=False)
    print(f"Wrote {len(merged):,} aligned sample predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
