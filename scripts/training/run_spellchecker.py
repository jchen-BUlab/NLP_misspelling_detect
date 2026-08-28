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
"""Run the dictionary-based SpellChecker baseline used by the paper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from spellchecker import SpellChecker
from tqdm import tqdm


RELEASE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = RELEASE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drug_spelling.io import LabeledExample, read_labeled_file
from drug_spelling.metrics import binary_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=RELEASE_ROOT / "data" / "classification" / "rxnorm")
    parser.add_argument(
        "--dictionary-data-dir",
        type=Path,
        default=None,
        help="Directory containing training/development files; defaults to --data-dir.",
    )
    parser.add_argument("--train-file", default="train1.txt")
    parser.add_argument("--dev-file", default="dev.txt")
    parser.add_argument("--test-file", default="test.txt")
    parser.add_argument("--output-dir", type=Path, default=RELEASE_ROOT / "artifacts" / "models" / "spellchecker")
    parser.add_argument("--distance", type=int, choices=[1, 2], default=1)
    parser.add_argument("--preserve-case", action="store_true")
    return parser.parse_args()


def tokenize(term: str, lower_case: bool) -> List[str]:
    value = term.lower() if lower_case else term
    return value.split()


def add_correct_terms(spell: SpellChecker, examples: Sequence[LabeledExample], lower_case: bool) -> int:
    words: List[str] = []
    for example in examples:
        if example.target == 0:
            words.extend(tokenize(example.term, lower_case))
    spell.word_frequency.load_words(words)
    return len(words)


def classify(
    spell: SpellChecker,
    term: str,
    lower_case: bool,
) -> Tuple[int, float]:
    """Return binary prediction and graded ranking score.

    The binary decision is positive when any token is absent from the extended
    dictionary. For AUC, known terms receive zero and misspelled terms receive
    the full character length of the term.
    """

    tokens = tokenize(term, lower_case)
    unknown_tokens = [token for token in tokens if spell[token] == 0]
    prediction = int(bool(unknown_tokens))
    score = float(len(term)) if unknown_tokens else 0.0
    return prediction, score


def main() -> int:
    args = parse_args()
    lower_case = not args.preserve_case
    dictionary_data_dir = args.dictionary_data_dir or args.data_dir
    train_path = Path(args.train_file)
    dev_path = Path(args.dev_file)
    test_path = Path(args.test_file)
    train = read_labeled_file(train_path if train_path.is_absolute() else dictionary_data_dir / train_path)
    dev = read_labeled_file(dev_path if dev_path.is_absolute() else dictionary_data_dir / dev_path)
    test = read_labeled_file(test_path if test_path.is_absolute() else args.data_dir / test_path)
    spell = SpellChecker(distance=args.distance)
    added = add_correct_terms(spell, train + dev, lower_case)

    predictions: List[int] = []
    scores: List[float] = []
    for example in tqdm(test, desc="SpellChecker inference"):
        prediction, score = classify(spell, example.term, lower_case)
        predictions.append(prediction)
        scores.append(score)
    targets = np.asarray([example.target for example in test], dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    metrics = binary_metrics(targets, prediction_array, score_array, include_brier=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "index": np.arange(len(test), dtype=np.int64),
            "term": [example.term for example in test],
            "target": targets,
            "prediction": prediction_array,
            "ranking_score": score_array,
        }
    ).to_csv(args.output_dir / "test_predictions.csv", index=False)
    np.savetxt(args.output_dir / "test_predict.txt", prediction_array, fmt="%d")
    np.savetxt(args.output_dir / "test_logits.txt", np.column_stack([1 - prediction_array, prediction_array]), fmt="%.8f")
    np.savetxt(args.output_dir / "test_edit_distance.txt", score_array, fmt="%.8f")
    np.savetxt(args.output_dir / "test_targets.txt", targets, fmt="%d")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "data_dir": str(args.data_dir),
                "dictionary_data_dir": str(dictionary_data_dir),
                "train_file": args.train_file,
                "dev_file": args.dev_file,
                "test_file": args.test_file,
                "distance": args.distance,
                "lower_case": lower_case,
                "dictionary_tokens_added": added,
                "threshold": "unknown token present",
                "ranking_score": "term_length",
                "auc_score": "0 for known terms; full term character length for misspelled terms",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, allow_nan=True))
    print(f"Saved SpellChecker outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
