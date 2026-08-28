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
"""Create the RxNorm-derived train/dev/test files used by the paper.

The output format is one example per line::

    __label__negative correctly spelled medication
    __label__positive misspeled medication

For training multiplier ``k``, every correctly spelled training term is
repeated ``k`` times and paired with ``k`` independently generated
misspellings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


RELEASE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = RELEASE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drug_spelling.io import LabeledExample, write_labeled_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brand-file", type=Path, required=True, help="RxNorm branded-name file, one name per line.")
    parser.add_argument("--generic-file", type=Path, required=True, help="RxNorm generic/ingredient-name file, one name per line.")
    parser.add_argument("--output-dir", type=Path, default=RELEASE_ROOT / "data" / "classification" / "rxnorm")
    parser.add_argument("--multipliers", default="1,2,4,6,8,10", help="Comma-separated numbers of typos per training term.")
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--augmentation-seed", type=int, default=0)
    parser.add_argument("--max-word-uses", type=int, default=10)
    return parser.parse_args()


def slash_is_literal(text: str, index: int) -> bool:
    """Return whether a slash belongs to a dosage or common abbreviation."""

    before = text[max(0, index - 4) : index].lower()
    after = text[index + 1 : index + 3].lower()
    if before.endswith("+") and after.startswith("-"):
        return True
    if before.endswith("w"):
        return True
    if index > 0 and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit():
        return True
    if before.endswith("195") and after.startswith("p"):
        return True
    return False


def split_rxnorm_name(raw_name: str) -> List[str]:
    """Split top-level RxNorm slash alternatives while preserving literal slashes."""

    text = raw_name.strip().strip("'\"").lower()
    pieces: List[str] = []
    buffer: List[str] = []
    depth = 0
    for index, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")" and depth:
            depth -= 1
        if character == "/" and depth == 0 and not slash_is_literal(text, index):
            candidate = "".join(buffer).strip()
            if len(candidate) >= 2:
                pieces.append(candidate)
            buffer = []
        else:
            buffer.append(character)
    candidate = "".join(buffer).strip()
    if len(candidate) >= 2:
        pieces.append(candidate)
    return pieces


def read_terms(path: Path, term_type: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    seen = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            for term in split_rxnorm_name(raw_line):
                normalized = " ".join(term.split())
                if normalized and normalized not in seen:
                    records.append((normalized, term_type))
                    seen.add(normalized)
    if not records:
        raise ValueError(f"No terms found in {path}")
    return records


def build_augmenter():
    """Build the five TextAttack transformations used in the original code."""

    try:
        from textattack.augmentation import Augmenter
        from textattack.constraints.pre_transformation import MinWordLength
        from textattack.transformations import (
            CompositeTransformation,
            WordSwapNeighboringCharacterSwap,
            WordSwapQWERTY,
            WordSwapRandomCharacterDeletion,
            WordSwapRandomCharacterInsertion,
            WordSwapRandomCharacterSubstitution,
        )
    except ImportError as exc:
        raise RuntimeError("TextAttack is required for typo generation; install requirements-training.txt") from exc

    class FixedWordSwapQWERTY(WordSwapQWERTY):
        """QWERTY substitution that also handles characters absent from the map."""

        def _get_replacement_words(self, word):
            if len(word) <= 1:
                return []
            start = 1 if self.skip_first_char else 0
            end = len(word) - (1 + self.skip_last_char)
            if start > end:
                return []
            index = random.randrange(start, end + 1)
            adjacent = self._get_adjacent(word[index])
            replacement = random.choice(adjacent) if adjacent else random.choice(list(self._keyboard_adjacency))
            return [word[:index] + replacement + word[index + 1 :]]

    transformations = CompositeTransformation(
        [
            WordSwapRandomCharacterDeletion(),
            WordSwapNeighboringCharacterSwap(),
            WordSwapRandomCharacterInsertion(),
            WordSwapRandomCharacterSubstitution(),
            FixedWordSwapQWERTY(),
        ]
    )
    return Augmenter(
        transformation=transformations,
        constraints=[MinWordLength(3)],
        pct_words_to_swap=0,
    )


def typo_for_word(word: str, augmenter, rng: random.Random) -> str:
    for _ in range(50):
        candidates = augmenter.augment(word)
        if candidates:
            candidate = str(candidates[0]).strip()
            if candidate and candidate.lower() != word.lower():
                return candidate
    # A deterministic last-resort deletion keeps the pipeline from hanging.
    index = rng.randrange(len(word))
    return word[:index] + word[index + 1 :]


def generate_typos(
    records: Sequence[Tuple[str, str]],
    count_per_term: int,
    augmenter,
    rng: random.Random,
    word_usage: Dict[str, int],
    max_word_uses: int,
) -> List[Tuple[str, str]]:
    generated: List[Tuple[str, str]] = []
    for term, term_type in records:
        source_words = term.split()
        for _ in range(count_per_term):
            eligible = [
                index
                for index, word in enumerate(source_words)
                if len(word) >= 3 and (word_usage[word] < max_word_uses or len(source_words) == 1)
            ]
            if not eligible:
                eligible = [index for index, word in enumerate(source_words) if len(word) >= 2]
            if not eligible:
                raise ValueError(f"Cannot introduce a character typo into {term!r}")
            word_index = rng.choice(eligible)
            words = list(source_words)
            words[word_index] = typo_for_word(words[word_index], augmenter, rng)
            word_usage[source_words[word_index]] += 1
            generated.append((" ".join(words).lower(), term_type))
    return generated


def write_metadata(path: Path, rows: Iterable[Tuple[str, str]], split: str, target: int) -> None:
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["term", "split", "target", "term_type"])
        if new_file:
            writer.writeheader()
        for term, term_type in rows:
            writer.writerow({"term": term, "split": split, "target": target, "term_type": term_type})


def write_aligned_metadata(path: Path, rows: Sequence[Tuple[str, str]], targets: Sequence[int], split: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["index", "sample_key", "term", "target", "target_label", "term_type", "type_norm", "data_type"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, ((term, term_type), target) in enumerate(zip(rows, targets)):
            writer.writerow(
                {
                    "index": index,
                    "sample_key": f"{split}|{index}",
                    "term": term,
                    "target": target,
                    "target_label": "positive" if target else "negative",
                    "term_type": term_type,
                    "type_norm": "BN" if term_type == "brand" else "IN",
                    "data_type": "rxnorm",
                }
            )


def write_base_split_metadata(
    path: Path,
    split_records: Sequence[Tuple[str, Sequence[Tuple[str, str]]]],
) -> None:
    """Write every original correctly spelled RxNorm term exactly once."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["sample_key", "term", "split", "target", "target_label", "term_type"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split, records in split_records:
            for index, (term, term_type) in enumerate(records):
                writer.writerow(
                    {
                        "sample_key": f"rxnorm_original|{split}|{index}",
                        "term": term,
                        "split": split,
                        "target": 0,
                        "target_label": "negative",
                        "term_type": term_type,
                    }
                )


def repeat_training_records(
    records: Sequence[Tuple[str, str]],
    multiplier: int,
) -> List[Tuple[str, str]]:
    """Repeat correctly spelled training records to match typo multiplicity."""

    if multiplier < 1:
        raise ValueError("multiplier must be positive")
    return [record for _ in range(multiplier) for record in records]


def main() -> int:
    args = parse_args()
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for the RxNorm split") from exc

    multipliers = sorted({int(value) for value in args.multipliers.split(",") if value.strip()})
    if not multipliers or multipliers[0] < 1:
        raise ValueError("--multipliers must contain positive integers")

    records = read_terms(args.brand_file, "brand") + read_terms(args.generic_file, "generic")
    terms = [record[0] for record in records]
    if len(terms) != len(set(terms)):
        # Prefer generic when a term appears in both input lists.
        deduplicated = {}
        for term, term_type in records:
            deduplicated[term] = "generic" if term_type == "generic" else deduplicated.get(term, term_type)
        records = list(deduplicated.items())

    train_dev, test = train_test_split(records, test_size=0.2, random_state=args.split_seed)
    train, dev = train_test_split(train_dev, test_size=0.25, random_state=args.split_seed)

    random.seed(args.augmentation_seed)
    np.random.seed(args.augmentation_seed)
    rng = random.Random(args.augmentation_seed)
    augmenter = build_augmenter()
    word_usage: Dict[str, int] = defaultdict(int)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_base_split_metadata(
        args.output_dir / "base_term_splits.csv",
        [("train", train), ("development", dev), ("test", test)],
    )
    metadata_path = args.output_dir / "term_metadata.csv"
    if metadata_path.exists():
        metadata_path.unlink()

    dev_positive = generate_typos(dev, 1, augmenter, rng, word_usage, args.max_word_uses)
    test_positive = generate_typos(test, 1, augmenter, rng, word_usage, args.max_word_uses)
    dev_examples = [LabeledExample(term, 0) for term, _ in dev] + [LabeledExample(term, 1) for term, _ in dev_positive]
    test_examples = [LabeledExample(term, 0) for term, _ in test] + [LabeledExample(term, 1) for term, _ in test_positive]
    write_labeled_file(args.output_dir / "dev.txt", dev_examples)
    write_labeled_file(args.output_dir / "test.txt", test_examples)
    write_aligned_metadata(
        args.output_dir / "dev_metadata.csv",
        list(dev) + list(dev_positive),
        [0] * len(dev) + [1] * len(dev_positive),
        "dev",
    )
    write_aligned_metadata(
        args.output_dir / "test_metadata.csv",
        list(test) + list(test_positive),
        [0] * len(test) + [1] * len(test_positive),
        "test",
    )
    write_metadata(metadata_path, dev, "dev", 0)
    write_metadata(metadata_path, dev_positive, "dev", 1)
    write_metadata(metadata_path, test, "test", 0)
    write_metadata(metadata_path, test_positive, "test", 1)

    for multiplier in multipliers:
        positives = generate_typos(train, multiplier, augmenter, rng, word_usage, args.max_word_uses)
        negative_rows = repeat_training_records(train, multiplier)
        examples = [
            LabeledExample(term, 0)
            for term, _term_type in negative_rows
        ]
        examples.extend(LabeledExample(term, 1) for term, _term_type in positives)
        write_labeled_file(args.output_dir / f"train{multiplier}.txt", examples)
        write_metadata(
            metadata_path,
            negative_rows,
            f"train{multiplier}",
            0,
        )
        write_metadata(metadata_path, positives, f"train{multiplier}", 1)
        print(
            f"train{multiplier}.txt: {len(examples):,} rows "
            f"({len(positives):,} positive, {len(negative_rows):,} negative)"
        )

    manifest = {
        "brand_input": {"name": args.brand_file.name, "sha256": sha256_file(args.brand_file)},
        "generic_input": {"name": args.generic_file.name, "sha256": sha256_file(args.generic_file)},
        "unique_original_terms": len(records),
        "split_rows": {"train": len(train), "development": len(dev), "test": len(test)},
        "multipliers": multipliers,
        "negative_repeats_per_term": "training multiplier",
        "split_seed": args.split_seed,
        "augmentation_seed": args.augmentation_seed,
        "max_word_uses": args.max_word_uses,
        "augmentation_methods": [
            "random_character_deletion",
            "adjacent_character_swap",
            "random_character_insertion",
            "random_character_substitution",
            "qwerty_adjacent_substitution",
        ],
    }
    (args.output_dir / "preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"dev.txt: {len(dev_examples):,} rows")
    print(f"test.txt: {len(test_examples):,} rows")
    print(f"Wrote RxNorm-derived data to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
