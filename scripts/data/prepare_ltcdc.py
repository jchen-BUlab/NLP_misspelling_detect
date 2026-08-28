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
"""Convert the two reviewed LTCDC workbooks into model-ready files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, List

import pandas as pd


CLEANED_NEGATIVE_CATEGORIES = ["correct drug name", "correct short drug name"]
CLEANED_POSITIVE_CATEGORIES = ["mispelled drug name", "misspelled drug name"]
UNCLEANED_NEGATIVE_CATEGORIES = ["correct non-drug name"]
UNCLEANED_POSITIVE_CATEGORIES = [
    "mispelled non-drug name",
    "misspelled non-drug name",
    "not sure",
    "nonstandard",
]
CATEGORY_ALIASES = {"nonstandard2": "mispelled drug name"}
VALID_DATA_TYPES = {"oov", "non-oov_cleaned", "uncleaned"}
VALID_TERM_TYPES = {"generic", "brand"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Reviewed LTCDC .xlsx/.csv files in set 1, set 2 order.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Creates <prefix>.txt, <prefix>_metadata.csv, and <prefix>_manifest.json.",
    )
    parser.add_argument("--term-column", default="medication_name")
    parser.add_argument("--category-column", default="category")
    parser.add_argument("--type-column", default="type_final")
    parser.add_argument("--frequency-column", default="frequency")
    parser.add_argument("--closest-term-column", default="closest_term")
    parser.add_argument(
        "--data-type-column",
        default="data_type",
        help="Optional existing oov/non-oov_cleaned/uncleaned column.",
    )
    parser.add_argument(
        "--known-name-file",
        type=Path,
        nargs="*",
        default=[],
        help=(
            "Generated RxNorm labeled files used to identify LTCDC overlap. "
            "For the manuscript, pass train1.txt, train6.txt, and dev.txt."
        ),
    )
    parser.add_argument("--negative-category", action="append", default=[])
    parser.add_argument("--positive-category", action="append", default=[])
    parser.add_argument(
        "--cleaned-only",
        action="store_true",
        help="Exclude non-drug, not-sure, and nonstandard categories.",
    )
    parser.add_argument("--lower-case", action="store_true")
    parser.add_argument(
        "--allow-missing-data-type",
        action="store_true",
        help="Allow cleaned rows without OOV/non-OOV membership.",
    )
    parser.add_argument(
        "--allow-missing-term-type",
        action="store_true",
        help="Allow cleaned drug rows without generic/brand membership.",
    )
    return parser.parse_args()


def normalize(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def normalize_category(value: object) -> str:
    normalized = normalize(value)
    return CATEGORY_ALIASES.get(normalized, normalized)


def normalize_data_type(value: object) -> str:
    normalized = normalize(value).replace("_", " ")
    aliases = {
        "oov": "oov",
        "out of vocabulary": "oov",
        "non oov cleaned": "non-oov_cleaned",
        "non-oov cleaned": "non-oov_cleaned",
        "cleaned non oov": "non-oov_cleaned",
        "uncleaned": "uncleaned",
    }
    return aliases.get(normalized, normalized)


def normalize_term_type(value: object) -> str:
    normalized = normalize(value).replace("_", " ")
    compact = normalized.replace(" ", "")
    if not compact:
        return "generic"
    if compact in {
        "in",
        "generic",
        "ingredient",
        "non-branded",
        "nonbranded",
        "bn,in",
        "in,bn",
    }:
        return "generic"
    if compact in {"bn", "brand", "branded"}:
        return "brand"
    return normalized


def normalize_frequency(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"low", "high"}:
        return lowered
    try:
        numeric = float(text)
    except ValueError:
        return lowered
    if math.isnan(numeric):
        return ""
    return "low" if numeric <= 100 else "high"


def clean_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    raise ValueError(f"Unsupported annotation format: {path.suffix}")


def inferred_source_split(path: Path, position: int) -> str:
    name = path.name.lower()
    if "set1" in name or "single" in name:
        return "single"
    if "set2" in name or "multi" in name:
        return "multi"
    return f"source{position + 1}"


def read_sources(paths: Iterable[Path], term_column: str, category_column: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for position, path in enumerate(paths):
        frame = read_table(path).copy()
        required = {term_column, category_column}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        if "source_workbook" not in frame.columns:
            frame["source_workbook"] = path.name
        if "source_split" not in frame.columns:
            frame["source_split"] = inferred_source_split(path, position)
        if "source_index" not in frame.columns:
            frame["source_index"] = range(len(frame))
        if "source_rowid" not in frame.columns:
            if "rowid" in frame.columns:
                frame["source_rowid"] = frame["rowid"]
            else:
                frame["source_rowid"] = range(1, len(frame) + 1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def parse_labeled_term(line: str) -> list[str]:
    parts = line.strip().split()
    if not parts:
        return []
    if parts[0].lower().startswith("__label__"):
        return parts[1:]
    return parts


def load_known_names(paths: Iterable[Path]) -> set[str]:
    """Load full terms and tokens, matching the manuscript overlap filter."""

    known: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                tokens = [token.lower() for token in parse_labeled_term(line)]
                if not tokens:
                    continue
                known.update(tokens)
                known.add(" ".join(tokens))
    return known


def main() -> int:
    args = parse_args()
    frame = read_sources(args.input, args.term_column, args.category_column)

    frame["_term_norm"] = frame[args.term_column].map(normalize)
    empty_term_mask = frame["_term_norm"].eq("")
    empty_rows = int(empty_term_mask.sum())
    frame = frame.loc[~empty_term_mask].copy()
    duplicate_mask = frame.duplicated(subset=["_term_norm"], keep="first")
    duplicate_rows = int(duplicate_mask.sum())
    frame = frame.loc[~duplicate_mask].copy().reset_index(drop=True)

    default_negatives = list(CLEANED_NEGATIVE_CATEGORIES)
    default_positives = list(CLEANED_POSITIVE_CATEGORIES)
    if not args.cleaned_only:
        default_negatives.extend(UNCLEANED_NEGATIVE_CATEGORIES)
        default_positives.extend(UNCLEANED_POSITIVE_CATEGORIES)
    negatives = {normalize_category(value) for value in (args.negative_category or default_negatives)}
    positives = {normalize_category(value) for value in (args.positive_category or default_positives)}
    uncleaned_categories = {
        normalize_category(value)
        for value in UNCLEANED_NEGATIVE_CATEGORIES + UNCLEANED_POSITIVE_CATEGORIES
    }
    known_names = load_known_names(args.known_name_file)

    rows: List[dict] = []
    unresolved_categories = 0
    for _, record in frame.iterrows():
        term = clean_cell(record[args.term_column])
        term_norm = normalize(term)
        raw_category = normalize(record[args.category_column])
        category = normalize_category(record[args.category_column])
        if category in negatives:
            target = 0
        elif category in positives:
            target = 1
        else:
            unresolved_categories += 1
            continue
        if args.lower_case:
            term = term.lower()

        data_type = normalize_data_type(record.get(args.data_type_column, ""))
        if not data_type and category in uncleaned_categories:
            data_type = "uncleaned"
        elif not data_type and known_names:
            data_type = "non-oov_cleaned" if term_norm in known_names else "oov"

        source_split = clean_cell(record.get("source_split", ""))
        source_index = clean_cell(record.get("source_index", ""))
        sample_key = clean_cell(record.get("sample_key", ""))
        if not sample_key:
            sample_key = f"{source_split}|{source_index}|{term_norm}"
        frequency_raw = clean_cell(record.get(args.frequency_column, ""))
        rows.append(
            {
                "index": len(rows),
                "sample_key": sample_key,
                "source_workbook": clean_cell(record.get("source_workbook", "")),
                "source_split": source_split,
                "source_index": source_index,
                "source_rowid": clean_cell(record.get("source_rowid", "")),
                "term": term,
                "target": target,
                "target_label": "positive" if target else "negative",
                "category": category,
                "raw_category": raw_category,
                "term_type": normalize_term_type(record.get(args.type_column, "")),
                "frequency": normalize_frequency(record.get(args.frequency_column, "")),
                "frequency_raw": frequency_raw,
                "data_type": data_type,
                "closest_term": clean_cell(record.get(args.closest_term_column, "")),
            }
        )

    if not rows:
        raise ValueError("No rows remained after category mapping")
    invalid_data_types = sorted(
        {row["data_type"] for row in rows if row["data_type"]} - VALID_DATA_TYPES
    )
    if invalid_data_types:
        raise ValueError(
            "Unrecognized data_type values after normalization: "
            f"{invalid_data_types}. Expected {sorted(VALID_DATA_TYPES)}."
        )
    missing_data_type = sum(not row["data_type"] for row in rows)
    if missing_data_type and not args.allow_missing_data_type:
        raise ValueError(
            f"{missing_data_type} cleaned rows have no OOV membership. Pass the generated "
            "RxNorm train1.txt, train6.txt, and dev.txt with --known-name-file, or use "
            "--allow-missing-data-type for a non-manuscript export."
        )
    invalid_term_types = sorted(
        {
            row["term_type"]
            for row in rows
            if row["category"] not in uncleaned_categories and row["term_type"]
        }
        - VALID_TERM_TYPES
    )
    if invalid_term_types:
        raise ValueError(
            "Unrecognized term_type values after normalization: "
            f"{invalid_term_types}. Expected {sorted(VALID_TERM_TYPES)}."
        )
    missing_cleaned_term_type = sum(
        row["category"] not in uncleaned_categories and not row["term_type"] for row in rows
    )
    if missing_cleaned_term_type and not args.allow_missing_term_type:
        raise ValueError(
            f"{missing_cleaned_term_type} cleaned drug rows have no generic/brand term type. "
            "Supply type_final or pass --allow-missing-term-type."
        )

    output_txt = args.output_prefix.with_suffix(".txt")
    output_csv = args.output_prefix.with_name(args.output_prefix.name + "_metadata.csv")
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(f"__label__{row['target_label']} {row['term']}\n")
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    excluded_rows = empty_rows + duplicate_rows + unresolved_categories
    manifest = {
        "inputs": [
            {"name": path.name, "sha256": sha256_file(path), "rows": int(len(read_table(path)))}
            for path in args.input
        ],
        "known_name_files": [
            {"name": path.name, "sha256": sha256_file(path)} for path in args.known_name_file
        ],
        "output_text": output_txt.name,
        "output_metadata": output_csv.name,
        "included_rows": len(rows),
        "excluded_rows": excluded_rows,
        "empty_term_rows": empty_rows,
        "duplicate_rows": duplicate_rows,
        "unresolved_category_rows": unresolved_categories,
        "negative_categories": sorted(negatives),
        "positive_categories": sorted(positives),
        "profile": "cleaned_only" if args.cleaned_only else "manuscript_full",
        "missing_data_type_rows": missing_data_type,
        "missing_cleaned_term_type_rows": missing_cleaned_term_type,
    }
    manifest_path = args.output_prefix.with_name(args.output_prefix.name + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows):,} labeled terms to {output_txt}")
    print(
        f"Wrote metadata to {output_csv}; removed {duplicate_rows:,} duplicates and "
        f"excluded {empty_rows + unresolved_categories:,} empty/unmapped rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
