# MIT License
# Copyright (c) 2026 Jinying Chen
#  
# author(s): Jinying Chen, Boston University Chobanian & Avedisian School of Medicine
# date: 2026-8-15
# ver: 1.0
# 
# This code was written to support model evaluation for the 2026 paper published 
# in JMIR Medical Informatics. 
# The code is for research use only, and is provided as it is.
# 
# See LICENSE in the project root for license terms.

#!/usr/bin/env python3
"""Analyze term-level model errors from one prediction file."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUT_FILE = "results/fullset_filtered.csv"
DEFAULT_OUTPUT_FILE = "results/error_analysis_results/fullset_filtered_model_errors_cleaned.csv"
PREDICTION_SUFFIX = "__pred_default"
CLEANED_CATEGORIES = {
    "correct drug name",
    "correct short drug name",
    "misspelled drug name",
    "mispelled drug name",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a model-output CSV, add model-specific correctness columns, "
            "and flag terms predicted incorrectly or correctly by all models. "
            "For each model, correct/easy means prediction equals target; "
            "incorrect/hard means prediction differs from target."
        )
    )
    parser.add_argument(
        "--input_file",
        default=DEFAULT_INPUT_FILE,
        help="Input CSV file.",
    )
    parser.add_argument(
        "--output_file",
        default=DEFAULT_OUTPUT_FILE,
        help="Output CSV file.",
    )
    parser.add_argument(
        "--output_prefix",
        default=None,
        help=(
            "Prefix for summary outputs. Defaults to the output file path without "
            "its suffix."
        ),
    )
    parser.add_argument(
        "--subset",
        choices=["all", "cleaned"],
        default="cleaned",
        help=(
            "Terms to analyze. 'all' keeps every row. 'cleaned' keeps rows whose "
            "category is correct drug name, correct short drug name, or misspelled "
            "drug name. Default: %(default)s."
        ),
    )
    return parser.parse_args()


def infer_output_prefix(output_file: Path, output_prefix: str | None) -> Path:
    if output_prefix:
        return Path(output_prefix)
    return output_file.with_suffix("")


def normalize_label(value: object) -> str:
    text = str(value or "").strip()
    try:
        numeric_value = float(text)
    except ValueError:
        return text
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return str(numeric_value)


def normalize_text(value: object, empty_label: str = "EMPTY") -> str:
    text = str(value or "").strip()
    return text if text else empty_label


def normalize_final_type(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "EMPTY"
    parts = [part.strip().upper() for part in re.split(r"[,;/|]+", text) if part.strip()]
    return ",".join(sorted(set(parts))) if parts else "EMPTY"


def get_prediction_columns(fieldnames: list[str] | None) -> list[str]:
    return [
        column
        for column in (fieldnames or [])
        if column.endswith(PREDICTION_SUFFIX)
    ]


def model_name_from_prediction_column(column: str) -> str:
    return column[: -len(PREDICTION_SUFFIX)]


def require_columns(fieldnames: list[str] | None, required_columns: list[str]) -> None:
    fieldname_set = set(fieldnames or [])
    missing = [column for column in required_columns if column not in fieldname_set]
    if missing:
        raise ValueError("Input file is missing required column(s): " + ", ".join(missing))


def append_missing(fieldnames: list[str], columns: list[str]) -> list[str]:
    output_fieldnames = list(fieldnames)
    for column in columns:
        if column not in output_fieldnames:
            output_fieldnames.append(column)
    return output_fieldnames


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def percent(count: int, denominator: int) -> float:
    return round(count / denominator * 100, 3) if denominator else 0.0


def percent_text(count: int, denominator: int) -> str:
    return f"{count} ({percent(count, denominator):.1f}%)"


def difficulty_tag(row: dict[str, str]) -> str:
    tag = normalize_text(row.get("difficulty_tag"), empty_label="other")
    return tag if tag in {"easy_for_all", "hard_for_all"} else "other"


def category_value(row: dict[str, str]) -> str:
    return normalize_text(row.get("category"))


def normalized_category_for_filter(row: dict[str, str]) -> str:
    return str(row.get("category", "")).strip().lower()


def keep_row_for_subset(row: dict[str, str], subset: str) -> bool:
    if subset == "all":
        return True
    if subset == "cleaned":
        return normalized_category_for_filter(row) in CLEANED_CATEGORIES
    raise ValueError(f"Unsupported subset: {subset}")


def final_type_value(row: dict[str, str]) -> str:
    if "final_type" in row:
        return normalize_final_type(row.get("final_type"))
    return normalize_final_type(row.get("type_final"))


def term_value(row: dict[str, str]) -> str:
    for column in ["medication_name", "term", "name", "name_norm"]:
        term = str(row.get(column, "")).strip()
        if term:
            return term
    return ""


def unique_example_terms(rows: list[dict[str, str]], limit: int | None = 10) -> list[str]:
    examples = []
    seen = set()
    for row in rows:
        term = term_value(row)
        if not term or term in seen:
            continue
        examples.append(term)
        seen.add(term)
        if limit is not None and len(examples) >= limit:
            break
    return examples


def build_difficulty_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    total_rows = len(rows)
    counts = Counter(difficulty_tag(row) for row in rows)
    return [
        {
            "difficulty_tag": tag,
            "count": counts[tag],
            "percent": percent(counts[tag], total_rows),
        }
        for tag in ["easy_for_all", "hard_for_all", "other"]
    ]


def build_model_correctness_summary(
    rows: list[dict[str, str]],
    model_names: list[str],
) -> list[dict[str, object]]:
    summary_rows = []
    total_rows = len(rows)
    for model_name in model_names:
        correct_column = f"{model_name}__correct"
        correct_count = sum(row.get(correct_column) == "1" for row in rows)
        incorrect_count = sum(row.get(correct_column) == "0" for row in rows)
        summary_rows.append(
            {
                "model": model_name,
                "correct_count": correct_count,
                "correct_percent": percent(correct_count, total_rows),
                "incorrect_count": incorrect_count,
                "incorrect_percent": percent(incorrect_count, total_rows),
            }
        )
    return summary_rows


def build_model_difficulty_summary(
    rows: list[dict[str, str]],
    model_names: list[str],
) -> list[dict[str, object]]:
    summary_rows = []
    total_rows = len(rows)
    for model_name in model_names:
        difficulty_column = f"{model_name}__difficulty"
        counts = Counter(row.get(difficulty_column, "unknown") or "unknown" for row in rows)
        summary_rows.append(
            {
                "model": model_name,
                "easy_count": counts["easy"],
                "easy_percent": percent(counts["easy"], total_rows),
                "hard_count": counts["hard"],
                "hard_percent": percent(counts["hard"], total_rows),
                "middle_count": counts["middle"],
                "middle_percent": percent(counts["middle"], total_rows),
                "unknown_count": counts["unknown"],
                "unknown_percent": percent(counts["unknown"], total_rows),
            }
        )
    return summary_rows


def build_category_by_difficulty_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows_by_tag = defaultdict(list)
    for row in rows:
        rows_by_tag[difficulty_tag(row)].append(row)

    summary_rows = []
    for tag in ["easy_for_all", "hard_for_all", "other"]:
        tag_rows = rows_by_tag[tag]
        denominator = len(tag_rows)
        counts = Counter(category_value(row) for row in tag_rows)
        for category, count in sorted(counts.items()):
            summary_rows.append(
                {
                    "difficulty_tag": tag,
                    "category": category,
                    "count": count,
                    "percent_within_difficulty_tag": percent(count, denominator),
                }
            )
    return summary_rows


def build_category_type_by_difficulty_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    nested_rows = defaultdict(list)
    for row in rows:
        nested_rows[(difficulty_tag(row), category_value(row))].append(row)

    summary_rows = []
    for tag in ["easy_for_all", "hard_for_all", "other"]:
        categories = sorted({category for row_tag, category in nested_rows if row_tag == tag})
        for category in categories:
            subset_rows = nested_rows[(tag, category)]
            denominator = len(subset_rows)
            counts = Counter(final_type_value(row) for row in subset_rows)
            for final_type, count in sorted(counts.items()):
                summary_rows.append(
                    {
                        "difficulty_tag": tag,
                        "category": category,
                        "final_type": final_type,
                        "count": count,
                        "percent_within_category_and_difficulty_tag": percent(
                            count,
                            denominator,
                        ),
                    }
                )
    return summary_rows


def is_model_correct(row: dict[str, str], model_name: str) -> bool:
    return row.get(f"{model_name}__correct") == "1"


def classify_model_difficulty(is_correct: bool) -> str:
    return "easy" if is_correct else "hard"


def build_spellchecker_bert_contrast_summary(
    rows: list[dict[str, str]],
    model_names: list[str],
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, str]]]]:
    required_models = {"SpellChecker", "BERTDrug", "CharBERTDrug"}
    if not required_models.issubset(set(model_names)):
        missing = sorted(required_models - set(model_names))
        print(
            "Warning: skipped SpellChecker/BERT contrast statistics because missing model(s): "
            + ", ".join(missing)
        )
        return [], {}

    contrast_groups = {
        "easy_spellchecker_hard_bert_and_charbert": [
            row
            for row in rows
            if is_model_correct(row, "SpellChecker")
            and not is_model_correct(row, "BERTDrug")
            and not is_model_correct(row, "CharBERTDrug")
        ],
        "hard_spellchecker_easy_bert_and_charbert": [
            row
            for row in rows
            if not is_model_correct(row, "SpellChecker")
            and is_model_correct(row, "BERTDrug")
            and is_model_correct(row, "CharBERTDrug")
        ],
    }

    summary_rows = []
    for contrast_group, group_rows in contrast_groups.items():
        denominator = len(group_rows)
        counts = Counter(category_value(row) for row in group_rows)
        if not counts:
            summary_rows.append(
                {
                    "contrast_group": contrast_group,
                    "category": "ALL",
                    "count": 0,
                    "percent_within_contrast_group": 0.0,
                }
            )
            continue
        for category, count in sorted(counts.items()):
            summary_rows.append(
                {
                    "contrast_group": contrast_group,
                    "category": category,
                    "count": count,
                    "percent_within_contrast_group": percent(count, denominator),
                }
            )
    return summary_rows, contrast_groups


def rows_matching(
    rows: list[dict[str, str]],
    *,
    tag: str | None = None,
    category: str | None = None,
    final_type: str | None = None,
) -> list[dict[str, str]]:
    matched = []
    for row in rows:
        if tag is not None and difficulty_tag(row) != tag:
            continue
        if category is not None and category_value(row) != category:
            continue
        if final_type is not None and final_type_value(row) != final_type:
            continue
        matched.append(row)
    return matched


def write_human_readable_summary(
    rows: list[dict[str, str]],
    subset: str,
    input_row_count: int,
    difficulty_summary: list[dict[str, object]],
    model_correctness_summary: list[dict[str, object]],
    model_difficulty_summary: list[dict[str, object]],
    category_by_difficulty: list[dict[str, object]],
    category_type_by_difficulty: list[dict[str, object]],
    spellchecker_bert_contrast: list[dict[str, object]],
    contrast_groups: dict[str, list[dict[str, str]]],
    output_prefix: Path,
) -> Path:
    output_path = Path(f"{output_prefix}_summary.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    category_type_lookup = defaultdict(list)
    for row in category_type_by_difficulty:
        category_type_lookup[(row["difficulty_tag"], row["category"])].append(row)

    contrast_labels = {
        "easy_spellchecker_hard_bert_and_charbert": (
            "Easy for SpellChecker, hard for BERTDrug and CharBERTDrug"
        ),
        "hard_spellchecker_easy_bert_and_charbert": (
            "Hard for SpellChecker, easy for BERTDrug and CharBERTDrug"
        ),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Model Error Pattern Summary\n")
        handle.write("=" * 80 + "\n\n")
        handle.write(f"Subset: {subset}\n")
        handle.write(f"Input rows read: {input_row_count}\n")
        handle.write(f"Total terms analyzed: {len(rows)}\n\n")

        handle.write("Difficulty Tag Definition\n")
        handle.write("-" * 80 + "\n")
        handle.write(
            "This single-prediction analysis defines easy_for_all as terms predicted "
            "correctly by every discovered model and hard_for_all as terms predicted "
            "incorrectly by every discovered model. For each individual model, easy "
            "means the model prediction equals the target label, and hard means the "
            "model prediction differs from the target label.\n\n"
        )

        handle.write("Model Correctness Breakdown\n")
        handle.write("-" * 80 + "\n")
        for row in model_correctness_summary:
            handle.write(
                f"{row['model']}: correct {row['correct_count']} "
                f"({row['correct_percent']:.1f}%), incorrect {row['incorrect_count']} "
                f"({row['incorrect_percent']:.1f}%)\n"
            )
        handle.write("\n")

        handle.write("Model Difficulty Breakdown\n")
        handle.write("-" * 80 + "\n")
        for row in model_difficulty_summary:
            handle.write(
                f"{row['model']}: easy {row['easy_count']} "
                f"({row['easy_percent']:.1f}%), hard {row['hard_count']} "
                f"({row['hard_percent']:.1f}%), middle {row['middle_count']} "
                f"({row['middle_percent']:.1f}%), unknown {row['unknown_count']} "
                f"({row['unknown_percent']:.1f}%)\n"
            )
        handle.write("\n")

        handle.write("Difficulty Tag Breakdown\n")
        handle.write("-" * 80 + "\n")
        for row in difficulty_summary:
            handle.write(f"{row['difficulty_tag']}: {row['count']} ({row['percent']:.1f}%)\n")
        handle.write("\n")

        handle.write("Category Breakdown Within Each Difficulty Tag\n")
        handle.write("-" * 80 + "\n")
        for tag in ["easy_for_all", "hard_for_all", "other"]:
            tag_summary = [row for row in category_by_difficulty if row["difficulty_tag"] == tag]
            if not tag_summary:
                continue
            total_count = sum(int(row["count"]) for row in tag_summary)
            handle.write(f"{tag} ({total_count} terms)\n")
            for row in tag_summary:
                category = str(row["category"])
                handle.write(
                    f"  {category}: "
                    f"{row['count']} ({row['percent_within_difficulty_tag']:.1f}%)\n"
                )
                category_examples = unique_example_terms(
                    rows_matching(rows, tag=tag, category=category),
                    limit=10,
                )
                if category_examples:
                    handle.write(f"    examples: {', '.join(category_examples)}\n")
                for type_row in category_type_lookup[(tag, category)]:
                    final_type = str(type_row["final_type"])
                    handle.write(
                        f"    {final_type}: "
                        f"{type_row['count']} "
                        f"({type_row['percent_within_category_and_difficulty_tag']:.1f}%)\n"
                    )
                    example_limit = 10
                    if (
                        tag == "hard_for_all"
                        and category == "correct drug name"
                        and final_type in {"BN", "BN,IN"}
                    ):
                        example_limit = None
                    examples = unique_example_terms(
                        rows_matching(
                            rows,
                            tag=tag,
                            category=category,
                            final_type=final_type,
                        ),
                        limit=example_limit,
                    )
                    if examples:
                        handle.write(f"      examples: {', '.join(examples)}\n")
            handle.write("\n")

        handle.write("SpellChecker vs BERT Contrast Category Breakdown\n")
        handle.write("-" * 80 + "\n")
        if not spellchecker_bert_contrast:
            handle.write("No SpellChecker/BERT contrast rows were generated.\n")
        else:
            for contrast_group in [
                "easy_spellchecker_hard_bert_and_charbert",
                "hard_spellchecker_easy_bert_and_charbert",
            ]:
                summary_rows = [
                    row
                    for row in spellchecker_bert_contrast
                    if row["contrast_group"] == contrast_group
                ]
                if not summary_rows:
                    continue
                total_count = sum(int(row["count"]) for row in summary_rows)
                handle.write(f"{contrast_labels[contrast_group]} ({total_count} terms)\n")
                group_rows = contrast_groups.get(contrast_group, [])
                for row in summary_rows:
                    category = str(row["category"])
                    handle.write(
                        f"  {category}: "
                        f"{row['count']} ({row['percent_within_contrast_group']:.1f}%)\n"
                    )
                    examples = unique_example_terms(
                        [
                            group_row
                            for group_row in group_rows
                            if category_value(group_row) == category
                        ],
                        limit=10,
                    )
                    if examples:
                        handle.write(f"    examples: {', '.join(examples)}\n")
                handle.write("\n")

    return output_path


def main() -> None:
    args = parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    output_prefix = infer_output_prefix(output_file, args.output_prefix)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    with input_file.open(newline="", encoding="utf-8-sig") as input_handle:
        reader = csv.DictReader(input_handle)
        fieldnames = list(reader.fieldnames or [])
        require_columns(fieldnames, ["medication_name", "target"])

        prediction_columns = get_prediction_columns(fieldnames)
        if not prediction_columns:
            raise ValueError(f"No columns ending with {PREDICTION_SUFFIX!r} were found.")

        correct_columns = [
            f"{model_name_from_prediction_column(column)}__correct"
            for column in prediction_columns
        ]
        model_names = [
            model_name_from_prediction_column(column)
            for column in prediction_columns
        ]
        easy_columns = [f"{model_name}__easy" for model_name in model_names]
        hard_columns = [f"{model_name}__hard" for model_name in model_names]
        difficulty_columns = [f"{model_name}__difficulty" for model_name in model_names]
        output_fieldnames = append_missing(
            fieldnames,
            (
                correct_columns
                + easy_columns
                + hard_columns
                + difficulty_columns
                + ["difficult_term", "easy_term", "difficulty_tag", "final_type"]
            ),
        )

        total_rows = 0
        analyzed_rows = 0
        difficult_rows = 0
        easy_rows = 0
        processed_rows = []
        with output_file.open("w", newline="", encoding="utf-8") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=output_fieldnames)
            writer.writeheader()

            for row in reader:
                total_rows += 1
                if not keep_row_for_subset(row, args.subset):
                    continue
                analyzed_rows += 1
                target = normalize_label(row.get("target", ""))
                easy_values = []
                hard_values = []

                for prediction_column, correct_column, model_name in zip(
                    prediction_columns,
                    correct_columns,
                    model_names,
                ):
                    prediction = normalize_label(row.get(prediction_column, ""))
                    is_correct = prediction == target
                    row[correct_column] = "1" if is_correct else "0"
                    model_difficulty = classify_model_difficulty(is_correct)
                    row[f"{model_name}__difficulty"] = model_difficulty
                    row[f"{model_name}__easy"] = "1" if model_difficulty == "easy" else "0"
                    row[f"{model_name}__hard"] = "1" if model_difficulty == "hard" else "0"
                    easy_values.append(model_difficulty == "easy")
                    hard_values.append(model_difficulty == "hard")

                row["difficult_term"] = "1" if all(hard_values) else "0"
                row["easy_term"] = "1" if all(easy_values) else "0"
                if row["easy_term"] == "1":
                    row["difficulty_tag"] = "easy_for_all"
                elif row["difficult_term"] == "1":
                    row["difficulty_tag"] = "hard_for_all"
                else:
                    row["difficulty_tag"] = "other"

                row["final_type"] = final_type_value(row)

                if row["difficult_term"] == "1":
                    difficult_rows += 1
                if row["easy_term"] == "1":
                    easy_rows += 1

                writer.writerow(row)
                processed_rows.append({fieldname: row.get(fieldname, "") for fieldname in output_fieldnames})

    flags_path = Path(f"{output_prefix}_model_error_flags.tsv")
    write_tsv(flags_path, processed_rows, output_fieldnames)

    model_correctness_summary = build_model_correctness_summary(processed_rows, model_names)
    model_correctness_path = Path(f"{output_prefix}_model_correctness_summary.tsv")
    write_tsv(
        model_correctness_path,
        model_correctness_summary,
        ["model", "correct_count", "correct_percent", "incorrect_count", "incorrect_percent"],
    )

    model_difficulty_summary = build_model_difficulty_summary(processed_rows, model_names)
    model_difficulty_path = Path(f"{output_prefix}_model_difficulty_summary.tsv")
    write_tsv(
        model_difficulty_path,
        model_difficulty_summary,
        [
            "model",
            "easy_count",
            "easy_percent",
            "hard_count",
            "hard_percent",
            "middle_count",
            "middle_percent",
            "unknown_count",
            "unknown_percent",
        ],
    )

    difficulty_summary = build_difficulty_summary(processed_rows)
    difficulty_summary_path = Path(f"{output_prefix}_difficulty_tag_summary.tsv")
    write_tsv(difficulty_summary_path, difficulty_summary, ["difficulty_tag", "count", "percent"])

    category_by_difficulty = build_category_by_difficulty_summary(processed_rows)
    category_by_difficulty_path = Path(f"{output_prefix}_category_by_difficulty_tag_summary.tsv")
    write_tsv(
        category_by_difficulty_path,
        category_by_difficulty,
        ["difficulty_tag", "category", "count", "percent_within_difficulty_tag"],
    )

    category_type_by_difficulty = build_category_type_by_difficulty_summary(processed_rows)
    category_type_by_difficulty_path = Path(
        f"{output_prefix}_category_type_by_difficulty_tag_summary.tsv"
    )
    write_tsv(
        category_type_by_difficulty_path,
        category_type_by_difficulty,
        [
            "difficulty_tag",
            "category",
            "final_type",
            "count",
            "percent_within_category_and_difficulty_tag",
        ],
    )

    spellchecker_bert_contrast, contrast_groups = build_spellchecker_bert_contrast_summary(
        processed_rows,
        model_names,
    )
    spellchecker_bert_contrast_path = Path(
        f"{output_prefix}_spellchecker_bert_contrast_category_summary.tsv"
    )
    write_tsv(
        spellchecker_bert_contrast_path,
        spellchecker_bert_contrast,
        ["contrast_group", "category", "count", "percent_within_contrast_group"],
    )

    readable_summary_path = write_human_readable_summary(
        processed_rows,
        args.subset,
        total_rows,
        difficulty_summary,
        model_correctness_summary,
        model_difficulty_summary,
        category_by_difficulty,
        category_type_by_difficulty,
        spellchecker_bert_contrast,
        contrast_groups,
        output_prefix,
    )

    print(f"Detected {len(prediction_columns)} prediction column(s):")
    for column in prediction_columns:
        print(f"  {column} -> {model_name_from_prediction_column(column)}__correct")
    print(f"Subset: {args.subset}")
    print("Difficulty mode: correctness")
    print(f"Input rows read: {total_rows}")
    print(f"Rows analyzed after subset filtering: {analyzed_rows}")
    print(f"Wrote {analyzed_rows} rows to {output_file}")
    print(f"Wrote model error flags to {flags_path}")
    print(f"Wrote model correctness summary to {model_correctness_path}")
    print(f"Wrote model difficulty summary to {model_difficulty_path}")
    print(f"Wrote difficulty tag summary to {difficulty_summary_path}")
    print(f"Wrote category by difficulty tag summary to {category_by_difficulty_path}")
    print(f"Wrote category/final-type by difficulty tag summary to {category_type_by_difficulty_path}")
    print(f"Wrote SpellChecker/BERT contrast summary to {spellchecker_bert_contrast_path}")
    print(f"Wrote human-readable summary to {readable_summary_path}")
    print(f"Difficult terms: {difficult_rows}")
    print(f"Easy terms: {easy_rows}")


if __name__ == "__main__":
    main()
