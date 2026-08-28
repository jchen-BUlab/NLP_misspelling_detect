#****************************************
# MIT License
# Copyright (c) 2026 Jinying Chen, Jiayu Lu
#
# author(s): Jinying Chen, Jiayu Lu, Boston University Chobanian & Avedisian School of Medicine
# date: 2026-8-15
# ver: 1.0
#
# This code was written to support model evaluation for the 2026 paper published
# in JMIR Medical Informatics.
# The code is for research use only, and is provided as it is.
#
# See LICENSE in the project root for license terms.

#!/usr/bin/env python
"""Bootstrap LTCDC model metrics and confidence intervals.

Reads LTCDC per-sample model scores, converts score columns to binary
predictions, and writes metric-level test-sample bootstrap summaries for:

- uncleaned: all rows
- cleaned: non-oov_cleaned + oov
- oov: oov only

Every bootstrap replicate samples with replacement within the positive and
negative classes, preserving the analysis subset's class ratio.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from bootstrap_statistics import stratified_bootstrap_f1_ci


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path("artifacts/predictions/ltcdc_predictions.csv")
FALLBACK_INPUT = DEFAULT_INPUT
DEFAULT_LTCDC_METADATA_DIR = Path("artifacts/data/ltcdc")

SUBSET_SPECS = OrderedDict(
    [
        ("uncleaned", None),
        ("cleaned", {"non-oov_cleaned", "oov"}),
        ("oov", {"oov"}),
    ]
)

CATEGORY_DISPLAY_NAMES = OrderedDict(
    [
        ("all_type", "all type"),
        ("generic", "generic"),
        ("branded", "branded"),
        ("low", "low"),
        ("high", "high"),
    ]
)

FREQUENCY_COLUMN_CANDIDATES = (
    "frequency",
    "frequency_raw",
    "freq",
    "frequency_group",
    "freq_group",
    "frequency_category",
    "frequency_level",
    "count_frequency",
    "count",
)
LOW_FREQUENCY_VALUES = {"low", "frequency_low", "low_frequency", "rare", "l"}
HIGH_FREQUENCY_VALUES = {"high", "frequency_high", "high_frequency", "frequent", "h"}
COUNT_LOW_THRESHOLD = 100.0

MODEL_SPECS = OrderedDict(
    [
        (
            "CharBERTDrug",
            {
                "score_column": "CharBERTDrug_probability",
                "score_type": "probability",
                "threshold_arg": "probability_threshold",
            },
        ),
        (
            "BERTDrug",
            {
                "score_column": "BERTDrug_probability",
                "score_type": "probability",
                "threshold_arg": "probability_threshold",
            },
        ),
        (
            "SpellChecker",
            {
                "score_column": "SpellChecker_edit_distance",
                "prediction_column": "SpellChecker_prediction",
                "score_type": "edit_distance",
            },
        ),
        (
            "fastTextML",
            {
                "score_column": "fasttext+xgboost_probability",
                "score_type": "probability",
                "threshold_arg": "probability_threshold",
            },
        ),
        (
            "BioWordVecML",
            {
                "score_column": "BioWordVec_probability",
                "score_type": "probability",
                "threshold_arg": "probability_threshold",
            },
        ),
    ]
)

TABLE_MODEL_ORDER = OrderedDict(
    [
        ("SpellChecker", "SpellChecker"),
        ("CharBERTDrug", "CharBERTDrug"),
        ("BERTDrug", "BERTDrug"),
        ("fastTextML", "fastTextML"),
        ("BioWordVecML", "BioWordVecML"),
    ]
)

METRIC_DISPLAY_NAMES = OrderedDict(
    [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("accuracy", "Accuracy"),
        ("specificity", "Specificity"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("brier_score", "Brier Score"),
    ]
)
THRESHOLD_METRICS = ("precision", "recall", "f1", "accuracy", "specificity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap model metrics for LTCDC samples."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV with target, data_type, and model score columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/results/bootstrap_ltcdc"),
        help="Directory where result CSVs are written.",
    )
    parser.add_argument(
        "--ltcdc-metadata-dir",
        type=Path,
        default=DEFAULT_LTCDC_METADATA_DIR,
        help=(
            "Directory containing annotated_test_metadata.csv, used for LTCDC "
            "type and high/low frequency categories."
        ),
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap samples.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for metric intervals.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=123,
        help="Random seed used for bootstrap sampling.",
    )
    parser.add_argument(
        "--metric-set",
        choices=["threshold", "all"],
        default="all",
        help=(
            "'threshold' computes precision/recall/F1/accuracy/specificity. "
            "'all' also computes ROC-AUC, PR-AUC, and Brier score."
        ),
    )
    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=0.5,
        help="Positive prediction threshold for probability score columns.",
    )
    parser.add_argument(
        "--skip-category-breakdowns",
        action="store_true",
        help="Only write the main uncleaned/cleaned/oov outputs; skip category_breakdowns folders.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages.",
    )
    return parser.parse_args()


def progress(args: argparse.Namespace, message: str) -> None:
    if not getattr(args, "quiet", False):
        print(message, flush=True)


def resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def resolve_input_path(project_root: Path, path: Path) -> Path:
    resolved = resolve_path(project_root, path)
    fallback = resolve_path(project_root, FALLBACK_INPUT)
    if not resolved.exists() and path == DEFAULT_INPUT and fallback.exists():
        return fallback
    return resolved


def metric_keys_for_args(args: argparse.Namespace) -> List[str]:
    if args.metric_set == "threshold":
        return list(THRESHOLD_METRICS)
    return list(METRIC_DISPLAY_NAMES.keys())


def parse_target(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "positive", "__label__positive", "true"}:
        return 1
    if text in {"0", "0.0", "negative", "__label__negative", "false"}:
        return 0
    raise ValueError(f"Could not parse target value: {value!r}")


def normalize_term(text: object) -> str:
    text = str(text)
    text = " ".join(text.strip().split())
    return text.lower()


def normalize_type(text: object) -> str:
    normalized = " ".join(str(text).strip().split())
    normalized = normalized.replace(" ,", ",").replace(", ", ",")
    if normalized == "BN,IN":
        return "BN, IN"
    return normalized


def normalize_frequency(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    lowered = text.lower()
    if lowered in {"low", "high"}:
        return lowered

    try:
        numeric_value = float(text)
    except ValueError:
        return lowered
    if math.isnan(numeric_value):
        return ""
    return "low" if numeric_value <= COUNT_LOW_THRESHOLD else "high"


def load_ltcdc_samples(input_path: Path) -> List[Dict[str, str]]:
    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} has no header row")

        required_columns = {"target", "data_type", "SpellChecker_prediction"}
        for spec in MODEL_SPECS.values():
            required_columns.add(str(spec["score_column"]))
        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"{input_path} is missing columns: {sorted(missing_columns)}")

        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError(f"No rows found in {input_path}")
    return rows


def metadata_record_from_row(row: Mapping[str, str], source_split: str, source_index: int) -> Dict[str, str]:
    frequency_raw = str(row.get("frequency_raw") or row.get("frequency", "")).strip()
    term = str(row.get("term") or row.get("medication_name", "")).strip()
    type_value = row.get("term_type") or row.get("type_final", "")
    return {
        "source_split": source_split,
        "source_index": str(source_index),
        "metadata_term": term,
        "metadata_term_norm": normalize_term(term),
        "category": str(row.get("category", "")).strip(),
        "type_norm": normalize_type(type_value),
        "frequency_raw": frequency_raw,
        "frequency": normalize_frequency(row.get("frequency", frequency_raw)),
    }


def load_ltcdc_metadata(metadata_dir: Path) -> Tuple[Dict[Tuple[str, int], Dict[str, str]], Dict[str, Dict[str, str]]]:
    metadata_by_source: Dict[Tuple[str, int], Dict[str, str]] = {}
    metadata_by_term: Dict[str, Dict[str, str]] = {}

    path = metadata_dir / "annotated_test_metadata.csv"
    if not path.exists():
        return metadata_by_source, metadata_by_term
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_split = str(row.get("source_split", "")).strip()
            try:
                source_index = int(str(row.get("source_index", "")).strip())
            except ValueError:
                continue
            record = metadata_record_from_row(row, source_split, source_index)
            metadata_by_source[(source_split, source_index)] = record
            if record["metadata_term_norm"]:
                metadata_by_term[record["metadata_term_norm"]] = record

    return metadata_by_source, metadata_by_term


def metadata_key_from_sample_key(sample_key: object) -> Optional[Tuple[str, int]]:
    parts = str(sample_key).split("|", 2)
    if len(parts) < 2:
        return None
    source_split = parts[0].strip()
    try:
        source_index = int(parts[1])
    except ValueError:
        return None
    return source_split, source_index


def find_ltcdc_metadata_for_row(
    row: Mapping[str, str],
    metadata_by_source: Mapping[Tuple[str, int], Mapping[str, str]],
    metadata_by_term: Mapping[str, Mapping[str, str]],
) -> Optional[Mapping[str, str]]:
    source_key = metadata_key_from_sample_key(row.get("sample_key", ""))
    if source_key is not None and source_key in metadata_by_source:
        return metadata_by_source[source_key]

    for term_column in ("term", "term_from_line", "metadata_term"):
        term_norm = normalize_term(row.get(term_column, ""))
        if term_norm in metadata_by_term:
            return metadata_by_term[term_norm]
    return None


def enrich_rows_with_ltcdc_metadata(
    rows: Sequence[Dict[str, str]],
    metadata_dir: Path,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    metadata_by_source, metadata_by_term = load_ltcdc_metadata(metadata_dir)
    enriched_rows: List[Dict[str, str]] = []
    counts = {"matched": 0, "unmatched": 0}

    for row in rows:
        enriched_row = dict(row)
        metadata = find_ltcdc_metadata_for_row(
            enriched_row,
            metadata_by_source=metadata_by_source,
            metadata_by_term=metadata_by_term,
        )
        if metadata is not None:
            counts["matched"] += 1
            for column in (
                "source_split",
                "source_index",
                "metadata_term",
                "metadata_term_norm",
                "category",
                "type_norm",
                "frequency_raw",
                "frequency",
            ):
                enriched_row[column] = str(metadata.get(column, ""))
        else:
            counts["unmatched"] += 1
            if "frequency" in enriched_row:
                enriched_row["frequency_raw"] = str(enriched_row.get("frequency", "")).strip()
                enriched_row["frequency"] = normalize_frequency(enriched_row.get("frequency", ""))
            elif "freq" in enriched_row:
                enriched_row["frequency_raw"] = str(enriched_row.get("freq", "")).strip()
                enriched_row["frequency"] = normalize_frequency(enriched_row.get("freq", ""))
            if "type_norm" in enriched_row:
                enriched_row["type_norm"] = normalize_type(enriched_row.get("type_norm", ""))

        enriched_rows.append(enriched_row)

    return enriched_rows, counts


def filter_rows_by_data_type(
    rows: Sequence[Dict[str, str]],
    allowed_data_types: Optional[set[str]],
) -> List[Dict[str, str]]:
    if allowed_data_types is None:
        return list(rows)
    filtered = [
        row
        for row in rows
        if str(row.get("data_type", "")).strip() in allowed_data_types
    ]
    if not filtered:
        raise ValueError(f"No rows found for data types: {sorted(allowed_data_types)}")
    return filtered


def normalize_category_value(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def first_nonempty_value(row: Mapping[str, str], columns: Sequence[str]) -> str:
    for column in columns:
        value = str(row.get(column, "")).strip()
        if value:
            return value
    return ""


def frequency_bucket(row: Mapping[str, str]) -> str:
    raw_value = first_nonempty_value(row, FREQUENCY_COLUMN_CANDIDATES)
    if not raw_value:
        return ""

    normalized_frequency = normalize_frequency(raw_value)
    normalized_category = normalize_category_value(raw_value)
    if normalized_frequency == "low" or normalized_category in LOW_FREQUENCY_VALUES:
        return "frequency_low"
    if normalized_frequency == "high" or normalized_category in HIGH_FREQUENCY_VALUES:
        return "frequency_high"
    return ""


def filter_rows_by_category(
    rows: Sequence[Dict[str, str]],
    category_key: str,
) -> List[Dict[str, str]]:
    if category_key == "all_type":
        return list(rows)
    if category_key == "generic":
        return [
            row
            for row in rows
            if str(row.get("term_type", "")).strip().lower() == "generic"
        ]
    if category_key == "branded":
        return [
            row
            for row in rows
            if str(row.get("term_type", "")).strip().lower() in {"brand", "branded"}
        ]
    if category_key == "low":
        return [row for row in rows if frequency_bucket(row) == "frequency_low"]
    if category_key == "high":
        return [row for row in rows if frequency_bucket(row) == "frequency_high"]
    raise ValueError(f"Unknown category key: {category_key}")


def threshold_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(np.int64)


def build_model_arrays(
    rows: Sequence[Mapping[str, str]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    scores_by_model: Dict[str, np.ndarray] = OrderedDict()
    predictions_by_model: Dict[str, np.ndarray] = OrderedDict()
    brier_scores_by_model: Dict[str, np.ndarray] = OrderedDict()
    thresholds_by_model: Dict[str, float] = OrderedDict()

    for model_name, spec in MODEL_SPECS.items():
        score_column = str(spec["score_column"])
        score_type = str(spec["score_type"])
        scores = np.asarray([float(row[score_column]) for row in rows], dtype=np.float64)
        if score_type == "edit_distance":
            prediction_column = str(spec["prediction_column"])
            if prediction_column not in rows[0]:
                raise ValueError(
                    f"SpellChecker requires `{prediction_column}` for binary predictions. "
                    f"`{score_column}` is used only for score-based metrics."
                )
            threshold = math.nan
            predictions = np.asarray(
                [parse_target(row[prediction_column]) for row in rows],
                dtype=np.int64,
            )
            brier_scores = np.full(scores.shape, np.nan, dtype=np.float64)
        else:
            threshold = float(getattr(args, str(spec["threshold_arg"])))
            predictions = threshold_scores(scores, threshold)
            brier_scores = np.clip(scores, 0.0, 1.0)

        scores_by_model[model_name] = scores
        predictions_by_model[model_name] = predictions
        brier_scores_by_model[model_name] = brier_scores
        thresholds_by_model[model_name] = threshold

    return scores_by_model, predictions_by_model, brier_scores_by_model, thresholds_by_model


def model_display_name(model_name: str) -> str:
    return TABLE_MODEL_ORDER.get(model_name, model_name)


def ordered_table_models(model_names: Sequence[str]) -> List[Tuple[str, str]]:
    remaining = [model_name for model_name in model_names if model_name not in TABLE_MODEL_ORDER]
    ordered = [
        (model_name, display_name)
        for model_name, display_name in TABLE_MODEL_ORDER.items()
        if model_name in model_names
    ]
    ordered.extend((model_name, model_name) for model_name in remaining)
    return ordered


def format_mean_ci_cell(summary_row: Optional[Mapping[str, object]]) -> str:
    if summary_row is None:
        return "N/A"
    mean = float(summary_row["mean"])
    lower = float(summary_row["ci95_lower"])
    upper = float(summary_row["ci95_upper"])
    if any(math.isnan(value) for value in (mean, lower, upper)):
        return "N/A"
    return f"{mean:.3f} [{lower:.3f}, {upper:.3f}]"


def format_mean_cell(summary_row: Optional[Mapping[str, object]]) -> str:
    if summary_row is None:
        return "N/A"
    mean = float(summary_row["mean"])
    if math.isnan(mean):
        return "N/A"
    return f"{mean:.3f}"


def summarize_models(
    *,
    subset_name: str,
    y_true: np.ndarray,
    predictions_by_model: Mapping[str, np.ndarray],
    scores_by_model: Mapping[str, np.ndarray],
    brier_scores_by_model: Mapping[str, np.ndarray],
    thresholds_by_model: Mapping[str, float],
    metric_keys: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    summary_rows: List[Dict[str, object]] = []
    bootstrap_rows: List[Dict[str, object]] = []

    model_items = list(predictions_by_model.items())
    for model_index, (model_name, y_pred) in enumerate(model_items, start=1):
        progress(
            args,
            f"[{subset_name}] model {model_index}/{len(model_items)}: {model_name}",
        )
        model_metric_keys = [
            metric_key
            for metric_key in metric_keys
            if not (model_name == "SpellChecker" and metric_key == "brier_score")
        ]
        summary = stratified_bootstrap_f1_ci(
            y_true=y_true,
            y_pred=y_pred,
            strata=y_true,
            n_bootstrap=args.n_bootstrap,
            confidence_level=args.confidence_level,
            random_state=args.random_state,
            metric_names=model_metric_keys,
            scores=scores_by_model[model_name],
            brier_scores=brier_scores_by_model[model_name],
        )
        spec = MODEL_SPECS[model_name]
        for metric_key in model_metric_keys:
            metric_summary = summary["metrics"][metric_key]
            summary_rows.append(
                {
                    "subset": subset_name,
                    "n_samples": len(y_true),
                    "positive_count": int(y_true.sum()),
                    "negative_count": int(len(y_true) - y_true.sum()),
                    "model": model_name,
                    "model_display": model_display_name(model_name),
                    "metric": metric_key,
                    "metric_display": METRIC_DISPLAY_NAMES[metric_key],
                    "score_column": spec["score_column"],
                    "score_type": spec["score_type"],
                    "threshold": thresholds_by_model[model_name],
                    "mean": metric_summary["value"],
                    "ci95_lower": metric_summary["ci_lower"],
                    "ci95_upper": metric_summary["ci_upper"],
                    "n_valid_bootstrap": metric_summary["n_valid_bootstrap"],
                    "n_bootstrap": args.n_bootstrap,
                    "confidence_level": args.confidence_level,
                    "n_strata": summary["n_strata"],
                }
            )
            for bootstrap_id, value in enumerate(metric_summary["bootstrap_values"], start=1):
                bootstrap_rows.append(
                    {
                        "bootstrap_id": bootstrap_id,
                        "model": model_name,
                        "metric": metric_key,
                        "metric_display": METRIC_DISPLAY_NAMES[metric_key],
                        "value": value,
                    }
                )

    return summary_rows, bootstrap_rows


def build_metric_table_rows(
    *,
    model_summary_rows: Sequence[Mapping[str, object]],
    model_names: Sequence[str],
    metric_keys: Sequence[str],
    include_ci: bool,
) -> Tuple[List[str], List[List[str]]]:
    lookup = {
        (str(row["model"]), str(row["metric"])): row
        for row in model_summary_rows
    }
    ordered_models = ordered_table_models(model_names)
    header = ["Metric", *[display_name for _model_name, display_name in ordered_models]]
    rows: List[List[str]] = []
    for metric_key in metric_keys:
        row = [METRIC_DISPLAY_NAMES[metric_key]]
        for model_name, _display_name in ordered_models:
            summary_row = lookup.get((model_name, metric_key))
            row.append(
                format_mean_ci_cell(summary_row)
                if include_ci
                else format_mean_cell(summary_row)
            )
        rows.append(row)
    return header, rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_table_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_subset_outputs(
    *,
    subset_output_dir: Path,
    model_names: Sequence[str],
    metric_keys: Sequence[str],
    model_summary_rows: Sequence[Mapping[str, object]],
    bootstrap_model_rows: Sequence[Mapping[str, object]],
) -> None:
    write_csv(
        subset_output_dir / "metric_summary.csv",
        model_summary_rows,
        [
            "subset",
            "n_samples",
            "positive_count",
            "negative_count",
            "model",
            "model_display",
            "metric",
            "metric_display",
            "score_column",
            "score_type",
            "threshold",
            "mean",
            "ci95_lower",
            "ci95_upper",
            "n_valid_bootstrap",
            "n_bootstrap",
            "confidence_level",
            "n_strata",
        ],
    )
    write_csv(
        subset_output_dir / "bootstrap_model_metric_values.csv",
        bootstrap_model_rows,
        ["bootstrap_id", "model", "metric", "metric_display", "value"],
    )

    table_header, table_rows = build_metric_table_rows(
        model_summary_rows=model_summary_rows,
        model_names=model_names,
        metric_keys=metric_keys,
        include_ci=True,
    )
    write_table_csv(subset_output_dir / "bootstrap_table_mean_ci.csv", table_header, table_rows)
    mean_header, mean_rows = build_metric_table_rows(
        model_summary_rows=model_summary_rows,
        model_names=model_names,
        metric_keys=metric_keys,
        include_ci=False,
    )
    write_table_csv(subset_output_dir / "bootstrap_table_mean.csv", mean_header, mean_rows)

    write_f1_outputs(
        subset_output_dir=subset_output_dir,
        model_summary_rows=model_summary_rows,
        bootstrap_model_rows=bootstrap_model_rows,
    )


def write_f1_outputs(
    *,
    subset_output_dir: Path,
    model_summary_rows: Sequence[Mapping[str, object]],
    bootstrap_model_rows: Sequence[Mapping[str, object]],
) -> None:
    f1_model_rows = [
        {
            "subset": row["subset"],
            "n_samples": row["n_samples"],
            "positive_count": row["positive_count"],
            "negative_count": row["negative_count"],
            "model": row["model"],
            "score_column": row["score_column"],
            "score_type": row["score_type"],
            "threshold": row["threshold"],
            "f1": row["mean"],
            "ci_lower": row["ci95_lower"],
            "ci_upper": row["ci95_upper"],
            "n_valid_bootstrap": row["n_valid_bootstrap"],
            "n_bootstrap": row["n_bootstrap"],
            "confidence_level": row["confidence_level"],
        }
        for row in model_summary_rows
        if row["metric"] == "f1"
    ]
    if f1_model_rows:
        write_csv(
            subset_output_dir / "model_f1_ci.csv",
            f1_model_rows,
            [
                "subset",
                "n_samples",
                "positive_count",
                "negative_count",
                "model",
                "score_column",
                "score_type",
                "threshold",
                "f1",
                "ci_lower",
                "ci_upper",
                "n_valid_bootstrap",
                "n_bootstrap",
                "confidence_level",
            ],
        )

    f1_bootstrap_model_rows = [
        {"bootstrap_id": row["bootstrap_id"], "model": row["model"], "f1": row["value"]}
        for row in bootstrap_model_rows
        if row["metric"] == "f1"
    ]
    if f1_bootstrap_model_rows:
        write_csv(
            subset_output_dir / "bootstrap_model_f1_values.csv",
            f1_bootstrap_model_rows,
            ["bootstrap_id", "model", "f1"],
        )


def write_category_mean_ci_outputs(
    *,
    category_output_dir: Path,
    category_key: str,
    model_names: Sequence[str],
    metric_keys: Sequence[str],
    model_summary_rows: Sequence[Mapping[str, object]],
) -> None:
    category_rows = [
        {
            "category": category_key,
            "category_display": CATEGORY_DISPLAY_NAMES[category_key],
            **row,
        }
        for row in model_summary_rows
    ]
    write_csv(
        category_output_dir / f"{category_key}_metric_summary.csv",
        category_rows,
        [
            "category",
            "category_display",
            "subset",
            "n_samples",
            "positive_count",
            "negative_count",
            "model",
            "model_display",
            "metric",
            "metric_display",
            "score_column",
            "score_type",
            "threshold",
            "mean",
            "ci95_lower",
            "ci95_upper",
            "n_valid_bootstrap",
            "n_bootstrap",
            "confidence_level",
            "n_strata",
        ],
    )

    table_header, table_rows = build_metric_table_rows(
        model_summary_rows=model_summary_rows,
        model_names=model_names,
        metric_keys=metric_keys,
        include_ci=True,
    )
    write_table_csv(
        category_output_dir / f"{category_key}_bootstrap_table_mean_ci.csv",
        table_header,
        table_rows,
    )


def run_category_breakdown_analysis(
    *,
    subset_name: str,
    rows: Sequence[Dict[str, str]],
    subset_output_dir: Path,
    metric_keys: Sequence[str],
    args: argparse.Namespace,
) -> None:
    category_output_dir = subset_output_dir / "category_breakdowns"
    skipped_categories: List[str] = []

    for category_key in CATEGORY_DISPLAY_NAMES:
        category_rows = filter_rows_by_category(rows, category_key)
        if not category_rows:
            skipped_categories.append(category_key)
            continue

        y_true = np.asarray([parse_target(row["target"]) for row in category_rows], dtype=np.int64)
        progress(
            args,
            f"[{subset_name}] category {CATEGORY_DISPLAY_NAMES[category_key]}: "
            f"n={len(y_true)}, positive={int(y_true.sum())}, "
            f"negative={int(len(y_true) - y_true.sum())}",
        )
        (
            scores_by_model,
            predictions_by_model,
            brier_scores_by_model,
            thresholds_by_model,
        ) = build_model_arrays(category_rows, args)

        category_subset_name = f"{subset_name}_{category_key}"
        model_summary_rows, bootstrap_model_rows = summarize_models(
            subset_name=category_subset_name,
            y_true=y_true,
            predictions_by_model=predictions_by_model,
            scores_by_model=scores_by_model,
            brier_scores_by_model=brier_scores_by_model,
            thresholds_by_model=thresholds_by_model,
            metric_keys=metric_keys,
            args=args,
        )
        write_subset_outputs(
            subset_output_dir=category_output_dir / category_key,
            model_names=list(predictions_by_model.keys()),
            metric_keys=metric_keys,
            model_summary_rows=model_summary_rows,
            bootstrap_model_rows=bootstrap_model_rows,
        )
        write_category_mean_ci_outputs(
            category_output_dir=category_output_dir,
            category_key=category_key,
            model_names=list(predictions_by_model.keys()),
            metric_keys=metric_keys,
            model_summary_rows=model_summary_rows,
        )

    if skipped_categories:
        skipped = ", ".join(CATEGORY_DISPLAY_NAMES[key] for key in skipped_categories)
        progress(args, f"Skipped {subset_name} category breakdowns with no usable rows: {skipped}")


def run_subset_analysis(
    *,
    subset_name: str,
    rows: Sequence[Dict[str, str]],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    y_true = np.asarray([parse_target(row["target"]) for row in rows], dtype=np.int64)
    metric_keys = metric_keys_for_args(args)
    progress(
        args,
        f"Starting {subset_name}: n={len(y_true)}, positive={int(y_true.sum())}, "
        f"negative={int(len(y_true) - y_true.sum())}, "
        f"metrics={','.join(metric_keys)}, n_bootstrap={args.n_bootstrap}",
    )
    (
        scores_by_model,
        predictions_by_model,
        brier_scores_by_model,
        thresholds_by_model,
    ) = build_model_arrays(rows, args)

    model_summary_rows, bootstrap_model_rows = summarize_models(
        subset_name=subset_name,
        y_true=y_true,
        predictions_by_model=predictions_by_model,
        scores_by_model=scores_by_model,
        brier_scores_by_model=brier_scores_by_model,
        thresholds_by_model=thresholds_by_model,
        metric_keys=metric_keys,
        args=args,
    )
    subset_output_dir = output_dir / subset_name
    write_subset_outputs(
        subset_output_dir=subset_output_dir,
        model_names=list(predictions_by_model.keys()),
        metric_keys=metric_keys,
        model_summary_rows=model_summary_rows,
        bootstrap_model_rows=bootstrap_model_rows,
    )
    if args.skip_category_breakdowns:
        progress(args, f"Skipped {subset_name} category breakdowns (--skip-category-breakdowns).")
    else:
        run_category_breakdown_analysis(
            subset_name=subset_name,
            rows=rows,
            subset_output_dir=subset_output_dir,
            metric_keys=metric_keys,
            args=args,
        )
    progress(
        args,
        f"Wrote {subset_name} results to {subset_output_dir} "
        f"(n={len(y_true)}, positive={int(y_true.sum())}, negative={int(len(y_true) - y_true.sum())})",
    )


def main() -> int:
    args = parse_args()
    project_root = PROJECT_ROOT.resolve()
    input_path = resolve_input_path(project_root, args.input)
    output_dir = resolve_path(project_root, args.output_dir)
    metadata_dir = resolve_path(project_root, args.ltcdc_metadata_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_ltcdc_samples(input_path)
    rows, metadata_counts = enrich_rows_with_ltcdc_metadata(rows, metadata_dir)
    progress(
        args,
        "LTCDC metadata enrichment: "
        f"matched={metadata_counts['matched']}, unmatched={metadata_counts['unmatched']} "
        f"from {metadata_dir}",
    )
    for subset_name, allowed_data_types in SUBSET_SPECS.items():
        subset_rows = filter_rows_by_data_type(rows, allowed_data_types)
        run_subset_analysis(
            subset_name=subset_name,
            rows=subset_rows,
            output_dir=output_dir,
            args=args,
        )

    progress(args, f"Wrote LTCDC bootstrap results under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
