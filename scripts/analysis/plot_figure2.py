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
"""Plot Figure 2 box plot for LTCDC OOV bootstrap metrics."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as sklearn_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drug_spelling.resampling import stratified_resample_indices


DEFAULT_INPUT = Path("artifacts/predictions/ltcdc_predictions.csv")
DEFAULT_OUTPUT = Path("artifacts/figures/Figure_2.png")

MODEL_SPECS = [
    {
        "name": "CharBERTDrug",
        "display": "CharBERTDrug",
        "score_col": "CharBERTDrug_probability",
        "score_type": "probability",
        "color": "#4C78A8",
    },
    {
        "name": "BERTDrug",
        "display": "BERTDrug",
        "score_col": "BERTDrug_probability",
        "score_type": "probability",
        "color": "#F58518",
    },
    {
        "name": "SpellChecker",
        "display": "SpellChecker",
        "score_col": "SpellChecker_edit_distance",
        "prediction_col": "SpellChecker_prediction",
        "score_type": "edit_distance",
        "color": "#54A24B",
    },
    {
        "name": "fasttext+xgboost",
        "display": r"$\mathregular{fastText_{ML}}$",
        "score_col": "fasttext+xgboost_probability",
        "score_type": "probability",
        "color": "#E45756",
    },
    {
        "name": "BioWordVec",
        "display": r"$\mathregular{BioWordVec_{ML}}$",
        "score_col": "BioWordVec_probability",
        "score_type": "probability",
        "color": "#B279A2",
    },
]

PLOT_METRICS = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
]

METRIC_ORDER = [display for _, display in PLOT_METRICS]
MODEL_ORDER = [spec["display"] for spec in MODEL_SPECS]
MODEL_PALETTE = {spec["display"]: spec["color"] for spec in MODEL_SPECS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the manuscript Figure 2 bootstrap metric distributions."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV with target, data_type, and model score columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path.",
    )
    parser.add_argument(
        "--data-type",
        default="oov",
        help="Value in the data_type column to plot. Defaults to oov.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap samples used for metric distributions.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=123,
        help="Random seed used for bootstrap sampling.",
    )
    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=0.5,
        help="Positive prediction threshold for probability score columns.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional figure title; the default omits it.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_target(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "positive", "__label__positive", "true"}:
        return 1
    if text in {"0", "0.0", "negative", "__label__negative", "false"}:
        return 0
    raise ValueError(f"Could not parse binary target value: {value}")


def load_subset(input_path: Path, data_type: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    required_columns = {"target", "data_type"}
    for spec in MODEL_SPECS:
        required_columns.add(str(spec["score_col"]))
        if "prediction_col" in spec:
            required_columns.add(str(spec["prediction_col"]))
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"{input_path} is missing required columns: {missing_columns}")

    data_type_normalized = data_type.strip().lower()
    subset = df[df["data_type"].astype(str).str.strip().str.lower() == data_type_normalized].copy()
    if subset.empty:
        raise ValueError(f"No rows found with data_type == {data_type!r} in {input_path}")

    subset["target_binary"] = subset["target"].map(parse_target).astype(np.int64)
    if subset["target_binary"].nunique() < 2:
        raise ValueError(f"Subset data_type == {data_type!r} does not contain both classes.")
    return subset


def threshold_for_model(spec: Dict[str, object], args: argparse.Namespace) -> float:
    return float(args.probability_threshold)


def safe_metric(metric_fn) -> float:
    try:
        value = float(metric_fn())
    except ValueError:
        return math.nan
    return value


def compute_model_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "precision": float(sklearn_metrics.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(sklearn_metrics.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(sklearn_metrics.f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_metric(lambda: sklearn_metrics.roc_auc_score(y_true, scores)),
        "pr_auc": safe_metric(lambda: sklearn_metrics.average_precision_score(y_true, scores)),
    }


def build_bootstrap_metrics_dataframe(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be positive.")

    y_true_all = df["target_binary"].astype(np.int64).to_numpy()
    model_scores = {
        spec["name"]: df[spec["score_col"]].astype(float).to_numpy()
        for spec in MODEL_SPECS
    }
    model_predictions = {}
    for spec in MODEL_SPECS:
        model_name = spec["name"]
        if "prediction_col" in spec:
            model_predictions[model_name] = df[str(spec["prediction_col"])].map(parse_target).astype(np.int64).to_numpy()
        else:
            threshold = threshold_for_model(spec, args)
            model_predictions[model_name] = (model_scores[model_name] >= threshold).astype(np.int64)
    rng = np.random.default_rng(args.random_state)
    rows: List[Dict[str, object]] = []

    for bootstrap_id in range(1, args.n_bootstrap + 1):
        indices = stratified_resample_indices(y_true_all, rng)
        sample_y_true = y_true_all[indices]

        for spec in MODEL_SPECS:
            model_name = spec["name"]
            display_name = spec["display"]
            scores = model_scores[model_name][indices]
            y_pred = model_predictions[model_name][indices]
            metric_values = compute_model_metrics(sample_y_true, scores, y_pred)

            for metric_key, metric_display in PLOT_METRICS:
                value = metric_values[metric_key]
                if math.isnan(value):
                    continue
                rows.append(
                    {
                        "bootstrap_id": bootstrap_id,
                        "metric": metric_key,
                        "Metric": metric_display,
                        "model": model_name,
                        "Model": display_name,
                        "Value": value,
                    }
                )

    metrics_df = pd.DataFrame(rows)
    metrics_df["Metric"] = pd.Categorical(
        metrics_df["Metric"],
        categories=METRIC_ORDER,
        ordered=True,
    )
    metrics_df["Model"] = pd.Categorical(
        metrics_df["Model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    return metrics_df[["bootstrap_id", "metric", "model", "Metric", "Value", "Model"]]


def plot_box(metrics_df: pd.DataFrame, output_path: Path, title: str = "") -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(
        data=metrics_df,
        x="Metric",
        y="Value",
        hue="Model",
        order=METRIC_ORDER,
        hue_order=MODEL_ORDER,
        palette=MODEL_PALETTE,
        width=0.6,
        fliersize=2.5,
        linewidth=1.0,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=13)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(title="", fontsize=11, title_fontsize=11, loc="upper right", frameon=True)
    if title:
        ax.set_title(title, fontsize=18)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    subset = load_subset(input_path, args.data_type)
    metrics_df = build_bootstrap_metrics_dataframe(subset, args)
    plot_box(metrics_df, output_path, title=args.title)

    positive_count = int(subset["target_binary"].sum())
    negative_count = int(len(subset) - positive_count)
    print(
        f"Saved Figure 2 box plot to: {output_path} "
        f"(data_type={args.data_type}, n={len(subset)}, "
        f"positive={positive_count}, negative={negative_count}, "
        f"n_bootstrap={args.n_bootstrap})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
