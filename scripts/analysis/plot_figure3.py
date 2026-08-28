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
"""Plot Figure 3 ROC curves for LTCDC OOV samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sklearn_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drug_spelling.resampling import stratified_resample_indices


DEFAULT_INPUT = Path("artifacts/predictions/ltcdc_predictions.csv")
DEFAULT_OUTPUT = Path("artifacts/figures/Figure_3.png")
MEAN_FPR = np.linspace(0.0, 1.0, 101)

MODEL_SPECS = [
    {
        "name": "CharBERTDrug",
        "display": "CharBERTDrug",
        "score_col": "CharBERTDrug_probability",
        "color": "#4C78A8",
    },
    {
        "name": "BERTDrug",
        "display": "BERTDrug",
        "score_col": "BERTDrug_probability",
        "color": "#F58518",
    },
    {
        "name": "SpellChecker",
        "display": "SpellChecker",
        "score_col": "SpellChecker_edit_distance",
        "color": "#54A24B",
    },
    {
        "name": "fasttext+xgboost",
        "display": r"$\mathregular{fastText_{ML}}$",
        "score_col": "fasttext+xgboost_probability",
        "color": "#E45756",
    },
    {
        "name": "BioWordVec",
        "display": r"$\mathregular{BioWordVec_{ML}}$",
        "score_col": "BioWordVec_probability",
        "color": "#B279A2",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Figure 3 ROC curves from LTCDC sample-level predictions."
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
        "--title",
        default="",
        help="Figure title. Use an empty string to omit the title.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap samples used for the ROC 95%% CI shaded area.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for the ROC shaded area.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=123,
        help="Random seed used for bootstrap sampling.",
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
    required_columns = {"target", "data_type", *(spec["score_col"] for spec in MODEL_SPECS)}
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


def interpolate_roc(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    fpr, tpr, _ = sklearn_metrics.roc_curve(y_true, scores)
    interp_tpr = np.interp(MEAN_FPR, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tpr[-1] = 1.0
    return interp_tpr


def collect_roc_curves(
    df: pd.DataFrame,
    n_bootstrap: int,
    confidence_level: float,
    random_state: int,
) -> tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, float]]:
    roc_summary: Dict[str, Dict[str, np.ndarray]] = {}
    auc_lookup: Dict[str, float] = {}
    rng = np.random.default_rng(random_state)
    lower_percentile = (1.0 - confidence_level) / 2.0 * 100.0
    upper_percentile = (1.0 + confidence_level) / 2.0 * 100.0

    for spec in MODEL_SPECS:
        model_name = spec["name"]
        display_name = spec["display"]
        score_col = spec["score_col"]

        model_df = df[["target_binary", score_col]].dropna().copy()
        y_true = model_df["target_binary"].astype(np.int64).to_numpy()
        scores = model_df[score_col].astype(float).to_numpy()
        if np.unique(y_true).size < 2:
            raise ValueError(f"Model `{model_name}` has fewer than two classes after dropping NaN scores.")

        bootstrap_tprs = []
        for _ in range(n_bootstrap):
            indices = stratified_resample_indices(y_true, rng)
            sample_y_true = y_true[indices]
            bootstrap_tprs.append(interpolate_roc(sample_y_true, scores[indices]))
        if not bootstrap_tprs:
            raise ValueError(f"No valid bootstrap ROC curves collected for model `{model_name}`.")

        bootstrap_tpr_matrix = np.vstack(bootstrap_tprs)

        roc_summary[display_name] = {
            "mean": np.mean(bootstrap_tpr_matrix, axis=0),
            "ci95_lower": np.percentile(bootstrap_tpr_matrix, lower_percentile, axis=0),
            "ci95_upper": np.percentile(bootstrap_tpr_matrix, upper_percentile, axis=0),
        }
        auc_lookup[model_name] = float(sklearn_metrics.roc_auc_score(y_true, scores))

    return roc_summary, auc_lookup


def plot_mean_roc_curves(
    roc_summary: Dict[str, Dict[str, np.ndarray]],
    mean_auc_lookup: Dict[str, float],
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(12, 8))
    for spec in MODEL_SPECS:
        display_name = spec["display"]
        model_name = spec["name"]
        mean_tpr = roc_summary[display_name]["mean"]
        lower_tpr = roc_summary[display_name]["ci95_lower"]
        upper_tpr = roc_summary[display_name]["ci95_upper"]
        label = display_name
        if model_name in mean_auc_lookup:
            label = f"{display_name} (AUC={mean_auc_lookup[model_name]:.3f})"
        plt.plot(
            MEAN_FPR,
            mean_tpr,
            color=spec["color"],
            label=label,
            lw=2.5,
        )
        plt.fill_between(
            MEAN_FPR,
            lower_tpr,
            upper_tpr,
            color="gray",
            alpha=0.3,
            linewidth=0,
        )

    plt.plot([0, 1], [0, 1], linestyle="--", lw=1.5, color="#777777", alpha=0.8)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    if title:
        plt.title(title, fontsize=18)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    subset = load_subset(input_path, args.data_type)
    roc_summary, auc_lookup = collect_roc_curves(
        subset,
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence_level,
        random_state=args.random_state,
    )
    plot_mean_roc_curves(roc_summary, auc_lookup, output_path, args.title)

    positive_count = int(subset["target_binary"].sum())
    negative_count = int(len(subset) - positive_count)
    print(
        f"Saved Figure 3 ROC curve to: {output_path} "
        f"(data_type={args.data_type}, n={len(subset)}, "
        f"positive={positive_count}, negative={negative_count}, "
        f"n_bootstrap={args.n_bootstrap}, confidence_level={args.confidence_level})"
    )
    for spec in MODEL_SPECS:
        print(f"{spec['display']}: AUC={auc_lookup[spec['name']]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
