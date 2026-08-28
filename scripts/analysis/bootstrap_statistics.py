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

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sklearn.metrics as sklearn_metrics
from sklearn.metrics import f1_score


RELEASE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = RELEASE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drug_spelling.resampling import stratified_resample_indices


# used for LTCDC test set
def bootstrap_f1_ci(
    y_true,
    y_pred,
    n_bootstrap=2000,
    confidence_level=0.95,
    random_state=123
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        # Sample participants with replacement
        indices = rng.integers(0, n, size=n)

        y_true_b = y_true[indices]
        y_pred_b = y_pred[indices]

        # Skip samples in which F1 is undefined because a class is absent
        if len(np.unique(y_true_b)) < 2:
            continue

        bootstrap_scores.append(
            f1_score(y_true_b, y_pred_b, zero_division=0)
        )

    bootstrap_scores = np.asarray(bootstrap_scores)

    alpha = 1 - confidence_level
    lower = np.quantile(bootstrap_scores, alpha / 2)
    upper = np.quantile(bootstrap_scores, 1 - alpha / 2)

    full_sample_f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "f1": full_sample_f1,
        "ci_lower": lower,
        "ci_upper": upper,
        "n_valid_bootstrap": len(bootstrap_scores),
    }


def stratified_bootstrap_f1_ci(
    y_true,
    y_pred,
    strata=None,
    n_bootstrap=2000,
    confidence_level=0.95,
    average="binary",
    random_state=123,
    metric_names=None,
    scores=None,
    brier_scores=None,
):
    """
    Estimate metrics and confidence intervals using stratified test-set bootstrap.

    Unlike bootstrap_f1_ci(), this function samples with replacement within
    each stratum and keeps the original stratum sizes in every bootstrap
    replicate. If strata is None, y_true is used as the stratum variable, which
    preserves the original class balance in each bootstrap sample.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metric_names = _as_metric_list(metric_names)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if scores is not None:
        scores = np.asarray(scores)
        if len(scores) != len(y_true):
            raise ValueError("scores must have the same length as y_true.")
    if brier_scores is not None:
        brier_scores = np.asarray(brier_scores)
        if len(brier_scores) != len(y_true):
            raise ValueError("brier_scores must have the same length as y_true.")

    if strata is None:
        strata = y_true
    strata = np.asarray(strata)

    if len(strata) != len(y_true):
        raise ValueError("strata must have the same length as y_true and y_pred.")

    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    stratum_values = np.unique(strata)
    stratum_indices = {
        stratum_value: np.flatnonzero(strata == stratum_value)
        for stratum_value in stratum_values
    }

    rng = np.random.default_rng(random_state)
    bootstrap_scores = {metric_name: [] for metric_name in metric_names}

    for _ in range(n_bootstrap):
        sampled_indices = stratified_resample_indices(strata, rng)

        y_true_b = y_true[sampled_indices]
        y_pred_b = y_pred[sampled_indices]
        scores_b = scores[sampled_indices] if scores is not None else None
        brier_scores_b = (
            brier_scores[sampled_indices]
            if brier_scores is not None
            else (scores_b if scores_b is not None else None)
        )
        metric_values = _two_way_run_metric_scores(
            y_true=y_true_b,
            predictions=y_pred_b.reshape(1, -1),
            weights=np.ones(len(y_true_b), dtype=np.int64),
            metric_names=metric_names,
            scores=scores_b.reshape(1, -1) if scores_b is not None else None,
            brier_scores=(
                brier_scores_b.reshape(1, -1)
                if brier_scores_b is not None
                else None
            ),
        )
        for metric_name in metric_names:
            bootstrap_scores[metric_name].append(metric_values[metric_name][0])

    alpha = 1 - confidence_level
    full_metric_values = _two_way_run_metric_scores(
        y_true=y_true,
        predictions=y_pred.reshape(1, -1),
        weights=np.ones(len(y_true), dtype=np.int64),
        metric_names=metric_names,
        scores=scores.reshape(1, -1) if scores is not None else None,
        brier_scores=(
            brier_scores.reshape(1, -1)
            if brier_scores is not None
            else (scores.reshape(1, -1) if scores is not None else None)
        ),
    )

    metric_results = {}
    for metric_name in metric_names:
        metric_bootstrap_scores = np.asarray(bootstrap_scores[metric_name], dtype=float)
        finite_bootstrap_scores = metric_bootstrap_scores[~np.isnan(metric_bootstrap_scores)]
        if finite_bootstrap_scores.size == 0:
            lower = np.nan
            upper = np.nan
        else:
            lower = np.quantile(finite_bootstrap_scores, alpha / 2)
            upper = np.quantile(finite_bootstrap_scores, 1 - alpha / 2)
        metric_results[metric_name] = {
            "value": float(full_metric_values[metric_name][0]),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "n_valid_bootstrap": int(finite_bootstrap_scores.size),
            "bootstrap_values": metric_bootstrap_scores,
        }

    result = {
        "metrics": metric_results,
        "n_strata": len(stratum_values),
        "stratum_counts": {
            stratum_value: len(indices)
            for stratum_value, indices in stratum_indices.items()
        },
    }

    if "f1" in metric_results:
        result.update(
            {
                "f1": metric_results["f1"]["value"],
                "ci_lower": metric_results["f1"]["ci_lower"],
                "ci_upper": metric_results["f1"]["ci_upper"],
                "n_valid_bootstrap": metric_results["f1"]["n_valid_bootstrap"],
            }
        )

    return result


###############################

# used for RxNorm test set
def _as_metric_list(metric_names):
    if metric_names is None:
        return ["f1"]
    if isinstance(metric_names, str):
        return [metric_names]
    return list(metric_names)


def _require_metric_matrix(name, matrix, expected_shape):
    if matrix is None:
        return None
    matrix = np.asarray(matrix)
    if matrix.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {matrix.shape}.")
    return matrix


def _safe_divide_array(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator != 0,
    )


def _weighted_roc_auc(y_true, y_score, weights):
    positive_weight = float(weights[y_true == 1].sum())
    negative_weight = float(weights[y_true == 0].sum())
    if positive_weight <= 0.0 or negative_weight <= 0.0:
        return np.nan
    mask = weights > 0
    try:
        return float(
            sklearn_metrics.roc_auc_score(
                y_true[mask],
                y_score[mask],
                sample_weight=weights[mask],
            )
        )
    except ValueError:
        return np.nan


def _weighted_pr_auc(y_true, y_score, weights):
    if float(weights[y_true == 1].sum()) <= 0.0:
        return np.nan
    mask = weights > 0
    try:
        return float(
            sklearn_metrics.average_precision_score(
                y_true[mask],
                y_score[mask],
                sample_weight=weights[mask],
            )
        )
    except ValueError:
        return np.nan


def _two_way_run_metric_scores(
    y_true,
    predictions,
    weights,
    metric_names,
    scores=None,
    brier_scores=None,
):
    y_true = np.asarray(y_true, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.int64)

    positive = y_true
    negative = 1 - y_true
    predicted_positive = predictions
    predicted_negative = 1 - predictions

    tp = (predicted_positive * positive) @ weights
    fp = (predicted_positive * negative) @ weights
    tn = (predicted_negative * negative) @ weights
    fn = (predicted_negative * positive) @ weights
    total = float(weights.sum())

    metric_scores = {}
    if "precision" in metric_names:
        metric_scores["precision"] = _safe_divide_array(tp, tp + fp)
    if "recall" in metric_names:
        metric_scores["recall"] = _safe_divide_array(tp, tp + fn)
    if "f1" in metric_names:
        metric_scores["f1"] = _safe_divide_array(2 * tp, (2 * tp) + fp + fn)
    if "accuracy" in metric_names:
        metric_scores["accuracy"] = ((tp + tn) / total).astype(np.float64)
    if "specificity" in metric_names:
        metric_scores["specificity"] = _safe_divide_array(tn, tn + fp)

    if "brier_score" in metric_names:
        if brier_scores is None:
            if scores is None:
                raise ValueError("brier_scores or scores are required for brier_score.")
            brier_scores = scores
        squared_errors = (np.asarray(brier_scores, dtype=np.float64) - y_true) ** 2
        metric_scores["brier_score"] = (squared_errors @ weights) / total

    if "roc_auc" in metric_names or "pr_auc" in metric_names:
        if scores is None:
            raise ValueError("scores are required for roc_auc and pr_auc.")
        scores = np.asarray(scores, dtype=np.float64)
        roc_values = []
        pr_values = []
        for run_index in range(scores.shape[0]):
            if "roc_auc" in metric_names:
                roc_values.append(_weighted_roc_auc(y_true, scores[run_index], weights))
            if "pr_auc" in metric_names:
                pr_values.append(_weighted_pr_auc(y_true, scores[run_index], weights))
        if "roc_auc" in metric_names:
            metric_scores["roc_auc"] = np.asarray(roc_values, dtype=np.float64)
        if "pr_auc" in metric_names:
            metric_scores["pr_auc"] = np.asarray(pr_values, dtype=np.float64)

    return metric_scores


def _finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def _finite_std(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size < 2:
        return np.nan
    return float(np.std(values, ddof=1))


def _finite_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def two_way_bootstrap_f1(
    y_true,
    predictions,
    n_bootstrap=2000,
    random_state=123,
    metric_names=None,
    scores=None,
    brier_scores=None,
    bootstrap_seeds=None,
):
    """
    y_true:
        Shape (n_test,)

    predictions:
        Binary predictions with shape (n_models, n_test)
    """
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    metric_names = _as_metric_list(metric_names)

    n_models, n_test = predictions.shape

    if len(y_true) != n_test:
        raise ValueError(
            "The number of test labels must match the prediction columns."
        )
    scores = _require_metric_matrix("scores", scores, predictions.shape)
    brier_scores = _require_metric_matrix("brier_scores", brier_scores, predictions.shape)

    full_run_scores = _two_way_run_metric_scores(
        y_true=y_true,
        predictions=predictions,
        weights=np.ones(n_test, dtype=np.int64),
        metric_names=metric_names,
        scores=scores,
        brier_scores=brier_scores,
    )

    if bootstrap_seeds is not None:
        bootstrap_seeds = [int(seed) for seed in bootstrap_seeds]
        n_bootstrap = len(bootstrap_seeds)
    rng = np.random.default_rng(random_state)
    bootstrap_estimates = {
        metric_name: np.empty(n_bootstrap, dtype=float)
        for metric_name in metric_names
    }

    for b in range(n_bootstrap):
        if bootstrap_seeds is None:
            bootstrap_rng = rng
        else:
            bootstrap_rng = np.random.default_rng(bootstrap_seeds[b])

        # Resample fitted models
        model_indices = bootstrap_rng.integers(
            low=0,
            high=n_models,
            size=n_models,
        )

        # Resample test participants
        test_indices = bootstrap_rng.integers(
            low=0,
            high=n_test,
            size=n_test,
        )

        weights = np.bincount(test_indices, minlength=n_test).astype(np.int64)
        bootstrap_run_scores = _two_way_run_metric_scores(
            y_true=y_true,
            predictions=predictions,
            weights=weights,
            metric_names=metric_names,
            scores=scores,
            brier_scores=brier_scores,
        )
        for metric_name in metric_names:
            bootstrap_estimates[metric_name][b] = _finite_mean(
                bootstrap_run_scores[metric_name][model_indices]
            )

    metric_results = {}
    for metric_name in metric_names:
        ci_lower, ci_upper = _finite_ci(bootstrap_estimates[metric_name])
        full_scores = full_run_scores[metric_name]
        metric_results[metric_name] = {
            "mean": _finite_mean(full_scores),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "training_sd": _finite_std(full_scores),
            "model_scores": full_scores,
            "bootstrap_values": bootstrap_estimates[metric_name],
            "n_valid_bootstrap": int(np.sum(~np.isnan(bootstrap_estimates[metric_name]))),
        }

    result = {"metrics": metric_results}
    if "f1" in metric_results:
        f1_result = metric_results["f1"]
        result.update(
            {
                "mean_f1": f1_result["mean"],
                "ci_lower": f1_result["ci_lower"],
                "ci_upper": f1_result["ci_upper"],
                "training_sd": f1_result["training_sd"],
                "model_f1_scores": f1_result["model_scores"],
                "bootstrap_estimates": f1_result["bootstrap_values"],
            }
        )
    return result


