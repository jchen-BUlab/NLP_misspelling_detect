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

"""Metrics shared by evaluation and secondary-analysis scripts."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import numpy as np
import sklearn.metrics as sk_metrics


MODEL_SPECS = OrderedDict(
    [
        ("CharBERTDrug", {"score": "CharBERTDrug_probability"}),
        ("BERTDrug", {"score": "BERTDrug_probability"}),
        (
            "SpellChecker",
            {
                "score": "SpellChecker_edit_distance",
                "prediction": "SpellChecker_prediction",
            },
        ),
        ("fastTextML", {"score": "fasttext+xgboost_probability"}),
        ("BioWordVecML", {"score": "BioWordVec_probability"}),
    ]
)


def safe_auc(function, targets: np.ndarray, scores: np.ndarray) -> float:
    """Return NaN when an AUC metric is undefined for a one-class subset."""

    if np.unique(targets).size < 2:
        return float("nan")
    return float(function(targets, scores))


def binary_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    *,
    include_brier: bool = True,
) -> Dict[str, float]:
    """Compute the metrics reported in the manuscript and appendices."""

    y_true = np.asarray(targets, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(predictions, dtype=np.int64).reshape(-1)
    y_score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not (len(y_true) == len(y_pred) == len(y_score)):
        raise ValueError("targets, predictions, and scores must have equal lengths")
    result = {
        "n": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "negative_count": int((1 - y_true).sum()),
        "precision": float(sk_metrics.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(sk_metrics.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(sk_metrics.f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(sk_metrics.accuracy_score(y_true, y_pred)),
        "specificity": float(sk_metrics.recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "roc_auc": safe_auc(sk_metrics.roc_auc_score, y_true, y_score),
        "pr_auc": safe_auc(sk_metrics.average_precision_score, y_true, y_score),
    }
    if include_brier:
        result["brier_score"] = float(sk_metrics.brier_score_loss(y_true, y_score))
    return result
