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

"""Manuscript-specified settings shared by the two XGBoost baselines."""

from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Mapping


CLASSIFICATION_THRESHOLD = 0.5
XGB_LEARNING_RATES = (0.03, 0.05, 0.1)
XGB_N_ESTIMATORS = (400, 600)
XGB_MAX_DEPTHS = (4, 6, 8, 10)
XGB_SUBSAMPLES = (0.8, 1.0)


def manuscript_xgb_grid() -> List[Dict[str, float | int]]:
    """Return the 48 XGBoost configurations described in the manuscript."""

    return [
        {
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "subsample": subsample,
        }
        for learning_rate, n_estimators, max_depth, subsample in product(
            XGB_LEARNING_RATES,
            XGB_N_ESTIMATORS,
            XGB_MAX_DEPTHS,
            XGB_SUBSAMPLES,
        )
    ]


def select_best_f1_trial(
    trials: Iterable[Mapping[str, object]],
) -> Mapping[str, object]:
    """Select the first configuration attaining the highest development F1."""

    trial_list = list(trials)
    if not trial_list:
        raise ValueError("At least one grid-search trial is required.")
    return max(trial_list, key=lambda trial: float(trial["dev_f1"]))


def validate_manuscript_xgb_config(
    config: Mapping[str, object],
) -> Dict[str, float | int]:
    """Normalize one configuration and require membership in the manuscript grid."""

    required = {"learning_rate", "n_estimators", "max_depth", "subsample"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"XGBoost configuration is missing: {sorted(missing)}")
    normalized: Dict[str, float | int] = {
        "learning_rate": float(config["learning_rate"]),
        "n_estimators": int(config["n_estimators"]),
        "max_depth": int(config["max_depth"]),
        "subsample": float(config["subsample"]),
    }
    if normalized not in manuscript_xgb_grid():
        raise ValueError(
            "XGBoost configuration is not one of the 48 manuscript combinations: "
            f"{normalized}"
        )
    return normalized
