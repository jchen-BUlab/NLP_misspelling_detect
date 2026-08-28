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

"""Reusable resampling helpers for manuscript analyses."""

from __future__ import annotations

from typing import Any

import numpy as np


def stratified_resample_indices(
    strata: np.ndarray,
    rng: Any,
) -> np.ndarray:
    """Sample rows with replacement while preserving every stratum count."""

    stratum_array = np.asarray(strata)
    if stratum_array.ndim != 1:
        raise ValueError("strata must be a one-dimensional array")
    if stratum_array.size == 0:
        raise ValueError("strata must not be empty")

    sampled = []
    for stratum in np.unique(stratum_array):
        indices = np.flatnonzero(stratum_array == stratum)
        sampled.append(rng.choice(indices, size=len(indices), replace=True))
    return rng.permutation(np.concatenate(sampled))
