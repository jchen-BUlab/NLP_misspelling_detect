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
"""Plot development F1 by epoch for Appendix 3 Figure A3-1."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        help="LABEL=CSV; repeat for BERTDrug and CharBERTDrug. CSV may be training_history.csv or a TensorBoard export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RELEASE_ROOT / "artifacts" / "figures" / "Figure_A3_1_training_f1.png",
    )
    parser.add_argument("--title", default="")
    return parser.parse_args()


def parse_series(value: str) -> Tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise ValueError("--series must use LABEL=CSV")
    return label.strip(), Path(path.strip())


def read_curve(path: Path) -> Tuple[pd.Series, pd.Series]:
    frame = pd.read_csv(path)
    epoch_column = next((column for column in ["epoch", "Epoch", "Step", "step"] if column in frame), None)
    value_column = next((column for column in ["f1", "dev_f1", "Value", "value"] if column in frame), None)
    if epoch_column is None or value_column is None:
        raise ValueError(f"{path} must contain epoch/Step and f1/Value columns")
    return pd.to_numeric(frame[epoch_column], errors="raise"), pd.to_numeric(frame[value_column], errors="raise")


def main() -> int:
    args = parse_args()
    plt.figure(figsize=(10, 6))
    for value in args.series:
        label, path = parse_series(value)
        epoch, f1_value = read_curve(path)
        plt.plot(epoch, f1_value, marker="o", linewidth=2, markersize=4, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Development F1")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved training curve to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
