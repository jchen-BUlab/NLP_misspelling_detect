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
"""Summarize development metrics across RxNorm training multipliers (Appendix 3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="BERTDrug or CharBERTDrug")
    parser.add_argument("--run", action="append", required=True, help="K=RUN_DIR; repeat for 1,2,4,6,8,10.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_run(value: str) -> tuple[int, Path]:
    multiplier, separator, path = value.partition("=")
    if not separator:
        raise ValueError("--run must use K=RUN_DIR")
    return int(multiplier), Path(path)


def main() -> int:
    args = parse_args()
    rows = []
    for value in args.run:
        multiplier, run_dir = parse_run(value)
        checkpoint_path = run_dir / "best_checkpoint.json"
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        metrics = payload["dev_metrics"]
        rows.append(
            {
                "model": args.model,
                "training_multiplier": multiplier,
                "selected_epoch": payload["epoch"],
                "selection_metric": payload["metric"],
                **metrics,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("training_multiplier").to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} training-sweep rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

