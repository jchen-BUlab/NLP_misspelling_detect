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
"""Create the 20 reproducible 80% training subsamples used for stability analysis."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List


RELEASE_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=RELEASE_ROOT / "data" / "classification" / "rxnorm")
    parser.add_argument("--train-file", default="train1.txt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--sample-ratio", type=float, default=0.8)
    parser.add_argument("--seed-base", type=int, default=2000)
    parser.add_argument("--stratified", action="store_true", help="Sample each label separately so every training subset preserves class balance.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def label_of(line: str) -> str:
    return line.split(maxsplit=1)[0]


def sample_lines(lines: List[str], ratio: float, seed: int, stratified: bool) -> List[str]:
    rng = random.Random(seed)
    if not stratified:
        count = max(1, int(len(lines) * ratio))
        return rng.sample(lines, count)
    groups = {}
    for line in lines:
        groups.setdefault(label_of(line), []).append(line)
    sampled = []
    for label in sorted(groups):
        group = groups[label]
        count = max(1, int(len(group) * ratio))
        sampled.extend(rng.sample(group, count))
    rng.shuffle(sampled)
    return sampled


def main() -> int:
    args = parse_args()
    if not 0 < args.sample_ratio <= 1:
        raise ValueError("--sample-ratio must be in (0, 1]")
    if args.num_runs < 1:
        raise ValueError("--num-runs must be positive")
    source_train = args.data_dir / args.train_file
    lines = source_train.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines:
        raise ValueError(f"No rows found in {source_train}")
    static_files = [name for name in ["dev.txt", "test.txt"] if (args.data_dir / name).exists()]
    manifest = {
        "source_train": str(source_train),
        "source_rows": len(lines),
        "sample_ratio": args.sample_ratio,
        "num_runs": args.num_runs,
        "seed_base": args.seed_base,
        "stratified": args.stratified,
        "runs": [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for run_id in range(1, args.num_runs + 1):
        run_dir = args.output_dir / f"run_{run_id:02d}" / "data"
        train_path = run_dir / args.train_file
        if train_path.exists() and not args.overwrite:
            raise FileExistsError(f"{train_path} already exists; pass --overwrite to replace generated samples")
        run_dir.mkdir(parents=True, exist_ok=True)
        seed = args.seed_base + run_id
        sampled = sample_lines(lines, args.sample_ratio, seed, args.stratified)
        train_path.write_text("".join(sampled), encoding="utf-8", newline="\n")
        # A generic alias makes each run usable with scripts that expect train.txt.
        (run_dir / "train.txt").write_text("".join(sampled), encoding="utf-8", newline="\n")
        for filename in static_files:
            shutil.copy2(args.data_dir / filename, run_dir / filename)
        run_record = {
            "run_id": run_id,
            "sample_seed": seed,
            "rows": len(sampled),
            "data_dir": str(run_dir.relative_to(args.output_dir)),
        }
        (run_dir.parent / "sample_manifest.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
        manifest["runs"].append(run_record)
        print(f"run_{run_id:02d}: {len(sampled):,} rows")
    (args.output_dir / "bootstrap_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
