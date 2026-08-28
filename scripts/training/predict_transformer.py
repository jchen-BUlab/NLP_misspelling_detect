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
"""Apply a saved BERTDrug/CharBERTDrug run to any labeled evaluation file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train_transformer import (
    BertBatchCollator,
    CharacterBatchCollator,
    build_model,
    evaluate,
    make_loader,
    read_labeled_file,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory created by train_transformer.py.")
    parser.add_argument("--input", type=Path, required=True, help="Labeled evaluation file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained-model", type=Path, default=None, help="Override path saved in run_config.json.")
    parser.add_argument("--tokenizer", type=Path, default=None, help="Override tokenizer path saved in run_config.json.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def path_from_config(value: object) -> Path:
    if value is None or not str(value).strip():
        raise ValueError("A required model/tokenizer path is absent from run_config.json; pass an override")
    return Path(str(value))


def main() -> int:
    args = parse_args()
    config_path = args.run_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing training configuration: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_args = SimpleNamespace(
        model_type=config["model_type"],
        pretrained_model=args.pretrained_model or path_from_config(config.get("pretrained_model")),
        tokenizer=args.tokenizer or (Path(str(config["tokenizer"])) if config.get("tokenizer") else None),
        do_lower_case=bool(config.get("do_lower_case", True)),
    )
    model, tokenizer = build_model(model_args)
    device = resolve_device(args.device)
    state = torch.load(args.run_dir / "model_best.pt", map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)

    max_length = int(config["max_length"])
    collator = (
        BertBatchCollator(tokenizer, max_length)
        if model_args.model_type == "bert"
        else CharacterBatchCollator(model_args.do_lower_case, max_length)
    )
    examples = read_labeled_file(args.input)
    batch_size = args.batch_size or int(config.get("eval_batch_size", 256))
    loader = make_loader(examples, collator, batch_size, False, args.num_workers)
    threshold = args.threshold if args.threshold is not None else float(config.get("threshold", 0.5))
    metrics, rows, probabilities, predictions, targets = evaluate(model, loader, device, threshold)
    rows.sort(key=lambda row: row["index"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "term", "target", "prediction", "probability"])
        writer.writeheader()
        writer.writerows(rows)
    np.savetxt(args.output_dir / "test_logits.txt", probabilities, fmt="%.8f")
    np.savetxt(args.output_dir / "test_predict.txt", predictions, fmt="%d")
    np.savetxt(args.output_dir / "test_targets.txt", targets, fmt="%d")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

