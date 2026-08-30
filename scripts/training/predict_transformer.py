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
from typing import Mapping

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


BENIGN_MISSING_STATE_KEYS = {"bert.embeddings.position_ids"}
BENIGN_UNEXPECTED_STATE_KEYS = {"bert.embeddings.position_ids"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing model_best.pt and, optionally, run_config.json.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Labeled evaluation file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-type",
        choices=["bert", "characterbert"],
        default=None,
        help="Required when run_config.json is not present.",
    )
    parser.add_argument("--pretrained-model", type=Path, default=None, help="Override path saved in run_config.json.")
    parser.add_argument("--tokenizer", type=Path, default=None, help="Override tokenizer path saved in run_config.json.")
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Required when run_config.json is not present.",
    )
    parser.add_argument(
        "--do-lower-case",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override preprocessing saved in run_config.json; defaults to enabled.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_run_config(run_dir: Path) -> Mapping[str, object]:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return {}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a JSON object in {config_path}")
    return config


def required_setting(cli_value: object, config: Mapping[str, object], key: str, option: str) -> object:
    value = cli_value if cli_value is not None else config.get(key)
    if value is None or not str(value).strip():
        raise ValueError(
            f"{key} is absent from run_config.json; pass {option} for a weight-only checkpoint"
        )
    return value


def normalize_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    if all(not key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_checkpoint(path: Path, device: torch.device) -> Mapping[str, torch.Tensor]:
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older supported PyTorch releases.
        state_dict = torch.load(path, map_location=device)
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Expected a state dictionary in {path}")
    return normalize_state_dict_keys(state_dict)


def apply_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    state_dict = load_checkpoint(checkpoint_path, device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    bad_missing = [key for key in missing_keys if key not in BENIGN_MISSING_STATE_KEYS]
    bad_unexpected = [key for key in unexpected_keys if key not in BENIGN_UNEXPECTED_STATE_KEYS]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Checkpoint did not match the selected model architecture.\n"
            f"Checkpoint: {checkpoint_path}\n"
            f"Missing keys: {bad_missing}\n"
            f"Unexpected keys: {bad_unexpected}"
        )


def main() -> int:
    args = parse_args()
    config = load_run_config(args.run_dir)
    model_type = str(required_setting(args.model_type, config, "model_type", "--model-type"))
    pretrained_model = Path(
        str(required_setting(args.pretrained_model, config, "pretrained_model", "--pretrained-model"))
    )
    tokenizer_value = args.tokenizer if args.tokenizer is not None else config.get("tokenizer")
    tokenizer = (
        Path(str(tokenizer_value))
        if tokenizer_value is not None and str(tokenizer_value).strip()
        else None
    )
    do_lower_case = (
        args.do_lower_case
        if args.do_lower_case is not None
        else bool(config.get("do_lower_case", True))
    )
    max_length = int(required_setting(args.max_length, config, "max_length", "--max-length"))
    if max_length < 4:
        raise ValueError("--max-length must be at least 4")
    model_args = SimpleNamespace(
        model_type=model_type,
        pretrained_model=pretrained_model,
        tokenizer=tokenizer,
        do_lower_case=do_lower_case,
    )
    model, tokenizer = build_model(model_args)
    device = resolve_device(args.device)
    checkpoint_path = args.run_dir / "model_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    apply_checkpoint(model, checkpoint_path, device)
    model.to(device)

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
