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
"""Fine-tune BERTmedical or CharacterBERTmedical for misspelling detection.

The manuscript training specification uses learning rate 5e-5, weight decay
0.01, and maximum development F1 for checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BasicTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)


RELEASE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = RELEASE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drug_spelling.character_indexer import CharacterIndexer
from drug_spelling.io import LabeledExample, read_labeled_file
from drug_spelling.metrics import binary_metrics
from drug_spelling.modeling import CharacterBertModel


class MedicationDataset(Dataset):
    def __init__(self, examples: Sequence[LabeledExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[int, str, int]:
        example = self.examples[index]
        return index, example.term, example.target


class BertBatchCollator:
    def __init__(self, tokenizer: BertTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: Sequence[Tuple[int, str, int]]) -> Dict[str, object]:
        indices, terms, targets = zip(*rows)
        encoded = self.tokenizer(
            list(terms),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(targets, dtype=torch.long)
        return {"indices": indices, "terms": terms, "model_inputs": encoded}


class CharacterBatchCollator:
    def __init__(self, do_lower_case: bool, max_length: int):
        self.tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.indexer = CharacterIndexer()
        self.max_length = max_length

    def tokenize(self, term: str) -> List[str]:
        tokens = self.tokenizer.tokenize(term)[: self.max_length - 2]
        return ["[CLS]"] + (tokens or [""]) + ["[SEP]"]

    def __call__(self, rows: Sequence[Tuple[int, str, int]]) -> Dict[str, object]:
        indices, terms, targets = zip(*rows)
        tokenized = [self.tokenize(term) for term in terms]
        batch_length = min(self.max_length, max(len(tokens) for tokens in tokenized))
        input_ids = self.indexer.as_padded_tensor(tokenized, maxlen=batch_length)
        attention_mask = torch.tensor(
            [[1] * min(len(tokens), batch_length) + [0] * max(0, batch_length - len(tokens)) for tokens in tokenized],
            dtype=torch.long,
        )
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": torch.zeros_like(attention_mask),
            "labels": torch.tensor(targets, dtype=torch.long),
        }
        return {"indices": indices, "terms": terms, "model_inputs": model_inputs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=["bert", "characterbert"], required=True)
    parser.add_argument("--pretrained-model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None, help="BERT tokenizer path; required for CharacterBERT.")
    parser.add_argument("--data-dir", type=Path, default=RELEASE_ROOT / "data" / "classification" / "rxnorm")
    parser.add_argument("--train-file", default="train1.txt")
    parser.add_argument("--dev-file", default="dev.txt")
    parser.add_argument("--test-file", default="test.txt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--do-lower-case", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=0, help="0 determines the maximum from the three splits, capped at 512.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device such as cuda:0.")
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def resolve_file(data_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_dir / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable")
    return device


def determine_max_length(
    examples: Iterable[LabeledExample],
    model_type: str,
    tokenizer: BertTokenizer,
    do_lower_case: bool,
) -> int:
    if model_type == "bert":
        maximum = max(len(tokenizer.tokenize(example.term)) + 2 for example in examples)
    else:
        basic = BasicTokenizer(do_lower_case=do_lower_case)
        maximum = max(len(basic.tokenize(example.term)) + 2 for example in examples)
    return min(512, max(4, maximum))


def build_model(args: argparse.Namespace) -> Tuple[BertForSequenceClassification, BertTokenizer]:
    tokenizer_path = args.tokenizer or args.pretrained_model
    if args.model_type == "characterbert" and args.tokenizer is None:
        raise ValueError("--tokenizer must point to a BERT vocabulary for CharacterBERT")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=args.do_lower_case)
    config = BertConfig.from_pretrained(args.pretrained_model, num_labels=2)
    if args.model_type == "bert":
        model = BertForSequenceClassification.from_pretrained(args.pretrained_model, config=config)
    else:
        model = BertForSequenceClassification(config)
        model.bert = CharacterBertModel.from_pretrained(args.pretrained_model, config=config)
    return model, tokenizer


def make_loader(
    examples: Sequence[LabeledExample],
    collator,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        MedicationDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )


def move_inputs(model_inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in model_inputs.items()}


def evaluate(
    model: BertForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Tuple[Dict[str, float], List[dict], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    batches = 0
    rows: List[dict] = []
    probabilities: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluation", leave=False):
            inputs = move_inputs(batch["model_inputs"], device)
            outputs = model(**inputs)
            total_loss += float(outputs.loss.detach().cpu())
            batches += 1
            probs = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()
            labels = inputs["labels"].detach().cpu().numpy()
            probabilities.append(probs)
            targets.append(labels)
            for index, term, target, probability in zip(batch["indices"], batch["terms"], labels, probs[:, 1]):
                rows.append(
                    {
                        "index": int(index),
                        "term": term,
                        "target": int(target),
                        "prediction": int(probability >= threshold),
                        "probability": float(probability),
                    }
                )
    probability_array = np.concatenate(probabilities, axis=0)
    target_array = np.concatenate(targets, axis=0)
    prediction_array = (probability_array[:, 1] >= threshold).astype(np.int64)
    metrics = binary_metrics(target_array, prediction_array, probability_array[:, 1])
    metrics["loss"] = total_loss / max(1, batches)
    return metrics, rows, probability_array, prediction_array, target_array


def is_better_f1(value: float, best: float | None) -> bool:
    return best is None or value > best


def save_history(path: Path, history: Sequence[Dict[str, float]]) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite_output:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; pass --overwrite-output to reuse it")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_examples = read_labeled_file(resolve_file(args.data_dir, args.train_file))
    dev_examples = read_labeled_file(resolve_file(args.data_dir, args.dev_file))
    test_examples = read_labeled_file(resolve_file(args.data_dir, args.test_file))
    model, tokenizer = build_model(args)
    if args.max_length <= 0:
        args.max_length = determine_max_length(
            train_examples + dev_examples + test_examples,
            args.model_type,
            tokenizer,
            args.do_lower_case,
        )
    collator = (
        BertBatchCollator(tokenizer, args.max_length)
        if args.model_type == "bert"
        else CharacterBatchCollator(args.do_lower_case, args.max_length)
    )
    train_loader = make_loader(train_examples, collator, args.train_batch_size, True, args.num_workers)
    dev_loader = make_loader(dev_examples, collator, args.eval_batch_size, False, args.num_workers)
    test_loader = make_loader(test_examples, collator, args.eval_batch_size, False, args.num_workers)

    model.to(device)
    no_decay = {"bias", "LayerNorm.weight"}
    parameter_groups = [
        {
            "params": [parameter for name, parameter in model.named_parameters() if not any(key in name for key in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [parameter for name, parameter in model.named_parameters() if any(key in name for key in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.learning_rate, eps=args.adam_epsilon)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_updates),
        num_training_steps=total_updates,
    )

    config_payload = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device_resolved": str(device),
        "train_rows": len(train_examples),
        "dev_rows": len(dev_examples),
        "test_rows": len(test_examples),
        "checkpoint_metric": "f1",
        "checkpoint_mode": "max",
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    best_value: float | None = None
    best_epoch = -1
    history: List[Dict[str, float]] = []
    global_step = 0
    optimizer.zero_grad()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"), start=1):
            inputs = move_inputs(batch["model_inputs"], device)
            outputs = model(**inputs)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps
            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        dev_metrics, _rows, _probs, _preds, _targets = evaluate(model, dev_loader, device, args.threshold)
        history_row = {"epoch": epoch, "train_loss": running_loss / max(1, len(train_loader)), **dev_metrics}
        history.append(history_row)
        save_history(args.output_dir / "training_history.csv", history)
        value = float(dev_metrics["f1"])
        print(f"epoch={epoch} dev_f1={value:.6f}")
        if is_better_f1(value, best_value):
            best_value = value
            best_epoch = epoch
            torch.save(model.state_dict(), args.output_dir / "model_best.pt")
            best_model_dir = args.output_dir / "best_model"
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            (args.output_dir / "best_checkpoint.json").write_text(
                json.dumps(
                    {
                        "epoch": best_epoch,
                        "metric": "f1",
                        "mode": "max",
                        "value": best_value,
                        "dev_metrics": dev_metrics,
                    },
                    indent=2,
                    allow_nan=True,
                ),
                encoding="utf-8",
            )

    state = torch.load(args.output_dir / "model_best.pt", map_location=device)
    model.load_state_dict(state, strict=True)
    test_metrics, test_rows, test_probs, test_predictions, test_targets = evaluate(
        model, test_loader, device, args.threshold
    )
    test_rows.sort(key=lambda row: row["index"])
    with (args.output_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "term", "target", "prediction", "probability"])
        writer.writeheader()
        writer.writerows(test_rows)
    np.savetxt(args.output_dir / "test_logits.txt", test_probs, fmt="%.8f")
    np.savetxt(args.output_dir / "test_predict.txt", test_predictions, fmt="%d")
    np.savetxt(args.output_dir / "test_targets.txt", test_targets, fmt="%d")
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps({"best_epoch": best_epoch, "best_value": best_value, "test": test_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
