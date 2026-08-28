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

"""Shared readers and writers for the paper's labeled text and prediction files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


LABEL_PREFIX = "__label__"


@dataclass(frozen=True)
class LabeledExample:
    """One medication term and its binary misspelling label."""

    term: str
    target: int

    @property
    def label(self) -> str:
        return "positive" if self.target == 1 else "negative"


def parse_binary_target(value: object) -> int:
    """Parse common binary label representations used by release inputs."""

    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "positive", "misspelled", "mispelled", "__label__positive"}:
        return 1
    if text in {"0", "0.0", "false", "negative", "correct", "__label__negative"}:
        return 0
    raise ValueError(f"Could not parse a binary target from {value!r}")


def parse_labeled_line(line: str) -> LabeledExample:
    """Parse ``__label__positive term`` or ``__label__negative term``."""

    stripped = line.strip()
    if not stripped:
        raise ValueError("Empty labeled line")
    label_token, separator, term = stripped.partition(" ")
    if not separator or not term.strip():
        raise ValueError(f"Expected '<label> <term>', got {line!r}")
    if not label_token.startswith(LABEL_PREFIX):
        raise ValueError(f"Expected a {LABEL_PREFIX!r} prefix, got {label_token!r}")
    label = label_token[len(LABEL_PREFIX) :].lower()
    if label not in {"positive", "negative"}:
        raise ValueError(f"Unsupported label {label!r}")
    return LabeledExample(term=term.strip(), target=int(label == "positive"))


def read_labeled_file(path: Path | str) -> List[LabeledExample]:
    """Read a labeled UTF-8 text file, rejecting malformed nonempty lines."""

    source = Path(path)
    examples: List[LabeledExample] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                examples.append(parse_labeled_line(line))
            except ValueError as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
    if not examples:
        raise ValueError(f"No labeled examples found in {source}")
    return examples


def write_labeled_file(
    path: Path | str,
    examples: Iterable[LabeledExample],
    *,
    lower_case: bool = True,
) -> int:
    """Write examples in fastText label format and return the row count."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            term = " ".join(example.term.split())
            if lower_case:
                term = term.lower()
            handle.write(f"{LABEL_PREFIX}{example.label} {term}\n")
            row_count += 1
    return row_count
