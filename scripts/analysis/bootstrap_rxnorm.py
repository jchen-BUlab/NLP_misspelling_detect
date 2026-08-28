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

#!/usr/bin/env python
"""Two-way bootstrap RxNorm model metrics and confidence intervals.

The script reads RxNorm test-sample metadata, discovers 20 saved prediction
runs per method, and writes metric tables for all, generic, and branded rows.

Run discovery uses the documented per-run directory convention:
<run_prefix>_run_* -> embedding directory -> newest timestamped output
directory -> test prediction/score files.
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import sklearn.metrics as sklearn_metrics

from bootstrap_statistics import two_way_bootstrap_f1


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUN_ID_PATTERN = re.compile(r"_run_(\d+)$")
PREDICTION_CANDIDATES = ("test_predict.txt", "predict.txt", "test_predict_label.txt")
SCORE_CANDIDATES = ("test_logits.txt", "logits.txt")
EDIT_DISTANCE_CANDIDATES = ("test_edit_distance.txt", "edit_distance.txt")

DEFAULT_MAX_WORKERS = 8

_BOOTSTRAP_WORKER_CONTEXT: Dict[str, object] = {}


@dataclass(frozen=True)
class MethodSpec:
    model_name: str
    run_prefix: str
    embedding_dir_name: str


SUBSET_SPECS = OrderedDict(
    [
        ("all", None),
        ("IN", {"IN"}),
        ("BN", {"BN"}),
    ]
)

DEFAULT_METHOD_SPECS = (
    MethodSpec(
        "CharBERTDrug",
        "charbertdrug_bootstrap",
        "CharBERTDrug",
    ),
    MethodSpec(
        "BERTDrug",
        "bertdrug_bootstrap",
        "BERTDrug",
    ),
    MethodSpec(
        "SpellChecker",
        "baselines_bootstrap",
        "SpellChecker",
    ),
    MethodSpec(
        "fastTextML",
        "baselines_bootstrap",
        "fastTextML",
    ),
    MethodSpec(
        "BioWordVecML",
        "baselines_bootstrap",
        "BioWordVecML",
    ),
)

TABLE_MODEL_ORDER = OrderedDict(
    [
        ("SpellChecker", "SpellChecker"),
        ("CharBERTDrug", "CharBERTDrug"),
        ("BERTDrug", "BERTDrug"),
        ("fastTextML", "fastTextML"),
        ("BioWordVecML", "BioWordVecML"),
    ]
)

METRIC_DISPLAY_NAMES = OrderedDict(
    [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("accuracy", "Accuracy"),
        ("specificity", "Specificity"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("brier_score", "Brier Score"),
    ]
)
THRESHOLD_METRICS = ("precision", "recall", "f1", "accuracy", "specificity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-way bootstrap model metrics and CIs for RxNorm test samples."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/predictions/rxnorm_test_predictions.csv"),
        help="Input CSV with target and type_norm columns aligned to saved test outputs.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("artifacts/bootstrap_models"),
        help="Root folder containing per-run experiment folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/results/bootstrap_rxnorm"),
        help="Directory where all/IN/BN result subfolders are written.",
    )
    parser.add_argument(
        "--method-spec",
        action="append",
        default=[],
        help=(
            "Optional override in the form 'model_name|run_prefix|embedding_dir_name'. "
            "May be passed multiple times. If omitted, the five documented "
            "20-run release-layout defaults are used."
        ),
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=20,
        help="Expected number of trained runs for each method.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Number of two-way bootstrap samples.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Number of multiprocessing workers for bootstrap computation. "
            "Use 0 for auto, 1 for sequential execution."
        ),
    )
    parser.add_argument(
        "--bootstrap-chunk-size",
        type=int,
        default=0,
        help=(
            "Number of bootstrap iterations per worker task. "
            "Use 0 for an automatic chunk size."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable bootstrap progress bars.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=123,
        help="Random seed used for model and test-sample bootstrap resampling.",
    )
    parser.add_argument(
        "--metric-set",
        choices=["threshold", "all"],
        default="all",
        help=(
            "'threshold' computes precision/recall/F1/accuracy/specificity. "
            "'all' also computes ROC-AUC, PR-AUC, and Brier score."
        ),
    )
    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=0.5,
        help="Threshold applied to transformer/ML score files. Defaults to the manuscript value 0.5.",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def parse_method_specs(raw_specs: Sequence[str]) -> List[MethodSpec]:
    if not raw_specs:
        return list(DEFAULT_METHOD_SPECS)

    specs: List[MethodSpec] = []
    seen = set()
    for raw_spec in raw_specs:
        parts = [part.strip() for part in raw_spec.split("|")]
        if len(parts) != 3 or any(not part for part in parts):
            raise ValueError(
                "Invalid --method-spec. Expected 'model_name|run_prefix|embedding_dir_name'."
            )
        spec = MethodSpec(parts[0], parts[1], parts[2])
        key = (spec.model_name, spec.run_prefix, spec.embedding_dir_name)
        if key not in seen:
            specs.append(spec)
            seen.add(key)

    model_names = [spec.model_name for spec in specs]
    if len(model_names) != len(set(model_names)):
        raise ValueError("Each --method-spec must use a unique model_name.")
    return specs


def metric_keys_for_args(args: argparse.Namespace) -> List[str]:
    if args.metric_set == "threshold":
        return list(THRESHOLD_METRICS)
    return list(METRIC_DISPLAY_NAMES.keys())


def resolve_num_workers(requested_workers: int, n_bootstrap: int) -> int:
    if n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be positive.")
    if requested_workers < 0:
        raise ValueError("--num-workers must be 0 or a positive integer.")
    if requested_workers == 1:
        return 1
    if requested_workers > 1:
        return min(requested_workers, n_bootstrap)

    cpu_count = mp.cpu_count()
    return max(1, min(DEFAULT_MAX_WORKERS, cpu_count, n_bootstrap))


def resolve_chunk_size(requested_chunk_size: int, n_bootstrap: int, num_workers: int) -> int:
    if requested_chunk_size < 0:
        raise ValueError("--bootstrap-chunk-size must be 0 or a positive integer.")
    if requested_chunk_size > 0:
        return requested_chunk_size
    target_chunks = max(1, num_workers * 8)
    return max(1, math.ceil(n_bootstrap / target_chunks))


def iter_with_progress(
    iterable: Iterable[object],
    *,
    total: int,
    description: str,
    enabled: bool,
) -> Iterable[object]:
    if not enabled:
        yield from iterable
        return

    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:
        print(f"{description}: 0/{total}")
        report_every = max(1, total // 20)
        for index, item in enumerate(iterable, start=1):
            if index == total or index % report_every == 0:
                print(f"{description}: {index}/{total}")
            yield item
        return

    yield from tqdm(iterable, total=total, desc=description)


def parse_target(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "positive", "__label__positive", "true"}:
        return 1
    if text in {"0", "0.0", "negative", "__label__negative", "false"}:
        return 0
    raise ValueError(f"Could not parse target value: {value!r}")


def load_rxnorm_samples(input_path: Path) -> List[Dict[str, str]]:
    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} has no header row")

        if "target" not in reader.fieldnames:
            raise ValueError(f"{input_path} is missing required column: target")
        if "type_norm" not in reader.fieldnames and "term_type" not in reader.fieldnames:
            raise ValueError(f"{input_path} must contain type_norm or term_type")

        rows = [dict(row) for row in reader]

    for row in rows:
        if not str(row.get("type_norm", "")).strip():
            term_type = str(row.get("term_type", "")).strip().lower()
            if term_type in {"brand", "branded", "bn"}:
                row["type_norm"] = "BN"
            elif term_type in {"generic", "non-branded", "nonbranded", "in"}:
                row["type_norm"] = "IN"

    if not rows:
        raise ValueError(f"No rows found in {input_path}")
    return rows


def filter_indices_by_type_norm(
    rows: Sequence[Mapping[str, str]],
    allowed_type_norms: Optional[set[str]],
) -> List[int]:
    if allowed_type_norms is None:
        return list(range(len(rows)))

    indices = [
        index
        for index, row in enumerate(rows)
        if str(row.get("type_norm", "")).strip() in allowed_type_norms
    ]
    if not indices:
        raise ValueError(f"No rows found for type_norm values: {sorted(allowed_type_norms)}")
    return indices


def extract_run_id(run_dir_name: str) -> int:
    match = RUN_ID_PATTERN.search(run_dir_name)
    if not match:
        raise ValueError(f"Could not parse run id from directory name: {run_dir_name}")
    return int(match.group(1))


def find_prediction_file(output_dir: Path) -> Path:
    for candidate in PREDICTION_CANDIDATES:
        path = output_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"No prediction file found in {output_dir}")


def find_score_file(output_dir: Path) -> Optional[Path]:
    for candidate in SCORE_CANDIDATES:
        path = output_dir / candidate
        if path.exists():
            return path
    return None


def find_edit_distance_file(output_dir: Path) -> Optional[Path]:
    for candidate in EDIT_DISTANCE_CANDIDATES:
        path = output_dir / candidate
        if path.exists():
            return path
    return None


def load_vector(path: Path) -> np.ndarray:
    values = np.loadtxt(path)
    return np.asarray(values)


def load_predictions(prediction_file: Path) -> np.ndarray:
    try:
        predictions = load_vector(prediction_file)
        return np.asarray(predictions, dtype=np.int64).reshape(-1)
    except ValueError:
        with prediction_file.open("r", encoding="utf-8") as handle:
            raw_values = [line.strip() for line in handle if line.strip()]
        mapped = []
        for value in raw_values:
            lowered = value.lower()
            if lowered == "positive":
                mapped.append(1)
            elif lowered == "negative":
                mapped.append(0)
            else:
                mapped.append(int(value))
        return np.asarray(mapped, dtype=np.int64).reshape(-1)


def load_scores(score_file: Optional[Path], predictions: np.ndarray) -> np.ndarray:
    if score_file is None:
        return predictions.astype(np.float64)

    scores = load_vector(score_file)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        return scores.reshape(-1)
    if scores.ndim == 2:
        if scores.shape[1] == 1:
            return scores[:, 0]
        return scores[:, 1]
    raise ValueError(f"Unsupported score shape in {score_file}: {scores.shape}")


def is_spellchecker_model(model_name: str) -> bool:
    return model_name.lower().startswith("spellchecker")


def metric_keys_for_model(model_name: str, metric_keys: Sequence[str]) -> List[str]:
    """Return manuscript metrics available for a model.

    SpellChecker provides a ranking score rather than a probability, so a
    Brier score is neither computed nor emitted for that model.
    """

    return [
        metric_key
        for metric_key in metric_keys
        if not (is_spellchecker_model(model_name) and metric_key == "brier_score")
    ]


def discover_run_outputs(
    results_root: Path,
    spec: MethodSpec,
    expected_runs: int,
) -> List[Dict[str, object]]:
    run_dirs = sorted(
        path for path in results_root.glob(f"{spec.run_prefix}_run_*") if path.is_dir()
    )
    if len(run_dirs) != expected_runs:
        raise ValueError(
            f"{spec.model_name}: expected {expected_runs} run directories matching "
            f"{spec.run_prefix}_run_* under {results_root}, found {len(run_dirs)}."
        )

    run_outputs: List[Dict[str, object]] = []
    seen_run_ids = set()
    for run_dir in run_dirs:
        run_id = extract_run_id(run_dir.name)
        if run_id in seen_run_ids:
            raise ValueError(f"{spec.model_name}: duplicate run id {run_id}.")
        seen_run_ids.add(run_id)

        embedding_dir = run_dir / spec.embedding_dir_name
        if not embedding_dir.exists():
            raise FileNotFoundError(
                f"{spec.model_name}: missing embedding directory {embedding_dir}"
            )
        output_candidates = [path for path in embedding_dir.iterdir() if path.is_dir()]
        if not output_candidates:
            raise FileNotFoundError(
                f"{spec.model_name}: no timestamped output directories under {embedding_dir}"
            )
        output_dir = max(output_candidates, key=lambda path: path.stat().st_mtime)
        prediction_file = find_prediction_file(output_dir)
        score_file = find_score_file(output_dir)
        edit_distance_file = find_edit_distance_file(output_dir)

        run_outputs.append(
            {
                "model": spec.model_name,
                "run_id": run_id,
                "run_dir": run_dir,
                "output_dir": output_dir,
                "prediction_file": prediction_file,
                "score_file": score_file,
                "edit_distance_file": edit_distance_file,
            }
        )

    return sorted(run_outputs, key=lambda item: int(item["run_id"]))


def validate_vector_length(
    *,
    model_name: str,
    run_id: object,
    vector_name: str,
    vector: np.ndarray,
    expected_samples: int,
    source_path: Path,
) -> None:
    if vector.shape[0] != expected_samples:
        raise ValueError(
            f"{model_name} run {run_id}: {vector_name} length {vector.shape[0]} "
            f"does not match input sample count {expected_samples}. File: {source_path}"
        )


def load_model_matrices_from_runs(
    *,
    results_root: Path,
    method_specs: Sequence[MethodSpec],
    expected_runs: int,
    expected_samples: int,
    args: argparse.Namespace,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, List[Dict[str, object]]],
]:
    predictions_by_model: Dict[str, np.ndarray] = OrderedDict()
    scores_by_model: Dict[str, np.ndarray] = OrderedDict()
    brier_scores_by_model: Dict[str, np.ndarray] = OrderedDict()
    run_outputs_by_model: Dict[str, List[Dict[str, object]]] = OrderedDict()
    for spec in method_specs:
        run_outputs = discover_run_outputs(
            results_root=results_root,
            spec=spec,
            expected_runs=expected_runs,
        )

        run_predictions: List[np.ndarray] = []
        run_scores: List[np.ndarray] = []
        run_brier_scores: List[np.ndarray] = []

        for run_output in run_outputs:
            run_id = run_output["run_id"]
            prediction_file = Path(run_output["prediction_file"])
            predictions = load_predictions(prediction_file)
            validate_vector_length(
                model_name=spec.model_name,
                run_id=run_id,
                vector_name="prediction",
                vector=predictions,
                expected_samples=expected_samples,
                source_path=prediction_file,
            )
            unique_values = set(np.unique(predictions).tolist())
            if not unique_values.issubset({0, 1}):
                raise ValueError(
                    f"{spec.model_name} run {run_id}: predictions must be binary 0/1 "
                    f"values, found {sorted(unique_values)} in {prediction_file}."
                )

            score_file = Path(run_output["score_file"]) if run_output["score_file"] else None
            edit_distance_file = (
                Path(run_output["edit_distance_file"])
                if run_output["edit_distance_file"]
                else None
            )
            score_source_file = score_file
            if is_spellchecker_model(spec.model_name):
                if edit_distance_file is None:
                    raise FileNotFoundError(
                        f"{spec.model_name} run {run_id}: missing edit-distance file "
                        f"under {run_output['output_dir']}"
                    )
                score_source_file = edit_distance_file

            scores = load_scores(score_source_file, predictions)
            validate_vector_length(
                model_name=spec.model_name,
                run_id=run_id,
                vector_name="score",
                vector=scores,
                expected_samples=expected_samples,
                source_path=score_source_file or prediction_file,
            )

            if is_spellchecker_model(spec.model_name):
                predictions = (scores > 0).astype(np.int64)
            else:
                predictions = (scores >= args.probability_threshold).astype(np.int64)

            brier_scores = load_scores(score_file, predictions)
            brier_source_file = score_file or prediction_file
            if is_spellchecker_model(spec.model_name):
                if edit_distance_file is None:
                    raise FileNotFoundError(
                        f"{spec.model_name} run {run_id}: missing edit-distance file "
                        f"under {run_output['output_dir']}"
                    )
                edit_distances = load_vector(edit_distance_file)
                brier_scores = np.full(edit_distances.shape, np.nan, dtype=np.float64)
                brier_source_file = edit_distance_file

            validate_vector_length(
                model_name=spec.model_name,
                run_id=run_id,
                vector_name="Brier score",
                vector=brier_scores,
                expected_samples=expected_samples,
                source_path=brier_source_file,
            )

            run_predictions.append(predictions.astype(np.int8, copy=False))
            run_scores.append(scores.astype(np.float64, copy=False))
            run_brier_scores.append(brier_scores.astype(np.float64, copy=False))

        predictions_by_model[spec.model_name] = np.stack(run_predictions, axis=0)
        scores_by_model[spec.model_name] = np.stack(run_scores, axis=0)
        brier_scores_by_model[spec.model_name] = np.stack(run_brier_scores, axis=0)
        run_outputs_by_model[spec.model_name] = run_outputs

    return predictions_by_model, scores_by_model, brier_scores_by_model, run_outputs_by_model


def finite_float(value: object) -> object:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    return "nan" if math.isnan(numeric) else numeric


def mean_finite(values: Iterable[float]) -> float:
    finite_values = np.asarray([value for value in values if not math.isnan(float(value))], dtype=float)
    if finite_values.size == 0:
        return float("nan")
    return float(np.mean(finite_values))


def std_finite(values: Iterable[float]) -> float:
    finite_values = np.asarray([value for value in values if not math.isnan(float(value))], dtype=float)
    if finite_values.size < 2:
        return float("nan")
    return float(np.std(finite_values, ddof=1))


def quantile_ci(values: Iterable[float]) -> Tuple[float, float]:
    finite_values = np.asarray([value for value in values if not math.isnan(float(value))], dtype=float)
    if finite_values.size == 0:
        return float("nan"), float("nan")
    lower, upper = np.quantile(finite_values, [0.025, 0.975])
    return float(lower), float(upper)


def format_run_ids(run_outputs: Sequence[Mapping[str, object]]) -> str:
    return ";".join(str(int(run_output["run_id"])) for run_output in run_outputs)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator != 0,
    )


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray, weights: np.ndarray) -> float:
    positive_weight = float(weights[y_true == 1].sum())
    negative_weight = float(weights[y_true == 0].sum())
    if positive_weight <= 0.0 or negative_weight <= 0.0:
        return float("nan")
    mask = weights > 0
    try:
        return float(
            sklearn_metrics.roc_auc_score(
                y_true[mask],
                y_score[mask],
                sample_weight=weights[mask],
            )
        )
    except ValueError:
        return float("nan")


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray, weights: np.ndarray) -> float:
    if float(weights[y_true == 1].sum()) <= 0.0:
        return float("nan")
    mask = weights > 0
    try:
        return float(
            sklearn_metrics.average_precision_score(
                y_true[mask],
                y_score[mask],
                sample_weight=weights[mask],
            )
        )
    except ValueError:
        return float("nan")


def compute_run_metric_scores(
    *,
    y_true: np.ndarray,
    prediction_matrix: np.ndarray,
    score_matrix: np.ndarray,
    brier_score_matrix: np.ndarray,
    weights: np.ndarray,
    metric_keys: Sequence[str],
) -> Dict[str, np.ndarray]:
    y_true_int = np.asarray(y_true, dtype=np.int64)
    weights_int = np.asarray(weights, dtype=np.int64)
    prediction_int = np.asarray(prediction_matrix, dtype=np.int64)

    positive = y_true_int
    negative = 1 - y_true_int
    predicted_positive = prediction_int
    predicted_negative = 1 - prediction_int

    tp = (predicted_positive * positive) @ weights_int
    fp = (predicted_positive * negative) @ weights_int
    tn = (predicted_negative * negative) @ weights_int
    fn = (predicted_negative * positive) @ weights_int
    total = float(weights_int.sum())

    metric_scores: Dict[str, np.ndarray] = {}
    if "precision" in metric_keys:
        metric_scores["precision"] = safe_divide(tp, tp + fp)
    if "recall" in metric_keys:
        metric_scores["recall"] = safe_divide(tp, tp + fn)
    if "f1" in metric_keys:
        metric_scores["f1"] = safe_divide(2 * tp, (2 * tp) + fp + fn)
    if "accuracy" in metric_keys:
        metric_scores["accuracy"] = ((tp + tn) / total).astype(np.float64)
    if "specificity" in metric_keys:
        metric_scores["specificity"] = safe_divide(tn, tn + fp)
    if "brier_score" in metric_keys:
        squared_errors = (np.asarray(brier_score_matrix, dtype=np.float64) - y_true_int) ** 2
        metric_scores["brier_score"] = (squared_errors @ weights_int) / total

    if "roc_auc" in metric_keys or "pr_auc" in metric_keys:
        score_matrix_float = np.asarray(score_matrix, dtype=np.float64)
        roc_values = []
        pr_values = []
        for run_index in range(score_matrix_float.shape[0]):
            if "roc_auc" in metric_keys:
                roc_values.append(
                    safe_roc_auc(y_true_int, score_matrix_float[run_index], weights_int)
                )
            if "pr_auc" in metric_keys:
                pr_values.append(
                    safe_pr_auc(y_true_int, score_matrix_float[run_index], weights_int)
                )
        if "roc_auc" in metric_keys:
            metric_scores["roc_auc"] = np.asarray(roc_values, dtype=np.float64)
        if "pr_auc" in metric_keys:
            metric_scores["pr_auc"] = np.asarray(pr_values, dtype=np.float64)

    return metric_scores


def compute_full_metric_scores_by_model(
    *,
    y_true: np.ndarray,
    predictions_by_model: Mapping[str, np.ndarray],
    scores_by_model: Mapping[str, np.ndarray],
    brier_scores_by_model: Mapping[str, np.ndarray],
    metric_keys: Sequence[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    weights = np.ones(len(y_true), dtype=np.int64)
    return {
        model_name: compute_run_metric_scores(
            y_true=y_true,
            prediction_matrix=predictions_by_model[model_name],
            score_matrix=scores_by_model[model_name],
            brier_score_matrix=brier_scores_by_model[model_name],
            weights=weights,
            metric_keys=metric_keys_for_model(model_name, metric_keys),
        )
        for model_name in predictions_by_model
    }


def make_bootstrap_seed_chunks(
    *,
    n_bootstrap: int,
    random_state: int,
    chunk_size: int,
) -> List[Tuple[int, List[int]]]:
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(
        low=0,
        high=np.iinfo(np.uint32).max,
        size=n_bootstrap,
        dtype=np.uint32,
    )
    chunks: List[Tuple[int, List[int]]] = []
    for start in range(0, n_bootstrap, chunk_size):
        seed_chunk = [int(seed) for seed in seeds[start : start + chunk_size]]
        chunks.append((start + 1, seed_chunk))
    return chunks


def init_bootstrap_worker(
    y_true: np.ndarray,
    predictions_by_model: Mapping[str, np.ndarray],
    scores_by_model: Mapping[str, np.ndarray],
    brier_scores_by_model: Mapping[str, np.ndarray],
    model_names: Sequence[str],
    metric_keys: Sequence[str],
) -> None:
    _BOOTSTRAP_WORKER_CONTEXT.clear()
    _BOOTSTRAP_WORKER_CONTEXT.update(
        {
            "y_true": np.asarray(y_true, dtype=np.int64),
            "predictions_by_model": dict(predictions_by_model),
            "scores_by_model": dict(scores_by_model),
            "brier_scores_by_model": dict(brier_scores_by_model),
            "model_names": list(model_names),
            "metric_keys": list(metric_keys),
        }
    )


def compute_bootstrap_chunk(
    task: Tuple[int, Sequence[int]],
) -> List[Dict[str, object]]:
    if not _BOOTSTRAP_WORKER_CONTEXT:
        raise RuntimeError("Bootstrap worker context has not been initialized.")

    start_bootstrap_id, seeds = task
    y_true = _BOOTSTRAP_WORKER_CONTEXT["y_true"]
    predictions_by_model = _BOOTSTRAP_WORKER_CONTEXT["predictions_by_model"]
    scores_by_model = _BOOTSTRAP_WORKER_CONTEXT["scores_by_model"]
    brier_scores_by_model = _BOOTSTRAP_WORKER_CONTEXT["brier_scores_by_model"]
    model_names = _BOOTSTRAP_WORKER_CONTEXT["model_names"]
    metric_keys = _BOOTSTRAP_WORKER_CONTEXT["metric_keys"]

    assert isinstance(y_true, np.ndarray)
    assert isinstance(predictions_by_model, dict)
    assert isinstance(scores_by_model, dict)
    assert isinstance(brier_scores_by_model, dict)
    assert isinstance(model_names, list)
    assert isinstance(metric_keys, list)

    model_rows: List[Dict[str, object]] = []
    for model_name in model_names:
        model_metric_keys = metric_keys_for_model(model_name, metric_keys)
        model_summary = two_way_bootstrap_f1(
            y_true=y_true,
            predictions=predictions_by_model[model_name],
            n_bootstrap=len(seeds),
            metric_names=model_metric_keys,
            scores=scores_by_model[model_name],
            brier_scores=brier_scores_by_model[model_name],
            bootstrap_seeds=seeds,
        )
        for metric_key in model_metric_keys:
            values = np.asarray(
                model_summary["metrics"][metric_key]["bootstrap_values"],
                dtype=np.float64,
            )
            for offset, value in enumerate(values):
                model_rows.append(
                    {
                        "bootstrap_id": start_bootstrap_id + offset,
                        "model": model_name,
                        "metric": metric_key,
                        "metric_display": METRIC_DISPLAY_NAMES[metric_key],
                        "value": float(value),
                    }
                )

    return model_rows


def compute_bootstrap_outputs(
    *,
    subset_name: str,
    y_true: np.ndarray,
    predictions_by_model: Mapping[str, np.ndarray],
    scores_by_model: Mapping[str, np.ndarray],
    brier_scores_by_model: Mapping[str, np.ndarray],
    metric_keys: Sequence[str],
    n_bootstrap: int,
    random_state: int,
    num_workers: int,
    chunk_size: int,
    show_progress: bool,
) -> Tuple[
    List[Dict[str, object]],
    Dict[str, Dict[str, List[float]]],
]:
    model_names = list(predictions_by_model.keys())
    seed_chunks = make_bootstrap_seed_chunks(
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        chunk_size=chunk_size,
    )

    model_rows: List[Dict[str, object]] = []
    progress_description = f"{subset_name} bootstrap"

    if num_workers == 1:
        init_bootstrap_worker(
            y_true,
            predictions_by_model,
            scores_by_model,
            brier_scores_by_model,
            model_names,
            metric_keys,
        )
        result_iterable = map(compute_bootstrap_chunk, seed_chunks)
        for chunk_model_rows in iter_with_progress(
            result_iterable,
            total=len(seed_chunks),
            description=progress_description,
            enabled=show_progress,
        ):
            model_rows.extend(chunk_model_rows)
    else:
        with mp.Pool(
            processes=num_workers,
            initializer=init_bootstrap_worker,
            initargs=(
                y_true,
                predictions_by_model,
                scores_by_model,
                brier_scores_by_model,
                model_names,
                metric_keys,
            ),
        ) as pool:
            result_iterable = pool.imap_unordered(compute_bootstrap_chunk, seed_chunks)
            for chunk_model_rows in iter_with_progress(
                result_iterable,
                total=len(seed_chunks),
                description=progress_description,
                enabled=show_progress,
            ):
                model_rows.extend(chunk_model_rows)

    model_order = {model_name: index for index, model_name in enumerate(model_names)}
    metric_order = {metric_key: index for index, metric_key in enumerate(metric_keys)}
    model_rows.sort(
        key=lambda row: (
            int(row["bootstrap_id"]),
            model_order[str(row["model"])],
            metric_order[str(row["metric"])],
        )
    )

    values_by_model_metric: Dict[str, Dict[str, List[float]]] = {
        model_name: {
            metric_key: []
            for metric_key in metric_keys_for_model(model_name, metric_keys)
        }
        for model_name in model_names
    }
    for row in model_rows:
        values_by_model_metric[str(row["model"])][str(row["metric"])].append(float(row["value"]))

    return model_rows, values_by_model_metric


def summarize_model_metrics(
    *,
    subset_name: str,
    y_true: np.ndarray,
    predictions_by_model: Mapping[str, np.ndarray],
    method_specs_by_model: Mapping[str, MethodSpec],
    run_outputs_by_model: Mapping[str, Sequence[Mapping[str, object]]],
    full_scores_by_model: Mapping[str, Mapping[str, np.ndarray]],
    values_by_model_metric: Mapping[str, Mapping[str, Sequence[float]]],
    metric_keys: Sequence[str],
    n_bootstrap: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model_name in predictions_by_model:
        spec = method_specs_by_model[model_name]
        run_outputs = run_outputs_by_model[model_name]
        for metric_key in metric_keys_for_model(model_name, metric_keys):
            full_scores = np.asarray(full_scores_by_model[model_name][metric_key], dtype=np.float64)
            bootstrap_values = values_by_model_metric[model_name][metric_key]
            ci_lower, ci_upper = quantile_ci(bootstrap_values)
            rows.append(
                {
                    "subset": subset_name,
                    "n_samples": len(y_true),
                    "positive_count": int(y_true.sum()),
                    "negative_count": int(len(y_true) - y_true.sum()),
                    "model": model_name,
                    "model_display": model_display_name(model_name),
                    "metric": metric_key,
                    "metric_display": METRIC_DISPLAY_NAMES[metric_key],
                    "run_prefix": spec.run_prefix,
                    "embedding_dir_name": spec.embedding_dir_name,
                    "n_runs": int(predictions_by_model[model_name].shape[0]),
                    "run_ids": format_run_ids(run_outputs),
                    "mean": mean_finite(full_scores),
                    "ci95_lower": ci_lower,
                    "ci95_upper": ci_upper,
                    "training_sd": finite_float(std_finite(full_scores)),
                    "n_bootstrap": n_bootstrap,
                }
            )
    return rows


def model_display_name(model_name: str) -> str:
    return TABLE_MODEL_ORDER.get(model_name, model_name)


def ordered_table_models(model_names: Sequence[str]) -> List[Tuple[str, str]]:
    remaining = [model_name for model_name in model_names if model_name not in TABLE_MODEL_ORDER]
    ordered = [
        (model_name, display_name)
        for model_name, display_name in TABLE_MODEL_ORDER.items()
        if model_name in model_names
    ]
    ordered.extend((model_name, model_name) for model_name in remaining)
    return ordered


def format_mean_ci_cell(summary_row: Optional[Mapping[str, object]]) -> str:
    if summary_row is None:
        return "N/A"
    mean = float(summary_row["mean"])
    lower = float(summary_row["ci95_lower"])
    upper = float(summary_row["ci95_upper"])
    if any(math.isnan(value) for value in (mean, lower, upper)):
        return "N/A"
    return f"{mean:.3f} [{lower:.3f}, {upper:.3f}]"


def format_mean_cell(summary_row: Optional[Mapping[str, object]]) -> str:
    if summary_row is None:
        return "N/A"
    mean = float(summary_row["mean"])
    if math.isnan(mean):
        return "N/A"
    return f"{mean:.3f}"


def build_metric_table_rows(
    *,
    model_summary_rows: Sequence[Mapping[str, object]],
    model_names: Sequence[str],
    metric_keys: Sequence[str],
    include_ci: bool,
) -> Tuple[List[str], List[List[str]]]:
    lookup = {
        (str(row["model"]), str(row["metric"])): row
        for row in model_summary_rows
    }
    ordered_models = ordered_table_models(model_names)
    header = ["Metric", *[display_name for _model_name, display_name in ordered_models]]
    rows: List[List[str]] = []
    for metric_key in metric_keys:
        row = [METRIC_DISPLAY_NAMES[metric_key]]
        for model_name, _display_name in ordered_models:
            summary_row = lookup.get((model_name, metric_key))
            row.append(
                format_mean_ci_cell(summary_row)
                if include_ci
                else format_mean_cell(summary_row)
            )
        rows.append(row)
    return header, rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_table_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_run_manifest(
    output_dir: Path,
    method_specs_by_model: Mapping[str, MethodSpec],
    run_outputs_by_model: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    rows: List[Dict[str, object]] = []
    for model_name, run_outputs in run_outputs_by_model.items():
        spec = method_specs_by_model[model_name]
        for run_output in run_outputs:
            rows.append(
                {
                    "model": model_name,
                    "model_display": model_display_name(model_name),
                    "run_prefix": spec.run_prefix,
                    "embedding_dir_name": spec.embedding_dir_name,
                    "run_id": run_output["run_id"],
                    "run_dir": str(run_output["run_dir"]),
                    "output_dir": str(run_output["output_dir"]),
                    "prediction_file": str(run_output["prediction_file"]),
                    "score_file": str(run_output["score_file"] or ""),
                    "edit_distance_file": str(run_output["edit_distance_file"] or ""),
                }
            )
    write_csv(
        output_dir / "run_manifest.csv",
        rows,
        [
            "model",
            "model_display",
            "run_prefix",
            "embedding_dir_name",
            "run_id",
            "run_dir",
            "output_dir",
            "prediction_file",
            "score_file",
            "edit_distance_file",
        ],
    )


def write_subset_outputs(
    *,
    subset_output_dir: Path,
    model_names: Sequence[str],
    metric_keys: Sequence[str],
    model_summary_rows: Sequence[Mapping[str, object]],
    bootstrap_model_rows: Sequence[Mapping[str, object]],
) -> None:
    bootstrap_output_dir = subset_output_dir / "bootstrap"
    write_csv(
        subset_output_dir / "metric_summary.csv",
        model_summary_rows,
        [
            "subset",
            "n_samples",
            "positive_count",
            "negative_count",
            "model",
            "model_display",
            "metric",
            "metric_display",
            "run_prefix",
            "embedding_dir_name",
            "n_runs",
            "run_ids",
            "mean",
            "ci95_lower",
            "ci95_upper",
            "training_sd",
            "n_bootstrap",
        ],
    )
    write_csv(
        bootstrap_output_dir / "bootstrap_model_metric_values.csv",
        bootstrap_model_rows,
        ["bootstrap_id", "model", "metric", "metric_display", "value"],
    )

    table_header, table_rows = build_metric_table_rows(
        model_summary_rows=model_summary_rows,
        model_names=model_names,
        metric_keys=metric_keys,
        include_ci=True,
    )
    write_table_csv(
        subset_output_dir / "bootstrap_table_mean_ci.csv",
        table_header,
        table_rows,
    )
    mean_table_header, mean_table_rows = build_metric_table_rows(
        model_summary_rows=model_summary_rows,
        model_names=model_names,
        metric_keys=metric_keys,
        include_ci=False,
    )
    write_table_csv(
        subset_output_dir / "bootstrap_table_mean.csv",
        mean_table_header,
        mean_table_rows,
    )

    f1_model_rows = [
        {
            "subset": row["subset"],
            "n_samples": row["n_samples"],
            "positive_count": row["positive_count"],
            "negative_count": row["negative_count"],
            "model": row["model"],
            "model_display": row["model_display"],
            "run_prefix": row["run_prefix"],
            "embedding_dir_name": row["embedding_dir_name"],
            "n_runs": row["n_runs"],
            "run_ids": row["run_ids"],
            "mean_f1": row["mean"],
            "ci_lower": row["ci95_lower"],
            "ci_upper": row["ci95_upper"],
            "training_sd": row["training_sd"],
            "n_bootstrap": row["n_bootstrap"],
        }
        for row in model_summary_rows
        if row["metric"] == "f1"
    ]
    if f1_model_rows:
        write_csv(
            subset_output_dir / "model_f1_ci.csv",
            f1_model_rows,
            [
                "subset",
                "n_samples",
                "positive_count",
                "negative_count",
                "model",
                "model_display",
                "run_prefix",
                "embedding_dir_name",
                "n_runs",
                "run_ids",
                "mean_f1",
                "ci_lower",
                "ci_upper",
                "training_sd",
                "n_bootstrap",
            ],
        )

    f1_bootstrap_model_rows = [
        {"bootstrap_id": row["bootstrap_id"], "model": row["model"], "f1": row["value"]}
        for row in bootstrap_model_rows
        if row["metric"] == "f1"
    ]
    if f1_bootstrap_model_rows:
        write_csv(
            bootstrap_output_dir / "bootstrap_model_f1_values.csv",
            f1_bootstrap_model_rows,
            ["bootstrap_id", "model", "f1"],
        )


def run_subset_analysis(
    *,
    subset_name: str,
    selected_indices: Sequence[int],
    y_true_full: np.ndarray,
    predictions_by_model_full: Mapping[str, np.ndarray],
    scores_by_model_full: Mapping[str, np.ndarray],
    brier_scores_by_model_full: Mapping[str, np.ndarray],
    method_specs_by_model: Mapping[str, MethodSpec],
    run_outputs_by_model: Mapping[str, Sequence[Mapping[str, object]]],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    selected_indices_array = np.asarray(selected_indices, dtype=np.int64)
    y_true = y_true_full[selected_indices_array]
    predictions_by_model = OrderedDict(
        (model_name, matrix[:, selected_indices_array])
        for model_name, matrix in predictions_by_model_full.items()
    )
    scores_by_model = OrderedDict(
        (model_name, matrix[:, selected_indices_array])
        for model_name, matrix in scores_by_model_full.items()
    )
    brier_scores_by_model = OrderedDict(
        (model_name, matrix[:, selected_indices_array])
        for model_name, matrix in brier_scores_by_model_full.items()
    )
    n_runs = next(iter(predictions_by_model.values())).shape[0]
    metric_keys = metric_keys_for_args(args)
    subset_output_dir = output_dir / subset_name
    num_workers = resolve_num_workers(args.num_workers, args.n_bootstrap)
    chunk_size = resolve_chunk_size(args.bootstrap_chunk_size, args.n_bootstrap, num_workers)

    full_scores_by_model = compute_full_metric_scores_by_model(
        y_true=y_true,
        predictions_by_model=predictions_by_model,
        scores_by_model=scores_by_model,
        brier_scores_by_model=brier_scores_by_model,
        metric_keys=metric_keys,
    )
    bootstrap_model_rows, values_by_model_metric = compute_bootstrap_outputs(
        subset_name=subset_name,
        y_true=y_true,
        predictions_by_model=predictions_by_model,
        scores_by_model=scores_by_model,
        brier_scores_by_model=brier_scores_by_model,
        metric_keys=metric_keys,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
        num_workers=num_workers,
        chunk_size=chunk_size,
        show_progress=not args.no_progress,
    )
    model_summary_rows = summarize_model_metrics(
        subset_name=subset_name,
        y_true=y_true,
        predictions_by_model=predictions_by_model,
        method_specs_by_model=method_specs_by_model,
        run_outputs_by_model=run_outputs_by_model,
        full_scores_by_model=full_scores_by_model,
        values_by_model_metric=values_by_model_metric,
        metric_keys=metric_keys,
        n_bootstrap=args.n_bootstrap,
    )
    write_subset_outputs(
        subset_output_dir=subset_output_dir,
        model_names=list(predictions_by_model.keys()),
        metric_keys=metric_keys,
        model_summary_rows=model_summary_rows,
        bootstrap_model_rows=bootstrap_model_rows,
    )
    print(
        f"Wrote {subset_name} results to {subset_output_dir} "
        f"(n={len(y_true)}, positive={int(y_true.sum())}, "
        f"negative={int(len(y_true) - y_true.sum())}, runs={n_runs}, "
        f"workers={num_workers}, chunk_size={chunk_size})"
    )


def main() -> int:
    args = parse_args()
    project_root = PROJECT_ROOT.resolve()
    input_path = resolve_path(project_root, args.input)
    results_root = resolve_path(project_root, args.results_root)
    output_dir = resolve_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_specs = parse_method_specs(args.method_spec)
    method_specs_by_model = OrderedDict((spec.model_name, spec) for spec in method_specs)
    rows = load_rxnorm_samples(input_path)
    y_true_full = np.asarray([parse_target(row["target"]) for row in rows], dtype=np.int8)
    (
        predictions_by_model_full,
        scores_by_model_full,
        brier_scores_by_model_full,
        run_outputs_by_model,
    ) = load_model_matrices_from_runs(
        results_root=results_root,
        method_specs=method_specs,
        expected_runs=args.num_runs,
        expected_samples=len(rows),
        args=args,
    )
    write_run_manifest(
        output_dir=output_dir,
        method_specs_by_model=method_specs_by_model,
        run_outputs_by_model=run_outputs_by_model,
    )

    for subset_name, allowed_type_norms in SUBSET_SPECS.items():
        selected_indices = filter_indices_by_type_norm(rows, allowed_type_norms)
        run_subset_analysis(
            subset_name=subset_name,
            selected_indices=selected_indices,
            y_true_full=y_true_full,
            predictions_by_model_full=predictions_by_model_full,
            scores_by_model_full=scores_by_model_full,
            brier_scores_by_model_full=brier_scores_by_model_full,
            method_specs_by_model=method_specs_by_model,
            run_outputs_by_model=run_outputs_by_model,
            output_dir=output_dir,
            args=args,
        )

    print(f"Wrote RxNorm two-way bootstrap results under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

