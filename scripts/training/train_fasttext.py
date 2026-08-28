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

"""Train/evaluate the fastText + XGBoost baseline used in the paper."""
import argparse
import csv
import datetime
import json
import logging
import os
import pickle
import random
from pathlib import Path
from collections import Counter
from collections import namedtuple

import numpy as np
import sklearn.metrics as sklearn_metrics
from tqdm import tqdm

from drug_spelling.baselines import (
    CLASSIFICATION_THRESHOLD,
    manuscript_xgb_grid,
    select_best_f1_trial,
    validate_manuscript_xgb_config,
)

ClassificationExample = namedtuple("ClassificationExample", ["id", "text", "label"])
RELEASE_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    logging.info("Random seed: %d", seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification"],
        help="Only classification is supported for this baseline.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Whether to apply lowercasing during tokenization.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=20,
        help="Number of epochs for fastText training.",
    )
    parser.add_argument(
        "--fasttext_learning_rate",
        default=0.05,
        type=float,
        help="Learning rate for FastText unsupervised training.",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Train FastText + classifier, then evaluate on test.",
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Load trained FastText + classifier and evaluate on test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="rxnorm",
        help="Dataset name used only when --data_dir is omitted.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing train{k}.txt, dev.txt, and test.txt.",
    )
    parser.add_argument("--train_file", type=str, default=None, help="Defaults to train{setnum}.txt.")
    parser.add_argument("--dev_file", type=str, default="dev.txt")
    parser.add_argument("--test_file", type=str, default="test.txt")
    parser.add_argument(
        "--setnum",
        type=int,
        default=1,
        help="Training set number (used to resolve train{setnum}.txt).",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Optional path to an existing fasttext result dir for --do_predict.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Training output directory; also used as the model directory for prediction when --model_dir is omitted.",
    )
    parser.add_argument(
        "--prediction_output_dir",
        type=str,
        default=None,
        help="Optional separate destination for --do_predict outputs.",
    )
    parser.add_argument(
        "--xgb_config",
        type=str,
        default=None,
        help=(
            "Optional selected_xgboost_hyperparameters.json from the point-estimate "
            "fit. The configuration must belong to the manuscript grid and is useful "
            "for applying the same selected settings to stability runs."
        ),
    )
    # FastText-specific options
    parser.add_argument("--fasttext_model", type=str, choices=["skipgram", "cbow"], default="skipgram")
    parser.add_argument(
        "--fasttext_training_mode",
        type=str,
        choices=["unsupervised", "supervised"],
        default="supervised",
        help="How to train FastText vectors used for feature extraction.",
    )
    parser.add_argument("--fasttext_dim", type=int, default=200)
    parser.add_argument("--fasttext_ws", type=int, default=5)
    parser.add_argument("--fasttext_minn", type=int, default=2)
    parser.add_argument("--fasttext_maxn", type=int, default=5)
    parser.add_argument("--fasttext_min_count", type=int, default=1)
    parser.add_argument("--fasttext_word_ngrams", type=int, default=2)
    parser.add_argument("--fasttext_bucket", type=int, default=200000)
    parser.add_argument("--fasttext_thread", type=int, default=max(1, (os.cpu_count() or 1) // 2))

    args = parser.parse_args()
    if not args.do_train and not args.do_predict:
        parser.error("At least one of --do_train or --do_predict must be set.")
    args.start_time = datetime.datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss")
    requested_output_dir = args.output_dir or args.model_dir
    args.data_dir = args.data_dir or str(RELEASE_ROOT / "data" / "classification" / args.exp_name)
    args.output_dir = requested_output_dir or str(
        RELEASE_ROOT / "artifacts" / "models" / "fasttext" / f"train_set{args.setnum}"
    )
    return args


def setup_logging(args):
    if args.do_train:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.basicConfig(
            handlers=[
                logging.FileHandler(os.path.join(args.output_dir, "Training.log"), mode="w"),
                logging.StreamHandler(),
            ],
            format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
            level=logging.INFO,
            force=True,
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
            level=logging.INFO,
            force=True,
        )


def example_to_text(example):
    return example.text


def load_dataset_split(args, split):
    base_dir = Path(args.data_dir)
    filename = (
        (args.train_file or f"train{args.setnum}.txt")
        if split == "train"
        else (args.dev_file if split == "dev" else args.test_file)
    )
    requested = Path(filename)
    path = requested if requested.is_absolute() else base_dir / requested
    if split == "train" and not path.exists():
        path = base_dir / "train.txt"

    examples = []
    with path.open("r", encoding="utf-8") as data_file:
        lines = data_file.readlines()
        for i, line in enumerate(lines):
            splitline = line.strip().split()
            if not splitline:
                continue
            label = splitline[0].split("__label__")[-1]
            text = " ".join(splitline[1:])
            if args.do_lower_case:
                text = text.lower()
            examples.append(ClassificationExample(id=i, text=text, label=label))
    logging.info("Number of `%s` examples: %d", split, len(examples))
    return examples


def write_fasttext_corpus(examples, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            text = example_to_text(example)
            if text:
                f.write(text + "\n")


def write_fasttext_supervised_corpus(examples, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            text = example_to_text(example)
            if not text:
                continue
            f.write(f"__label__{example.label} {text}\n")


def train_fasttext_model(args, train_examples):
    try:
        import fasttext  # pylint: disable=import-error
    except ImportError as exc:
        raise ImportError(
            "fasttext is not installed. Install it with `pip install fasttext`."
        ) from exc

    if args.fasttext_training_mode == "unsupervised":
        corpus_path = os.path.join(args.output_dir, "fasttext_train_corpus.txt")
        write_fasttext_corpus(train_examples, corpus_path)
        logging.info("Training FastText (unsupervised) model on: %s", corpus_path)
        model = fasttext.train_unsupervised(
            input=corpus_path,
            model=args.fasttext_model,
            dim=args.fasttext_dim,
            ws=args.fasttext_ws,
            minn=args.fasttext_minn,
            maxn=args.fasttext_maxn,
            minCount=args.fasttext_min_count,
            epoch=args.num_train_epochs,
            lr=args.fasttext_learning_rate,
            thread=args.fasttext_thread,
            bucket=args.fasttext_bucket,
        )
    else:
        corpus_path = os.path.join(args.output_dir, "fasttext_train_supervised.txt")
        write_fasttext_supervised_corpus(train_examples, corpus_path)
        logging.info("Training FastText (supervised) model on: %s", corpus_path)
        model = fasttext.train_supervised(
            input=corpus_path,
            dim=args.fasttext_dim,
            ws=args.fasttext_ws,
            minn=args.fasttext_minn,
            maxn=args.fasttext_maxn,
            minCount=args.fasttext_min_count,
            epoch=args.num_train_epochs,
            lr=args.fasttext_learning_rate,
            thread=args.fasttext_thread,
            wordNgrams=args.fasttext_word_ngrams,
            bucket=args.fasttext_bucket,
        )
    model_path = os.path.join(args.output_dir, "fasttext.bin")
    model.save_model(model_path)
    logging.info("Saved FastText model to %s", model_path)
    return model


def load_fasttext_model(model_dir):
    try:
        import fasttext  # pylint: disable=import-error
    except ImportError as exc:
        raise ImportError(
            "fasttext is not installed. Install it with `pip install fasttext`."
        ) from exc
    return fasttext.load_model(os.path.join(model_dir, "fasttext.bin"))


def build_feature_matrix(examples, fasttext_model):
    dim = fasttext_model.get_dimension()
    x = np.zeros((len(examples), dim), dtype=np.float32)
    for i, example in tqdm(enumerate(examples), total=len(examples), desc="FastText features"):
        text = example_to_text(example)
        if text:
            x[i] = fasttext_model.get_sentence_vector(text)
    return x


def examples_to_targets(examples, label_to_id):
    return np.array([label_to_id[example.label] for example in examples], dtype=np.int64)


def train_xgb_classifier(args, x_train, y_train, config):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is not installed. Install the training requirements first."
        ) from exc

    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=config["learning_rate"],
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        subsample=config["subsample"],
        random_state=args.seed,
        n_jobs=-1,
    )
    classifier.fit(x_train, y_train)
    return classifier


def run_manuscript_grid_search(
    args,
    x_train,
    y_train,
    x_dev,
    y_dev,
    requested_config=None,
):
    trials = []
    best_classifier = None
    best_trial = None
    grid = [requested_config] if requested_config is not None else manuscript_xgb_grid()
    for trial_number, config in enumerate(grid, start=1):
        classifier = train_xgb_classifier(args, x_train, y_train, config)
        dev_scores, _ = get_positive_scores(classifier, x_dev)
        dev_predictions = labels_from_scores(dev_scores, CLASSIFICATION_THRESHOLD)
        dev_f1 = float(
            sklearn_metrics.f1_score(
                y_dev,
                dev_predictions,
                average="binary",
                zero_division=0,
            )
        )
        trial = {
            "trial": trial_number,
            **config,
            "threshold": CLASSIFICATION_THRESHOLD,
            "dev_f1": dev_f1,
        }
        trials.append(trial)
        if best_trial is None or dev_f1 > float(best_trial["dev_f1"]):
            best_trial = trial
            best_classifier = classifier
        logging.info(
            "XGBoost grid %d/%d: %s; dev F1=%.6f",
            trial_number,
            len(grid),
            config,
            dev_f1,
        )

    selected_trial = dict(select_best_f1_trial(trials))
    if best_classifier is None or best_trial is None:
        raise RuntimeError("The XGBoost grid search did not produce a model.")
    if selected_trial["trial"] != best_trial["trial"]:
        raise RuntimeError("Internal error while selecting the best development F1.")
    selected_config = {
        key: selected_trial[key]
        for key in ("learning_rate", "n_estimators", "max_depth", "subsample")
    }
    logging.info("Selected XGBoost configuration: %s", selected_config)
    return best_classifier, selected_config, trials


def get_positive_scores(classifier, x):
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(x)
        return probs[:, 1], probs

    decision = classifier.decision_function(x)
    scores = 1.0 / (1.0 + np.exp(-decision))
    probs = np.stack([1.0 - scores, scores], axis=1)
    return scores, probs


def labels_from_scores(y_scores, threshold):
    return (y_scores >= threshold).astype(np.int64)


def compute_metrics(y_true, y_pred, y_scores, y_probs):
    results = {
        "precision": sklearn_metrics.precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": sklearn_metrics.recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": sklearn_metrics.f1_score(y_true, y_pred, average="binary", zero_division=0),
        "accuracy": sklearn_metrics.accuracy_score(y_true, y_pred),
        "specificity": sklearn_metrics.recall_score(
            y_true, y_pred, pos_label=0, average="binary", zero_division=0
        ),
    }

    try:
        results["ROC AUC"] = sklearn_metrics.roc_auc_score(y_true, y_scores)
    except ValueError:
        results["ROC AUC"] = float("nan")

    try:
        results["PR AUC"] = sklearn_metrics.average_precision_score(y_true, y_scores)
    except ValueError:
        results["PR AUC"] = float("nan")

    try:
        results["loss"] = sklearn_metrics.log_loss(y_true, y_probs, labels=[0, 1])
    except ValueError:
        results["loss"] = float("nan")

    return {k: float(v) for k, v in results.items()}


def evaluate_split(
    classifier,
    examples,
    label_to_id,
    fasttext_model,
    split_name,
    threshold,
    feature_matrix=None,
):
    x = feature_matrix if feature_matrix is not None else build_feature_matrix(examples, fasttext_model)
    y_true = examples_to_targets(examples, label_to_id)
    y_scores, y_probs = get_positive_scores(classifier, x)
    y_pred = labels_from_scores(y_scores, threshold)
    metrics = compute_metrics(y_true, y_pred, y_scores, y_probs)
    metrics["threshold"] = float(threshold)

    logging.info("***** %s results *****", split_name)
    for key in sorted(metrics.keys()):
        logging.info("  %s = %s", key, metrics[key])
    return metrics, y_pred, y_probs, y_true


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_metrics_file(path, title, metrics):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"--- {title} ---\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


def save_prediction_csv(path, examples, targets, predictions, probabilities):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "term", "target", "prediction", "probability"])
        writer.writeheader()
        for index, (example, target, prediction, probability) in enumerate(
            zip(examples, targets, predictions, probabilities[:, 1])
        ):
            writer.writerow(
                {
                    "index": index,
                    "term": example.text,
                    "target": int(target),
                    "prediction": int(prediction),
                    "probability": float(probability),
                }
            )


def save_grid_search_csv(path, trials):
    fieldnames = [
        "trial",
        "learning_rate",
        "n_estimators",
        "max_depth",
        "subsample",
        "threshold",
        "dev_f1",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trials)


def load_requested_xgb_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("The XGBoost configuration file must contain a JSON object.")
    config = payload.get("configuration", payload)
    if not isinstance(config, dict):
        raise ValueError("The XGBoost 'configuration' field must be a JSON object.")
    return validate_manuscript_xgb_config(config)


def run_training(args):
    os.makedirs(args.output_dir, exist_ok=True)
    data = {
        "train": load_dataset_split(args, "train"),
        "dev": load_dataset_split(args, "dev"),
        "test": load_dataset_split(args, "test"),
    }

    counter = Counter(example.label for split in data.values() for example in split)
    labels = sorted(counter.keys())
    if len(labels) != 2:
        raise ValueError(
            f"This baseline currently expects binary labels, found labels={labels}."
        )
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    logging.info("Label distribution: %s", dict(counter))
    logging.info("Label mapping: %s", label_to_id)
    fasttext_model = train_fasttext_model(args, data["train"])
    x_train = build_feature_matrix(data["train"], fasttext_model)
    y_train = examples_to_targets(data["train"], label_to_id)
    x_dev = build_feature_matrix(data["dev"], fasttext_model)
    y_dev = examples_to_targets(data["dev"], label_to_id)
    requested_config = (
        load_requested_xgb_config(args.xgb_config)
        if args.xgb_config
        else None
    )
    classifier, selected_config, grid_trials = run_manuscript_grid_search(
        args,
        x_train,
        y_train,
        x_dev,
        y_dev,
        requested_config=requested_config,
    )

    with open(os.path.join(args.output_dir, "classifier.pkl"), "wb") as f:
        pickle.dump(classifier, f)
    save_json(
        os.path.join(args.output_dir, "label_to_id.json"),
        {"label_to_id": label_to_id, "id_to_label": id_to_label},
    )
    args.selected_xgb_hyperparameters = selected_config
    save_json(os.path.join(args.output_dir, "training_args.json"), vars(args))
    save_grid_search_csv(
        os.path.join(args.output_dir, "xgboost_grid_search.csv"),
        grid_trials,
    )
    save_json(
        os.path.join(args.output_dir, "selected_xgboost_hyperparameters.json"),
        {
            "selection_metric": "development_f1",
            "selection_source": (
                "provided_point_estimate_configuration"
                if requested_config is not None
                else "manuscript_grid_search"
            ),
            "threshold": CLASSIFICATION_THRESHOLD,
            "configuration": selected_config,
        },
    )
    save_json(
        os.path.join(args.output_dir, "decision_threshold.json"),
        {"threshold": CLASSIFICATION_THRESHOLD, "source": "manuscript"},
    )

    dev_metrics, _, _, _ = evaluate_split(
        classifier=classifier,
        examples=data["dev"],
        label_to_id=label_to_id,
        fasttext_model=fasttext_model,
        split_name="Dev",
        threshold=CLASSIFICATION_THRESHOLD,
        feature_matrix=x_dev,
    )
    test_metrics, test_pred_ids, test_probs, test_targets = evaluate_split(
        classifier=classifier,
        examples=data["test"],
        label_to_id=label_to_id,
        fasttext_model=fasttext_model,
        split_name="Test",
        threshold=CLASSIFICATION_THRESHOLD,
    )

    save_metrics_file(
        os.path.join(args.output_dir, "performance_on_dev_set.txt"),
        "Performance on dev set",
        dev_metrics,
    )
    save_metrics_file(
        os.path.join(args.output_dir, "performance_on_test_set.txt"),
        "Performance on test set",
        test_metrics,
    )

    with open(os.path.join(args.output_dir, "predict.txt"), "w", encoding="utf-8") as f:
        for pred in test_pred_ids:
            f.write(f"{id_to_label[int(pred)]}\n")
    np.savetxt(os.path.join(args.output_dir, "test_predict.txt"), test_pred_ids, fmt="%d")
    np.savetxt(os.path.join(args.output_dir, "test_logits.txt"), test_probs, fmt="%.8f")
    np.savetxt(os.path.join(args.output_dir, "test_targets.txt"), test_targets, fmt="%d")
    save_prediction_csv(
        os.path.join(args.output_dir, "test_predictions.csv"),
        data["test"],
        test_targets,
        test_pred_ids,
        test_probs,
    )

    for k, v in test_metrics.items():
        print(k + "\t" + str(v))


def run_predict(args):
    model_dir = args.output_dir
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    test_examples = load_dataset_split(args, "test")
    fasttext_model = load_fasttext_model(model_dir)
    with open(os.path.join(model_dir, "classifier.pkl"), "rb") as f:
        classifier = pickle.load(f)
    with open(os.path.join(model_dir, "label_to_id.json"), "r", encoding="utf-8") as f:
        label_map_payload = json.load(f)
    label_to_id = {k: int(v) for k, v in label_map_payload["label_to_id"].items()}
    id_to_label = {int(k): v for k, v in label_map_payload["id_to_label"].items()}

    test_metrics, test_pred_ids, test_probs, test_targets = evaluate_split(
        classifier=classifier,
        examples=test_examples,
        label_to_id=label_to_id,
        fasttext_model=fasttext_model,
        split_name="Test",
        threshold=CLASSIFICATION_THRESHOLD,
    )

    prediction_output_dir = args.prediction_output_dir or model_dir
    os.makedirs(prediction_output_dir, exist_ok=True)
    save_metrics_file(
        os.path.join(prediction_output_dir, "performance_on_test_set_predict.txt"),
        "Performance on test set",
        test_metrics,
    )
    with open(os.path.join(prediction_output_dir, "test_predict_label.txt"), "w", encoding="utf-8") as f:
        for pred in test_pred_ids:
            f.write(f"{id_to_label[int(pred)]}\n")
    np.savetxt(os.path.join(prediction_output_dir, "test_predict.txt"), test_pred_ids, fmt="%d")
    np.savetxt(os.path.join(prediction_output_dir, "test_logits.txt"), test_probs, fmt="%.8f")
    np.savetxt(os.path.join(prediction_output_dir, "test_targets.txt"), test_targets, fmt="%d")
    save_prediction_csv(
        os.path.join(prediction_output_dir, "test_predictions.csv"),
        test_examples,
        test_targets,
        test_pred_ids,
        test_probs,
    )

    for k, v in test_metrics.items():
        print(k + "\t" + str(v))


def main():
    args = parse_args()
    setup_logging(args)
    set_seed(args.seed)
    logging.info("Using the following arguments:")
    for key, value in vars(args).items():
        logging.info("* %s: %s", key, value)

    if args.do_train:
        run_training(args)

    if args.do_predict:
        run_predict(args)


if __name__ == "__main__":
    main()
