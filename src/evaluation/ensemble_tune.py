"""Ensemble + threshold tuning: average sigmoid probabilities of two or
more models, tune per-class thresholds on the validation set, then evaluate
on the test set.

Usage:
  python run_ensemble_tune.py <model1> <model2> <report_dir> <lang> [model3 ...]
"""

import json
import os
import sys

import numpy as np
import torch

from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)

from transformers import AutoTokenizer

from src.config.config import Config
from src.data.dataset_loader import (
    create_label_vector,
    load_dataframe,
)
from src.models.custom_model import SensitiveCustomModel
from src.training.metrics import Metrics, LABELS

if len(sys.argv) < 4:
    print("Usage: python run_ensemble_tune.py <model1> <model2> <report_dir> <lang> [model3 ...]")
    sys.exit(1)

MODEL_PATHS = sys.argv[1:-2]
REPORT_DIR = sys.argv[-2]
TEST_LANG = sys.argv[-1]

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64

THRESHOLD_GRID = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]


def load_model(path):
    model = SensitiveCustomModel.from_pretrained(path)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


def run_inference(model, tokenizer, df):

    batch = tokenizer(
        df["normalized_text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=config.max_length,
        return_tensors="pt",
    )

    all_logits = []

    for i in range(0, len(df), BATCH_SIZE):

        inputs_ids = batch["input_ids"][i : i + BATCH_SIZE].to(device)
        attention = batch["attention_mask"][i : i + BATCH_SIZE].to(device)

        with torch.no_grad():

            outputs = model(
                input_ids=inputs_ids,
                attention_mask=attention,
            )

        all_logits.append(outputs["logits"].cpu().numpy())

    return np.concatenate(all_logits, axis=0)


def ensemble_probs(model_paths, tokenizer_df):
    """Return averaged sigmoid probabilities across models."""

    probs_list = []

    for path in model_paths:

        print(f"\nLoading {path}...")

        model, tokenizer = load_model(path)

        print("Inference...")

        logits = run_inference(model, tokenizer, tokenizer_df)

        probs_list.append(Metrics.predict_probability(logits))

    return np.mean(probs_list, axis=0)


def find_best_thresholds(labels, probs):

    best = {}

    for i, label in enumerate(LABELS):

        best_f1 = -1.0
        best_thr = 0.5

        for thr in THRESHOLD_GRID:

            preds = (probs[:, i] >= thr).astype(int)

            f1 = f1_score(labels[:, i], preds, zero_division=0)

            if f1 > best_f1:

                best_f1 = f1
                best_thr = thr

        best[label] = best_thr

    return best


def evaluate_with_thresholds(labels, probs, thresholds):

    thr_array = np.array(
        [thresholds.get(l, 0.5) for l in LABELS]
    )

    predictions = (probs >= thr_array).astype(int)

    acc = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="micro", zero_division=0
    )

    per_class_p, per_class_r, per_class_f1, _ = (
        precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
    )

    return {
        "accuracy": round(float(acc), 4),
        "micro_precision": round(float(precision), 4),
        "micro_recall": round(float(recall), 4),
        "micro_f1": round(float(f1), 4),
        "roc_auc": round(
            float(roc_auc_score(labels, probs, average="micro")), 4
        ),
        "pr_auc": round(
            float(average_precision_score(labels, probs, average="micro")), 4
        ),
        "per_class": {
            label: {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
            }
            for label, p, r, f in zip(
                LABELS, per_class_p, per_class_r, per_class_f1
            )
        },
    }


def main():

    print("=" * 60)
    print("SensitiveAI Ensemble + Threshold Tuning")
    print(f"Models  : {MODEL_PATHS}")
    print(f"Device  : {device}")
    print("=" * 60)

    val_df = load_dataframe(config.val_file)
    val_df = create_label_vector(val_df)
    val_df = val_df[val_df["language"] == TEST_LANG]

    test_df = load_dataframe(config.test_file)
    test_df = create_label_vector(test_df)
    test_df = test_df[test_df["language"] == TEST_LANG]

    print(f"\nValidation samples : {len(val_df)}")
    print(f"Test samples       : {len(test_df)}")

    print("\nEnsemble inference on validation...")
    val_probs = ensemble_probs(MODEL_PATHS, val_df)
    val_labels = np.array(val_df["labels"].tolist())

    print("\nEnsemble inference on test...")
    test_probs = ensemble_probs(MODEL_PATHS, test_df)
    test_labels = np.array(test_df["labels"].tolist())

    print("\n[Baseline threshold 0.5]")
    base = evaluate_with_thresholds(
        test_labels, test_probs, {l: 0.5 for l in LABELS}
    )
    print(json.dumps(base, indent=2))

    best_thresholds = find_best_thresholds(val_labels, val_probs)

    print("\n[Best per-class thresholds by F1]")
    print(json.dumps(best_thresholds, indent=2))

    print("\n[Evaluation with tuned thresholds]")
    tuned = evaluate_with_thresholds(
        test_labels, test_probs, best_thresholds
    )
    print(json.dumps(tuned, indent=2))

    predictions = (
        test_probs
        >= np.array([best_thresholds.get(l, 0.5) for l in LABELS])
    ).astype(int)

    report = classification_report(
        test_labels,
        predictions,
        target_names=LABELS,
        zero_division=0,
    )

    os.makedirs(REPORT_DIR, exist_ok=True)

    out = {
        "models": MODEL_PATHS,
        "baseline_0_5": base,
        "best_thresholds_f1": best_thresholds,
        "tuned_f1": tuned,
    }

    with open(
        os.path.join(REPORT_DIR, "ensemble_tuning.json"),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(out, f, indent=4, ensure_ascii=False)

    with open(
        os.path.join(REPORT_DIR, "classification_report_tuned.txt"),
        "w",
        encoding="utf-8",
    ) as f:

        f.write(report)

    print(f"\nSaved report -> {REPORT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()