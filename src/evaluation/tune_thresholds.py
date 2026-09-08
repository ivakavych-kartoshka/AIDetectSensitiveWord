"""Tune per-class / global decision thresholds on the validation set,
then evaluate the chosen thresholds on the test set.

Objectives:
  f1      : maximize per-label F1
  f2      : maximize per-label F2 (beta=2 -> recall counts 2x precision)
  recall  : maximize per-label recall with a minimum precision floor

Labels with no positive samples in the validation split are skipped
(their threshold stays at the DEFAULT_THRESHOLD) to avoid predicting
everything as positive.

Usage:
  python run_tune_thresholds.py <model_path> <report_dir> <lang> \
      [--objective f2] [--min-precision 0.5]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    f1_score,
    precision_score,
    recall_score,
)

from transformers import AutoTokenizer

from src.config.config import Config
from src.data.dataset_loader import (
    create_label_vector,
    load_dataframe,
)
from src.models.custom_model import SensitiveCustomModel
from src.training.metrics import Metrics, LABELS

config = Config()

MODEL_PATH = "models/sensitiveai-vi-custom-dlr2stage"
REPORT_DIR = "reports/vi-custom-dlr2stage"
TEST_LANG = "vi"
OBJECTIVE = "f2"
MIN_PRECISION = 0.5

if len(sys.argv) >= 4:
    MODEL_PATH = sys.argv[1]
    REPORT_DIR = sys.argv[2]
    TEST_LANG = sys.argv[3]

# --objective f1|recall|f2
for i, a in enumerate(sys.argv):
    if a == "--objective" and i + 1 < len(sys.argv):
        OBJECTIVE = sys.argv[i + 1]

# --min-precision <float>
for i, a in enumerate(sys.argv):
    if a == "--min-precision" and i + 1 < len(sys.argv):
        MIN_PRECISION = float(sys.argv[i + 1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64

THRESHOLD_GRID = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]


def f_beta(precision, recall, beta=1.0):
    """F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)."""

    if precision is None or recall is None:
        return 0.0

    beta2 = beta * beta

    denom = beta2 * precision + recall

    if denom <= 0:
        return 0.0

    return (1.0 + beta2) * precision * recall / denom


def run_inference(model, tokenizer, df):
    """Compute label probabilities for a dataframe."""

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

    all_logits = np.concatenate(all_logits, axis=0)

    return Metrics.predict_probability(all_logits)


def score_threshold(y_true_col, y_score_col, thr, objective, min_precision):
    """Score one (label, threshold) candidate on the validation column."""

    pred = (y_score_col >= thr).astype(int)

    p = precision_score(y_true_col, pred, zero_division=0)
    r = recall_score(y_true_col, pred, zero_division=0)

    if objective == "recall":

        if p >= min_precision:
            return r  # higher recall wins; tie-break by F1 below

        return -1 - f_beta(p, r, beta=2.0)  # penalized fallback

    if objective == "f2":

        return f_beta(p, r, beta=2.0)

    return f_beta(p, r, beta=1.0)  # f1


def find_best_thresholds(labels, probs, objective, min_precision=0.5):
    """Grid-search one threshold per label maximizing `objective`.

    Labels without any positive example in `labels` keep the default
    threshold so they do not collapse to all-positive predictions.
    """

    best = {}

    for i, label in enumerate(LABELS):

        col = labels[:, i]

        if col.sum() <= 0:

            best[label] = Metrics.DEFAULT_THRESHOLD
            continue

        best_score = -1.0
        best_thr = Metrics.DEFAULT_THRESHOLD
        best_f1 = -1.0

        for thr in THRESHOLD_GRID:

            score = score_threshold(col, probs[:, i], thr, objective, min_precision)

            if objective == "recall" and score < 0:

                continue  # precision floor not met

            # Tie-break: prefer the candidate with the higher F1.
            f1 = score_threshold(col, probs[:, i], thr, "f1", min_precision)

            if score > best_score or (score == best_score and f1 > best_f1):

                best_score = score
                best_thr = thr
                best_f1 = f1

        best[label] = best_thr

    return best


def evaluate_with_thresholds(labels, probs, thresholds):
    """Apply per-class thresholds and compute micro + per-class metrics."""

    thr_array = np.array(
        [thresholds.get(l, Metrics.DEFAULT_THRESHOLD) for l in LABELS]
    )

    predictions = (probs >= thr_array).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="micro",
        zero_division=0,
    )

    per_class_p, per_class_r, per_class_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average=None,
            zero_division=0,
        )
    )

    return {
        "micro_precision": round(float(precision), 4),
        "micro_recall": round(float(recall), 4),
        "micro_f1": round(float(f1), 4),
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
    print(f"SensitiveAI Threshold Tuning - {TEST_LANG}")
    print(f"Model    : {MODEL_PATH}")
    print(f"Device   : {device}")
    print(f"Objective: {OBJECTIVE} (min_precision={MIN_PRECISION})")
    print("=" * 60)

    # ==========================
    # Load model
    # ==========================

    print("\nLoading model...")

    model = SensitiveCustomModel.from_pretrained(MODEL_PATH)

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # ==========================
    # Load validation + test
    # ==========================

    val_df = load_dataframe(config.val_file)
    val_df = create_label_vector(val_df)
    val_df = val_df[val_df["language"] == TEST_LANG]

    test_df = load_dataframe(config.test_file)
    test_df = create_label_vector(test_df)
    test_df = test_df[test_df["language"] == TEST_LANG]

    print(f"Validation samples : {len(val_df)}")
    print(f"Test samples       : {len(test_df)}")

    # ==========================
    # Inference
    # ==========================

    print("\nInference on validation...")

    val_probs = run_inference(model, tokenizer, val_df)
    val_labels = np.array(val_df["labels"].tolist())

    print("Inference on test...")

    test_probs = run_inference(model, tokenizer, test_df)
    test_labels = np.array(test_df["labels"].tolist())

    # ==========================
    # Baselines
    # ==========================

    print("\n[Baseline: uniform threshold 0.5]")
    base_flat = evaluate_with_thresholds(
        test_labels,
        test_probs,
        {l: Metrics.DEFAULT_THRESHOLD for l in LABELS},
    )
    print(json.dumps(base_flat, indent=2))

    print("\n[Baseline: config per-class thresholds]")
    base_config = evaluate_with_thresholds(
        test_labels,
        test_probs,
        Config().per_class_thresholds,
    )
    print(json.dumps(base_config, indent=2))

    # ==========================
    # Tune thresholds per objective
    # ==========================

    results = {}

    for obj in ["f1", "f2", "recall"]:

        print(f"\n[Tuning objective: {obj}]")

        thr = find_best_thresholds(
            val_labels,
            val_probs,
            objective=obj,
            min_precision=MIN_PRECISION,
        )

        print(json.dumps(thr, indent=2))

        ev = evaluate_with_thresholds(test_labels, test_probs, thr)

        results[obj] = {
            "thresholds": thr,
            "eval": ev,
        }

        print(json.dumps(ev, indent=2))

    # ==========================
    # Active objective = chosen
    # ==========================

    chosen = results[OBJECTIVE]

    # ==========================
    # Save results
    # ==========================

    os.makedirs(REPORT_DIR, exist_ok=True)

    report = {
        "objective": OBJECTIVE,
        "min_precision": MIN_PRECISION,
        "baseline_flat_0_5": base_flat,
        "baseline_config_thresholds": base_config,
        "tuned": results,
        "recommended": {
            "thresholds": chosen["thresholds"],
            "eval": chosen["eval"],
        },
    }

    with open(
        os.path.join(REPORT_DIR, "threshold_tuning.json"),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(report, f, indent=4, ensure_ascii=False)

    print("\n[Comparison - micro on test]")
    print(
        f"{'config':<10}{'P':>8}{'R':>8}{'F1':>8}"
    )
    print(
        f"{'baseline 0.5':<10}"
        f"{base_flat['micro_precision']:>8.4f}"
        f"{base_flat['micro_recall']:>8.4f}"
        f"{base_flat['micro_f1']:>8.4f}"
    )
    print(
        f"{'config thr':<10}"
        f"{base_config['micro_precision']:>8.4f}"
        f"{base_config['micro_recall']:>8.4f}"
        f"{base_config['micro_f1']:>8.4f}"
    )

    for obj in results:

        ev = results[obj]["eval"]

        print(
            f"{'tuned ' + obj:<10}"
            f"{ev['micro_precision']:>8.4f}"
            f"{ev['micro_recall']:>8.4f}"
            f"{ev['micro_f1']:>8.4f}"
        )

    print(
        f"\nRecommended thresholds ({OBJECTIVE}): "
        f"{json.dumps(chosen['thresholds'], ensure_ascii=False)}"
    )

    print(f"\nSaved report -> {REPORT_DIR}/threshold_tuning.json")
    print("=" * 60)


if __name__ == "__main__":
    main()