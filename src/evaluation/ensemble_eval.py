"""Ensemble evaluation: average sigmoid probabilities of two or more models,
then apply per-class / global thresholds.

Usage:
  python run_ensemble_eval.py <model1> <model2> <report_dir> <lang> [model3 ...]
"""

import json
import os
import sys

import numpy as np
import torch

from sklearn.metrics import classification_report

from transformers import AutoTokenizer

from src.config.config import Config
from src.data.dataset_loader import (
    create_label_vector,
    load_dataframe,
)
from src.models.custom_model import SensitiveCustomModel
from src.training.metrics import Metrics, LABELS

config = Config()

if len(sys.argv) < 4:
    print("Usage: python run_ensemble_eval.py <model1> <model2> <report_dir> <lang> [model3 ...]")
    sys.exit(1)

MODEL_PATHS = sys.argv[1:-2]
REPORT_DIR = sys.argv[-2]
TEST_LANG = sys.argv[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64


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


def main():

    print("=" * 60)
    print("SensitiveAI Ensemble Evaluation")
    print(f"Models  : {MODEL_PATHS}")
    print(f"Device  : {device}")
    print("=" * 60)

    test_df = load_dataframe(config.test_file)
    test_df = create_label_vector(test_df)
    test_df = test_df[test_df["language"] == TEST_LANG]

    labels = np.array(test_df["labels"].tolist())

    print(f"\nTest samples : {len(test_df)}")

    # ==========================
    # Inference per model
    # ==========================

    probs_list = []

    for path in MODEL_PATHS:

        print(f"\nLoading {path}...")

        model, tokenizer = load_model(path)

        print("Inference...")

        logits = run_inference(model, tokenizer, test_df)

        probs = Metrics.predict_probability(logits)

        probs_list.append(probs)

    # ==========================
    # Ensemble average
    # ==========================

    ensemble_probs = np.mean(probs_list, axis=0)

    # ==========================
    # Metrics @ 0.5
    # ==========================

    from src.training.metrics import Metrics as M
    from sklearn.metrics import (
        precision_recall_fscore_support,
        accuracy_score,
        roc_auc_score,
        average_precision_score,
    )

    predictions = (ensemble_probs >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="micro", zero_division=0
    )

    accuracy = accuracy_score(labels, predictions)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(labels, ensemble_probs, average="micro")), 4),
        "pr_auc": round(float(average_precision_score(labels, ensemble_probs, average="micro")), 4),
    }

    print("\n[Ensemble metrics @0.5]")
    print(json.dumps(metrics, indent=2))

    # ==========================
    # Classification report
    # ==========================

    report = classification_report(
        labels,
        predictions,
        target_names=LABELS,
        zero_division=0,
    )

    print("\n[Classification report]")
    print(report)

    # ==========================
    # Save
    # ==========================

    os.makedirs(REPORT_DIR, exist_ok=True)

    with open(os.path.join(REPORT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    with open(os.path.join(REPORT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nSaved -> {REPORT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()