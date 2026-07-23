import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
)

from transformers import AutoModelForSequenceClassification

from src.config.config import Config
from src.data.dataset_loader import load_dataset
from src.training.metrics import Metrics, LABELS

# =====================================================
# CONFIG
# =====================================================

config = Config()

REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 60)
print("SensitiveAI Evaluation")
print("=" * 60)

print("\nLoading test dataset...")

_, _, test_dataset = load_dataset()

print(f"Test Samples : {len(test_dataset)}")

# =====================================================
# LOAD MODEL
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nDevice : {device}")

model = AutoModelForSequenceClassification.from_pretrained(config.output_dir)

model.to(device)

model.eval()

# =====================================================
# PREDICT
# =====================================================

all_logits = []
all_labels = []

print("\nRunning inference...")

for sample in test_dataset:

    inputs = {
        "input_ids": sample["input_ids"].unsqueeze(0).to(device),
        "attention_mask": sample["attention_mask"].unsqueeze(0).to(device),
    }

    with torch.no_grad():

        outputs = model(**inputs)

    logits = outputs.logits.squeeze().cpu().numpy()

    all_logits.append(logits)

    all_labels.append(sample["labels"].cpu().numpy())

all_logits = np.array(all_logits)

all_labels = np.array(all_labels)

# =====================================================
# METRICS
# =====================================================

print("\nCalculating metrics...")

metrics = Metrics.calculate(
    (
        all_logits,
        all_labels,
    )
)

print(json.dumps(metrics, indent=4))

Metrics.save_metrics(
    metrics,
    os.path.join(
        REPORT_DIR,
        "metrics.json",
    ),
)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

predictions = Metrics.predict_binary(all_logits)

report = classification_report(
    all_labels,
    predictions,
    target_names=LABELS,
    zero_division=0,
)

print("\nClassification Report\n")

print(report)

with open(
    os.path.join(
        REPORT_DIR,
        "classification_report.txt",
    ),
    "w",
    encoding="utf-8",
) as f:

    f.write(report)

# =====================================================
# SAVE PREDICTIONS
# =====================================================

print("\nSaving prediction examples...")

probabilities = Metrics.predict_probability(all_logits)

rows = []

for i in range(len(test_dataset)):

    row = {}

    for j, label in enumerate(LABELS):

        row[f"{label}_true"] = int(all_labels[i][j])

        row[f"{label}_pred"] = int(predictions[i][j])

        row[f"{label}_score"] = round(
            float(probabilities[i][j]),
            4,
        )

    rows.append(row)

pd.DataFrame(rows).to_csv(
    os.path.join(
        REPORT_DIR,
        "prediction_examples.csv",
    ),
    index=False,
)

# =====================================================
# CONFUSION MATRIX
# =====================================================

print("\nGenerating Confusion Matrix...")

for i, label in enumerate(LABELS):

    cm = confusion_matrix(
        all_labels[:, i],
        predictions[:, i],
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
    )

    disp.plot()

    plt.title(label)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            REPORT_DIR,
            f"confusion_matrix_{label}.png",
        )
    )

    plt.close()

# =====================================================
# ROC CURVE
# =====================================================

print("Generating ROC Curves...")

for i, label in enumerate(LABELS):

    fpr, tpr, _ = roc_curve(
        all_labels[:, i],
        probabilities[:, i],
    )

    plt.figure()

    plt.plot(fpr, tpr)

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title(label)

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            REPORT_DIR,
            f"roc_curve_{label}.png",
        )
    )

    plt.close()

# =====================================================
# PR CURVE
# =====================================================

print("Generating Precision-Recall Curves...")

for i, label in enumerate(LABELS):

    precision, recall, _ = precision_recall_curve(
        all_labels[:, i],
        probabilities[:, i],
    )

    plt.figure()

    plt.plot(
        recall,
        precision,
    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title(label)

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            REPORT_DIR,
            f"pr_curve_{label}.png",
        )
    )

    plt.close()

# =====================================================
# FINISH
# =====================================================

print("\nEvaluation Finished")

print(f"\nReports saved to : {REPORT_DIR}")
