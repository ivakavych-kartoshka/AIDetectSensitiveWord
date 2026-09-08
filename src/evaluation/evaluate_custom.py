import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
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

from transformers import AutoTokenizer

from src.config.config import Config
from src.data.dataset_loader import (
    create_label_vector,
    load_dataframe,
)
from src.models.custom_model import SensitiveCustomModel
from src.training.metrics import Metrics, LABELS

# =====================================================
# CONFIG
# =====================================================

config = Config()

MODEL_PATH = "models/sensitiveai-en-custom"
REPORT_DIR = "reports/en-custom"
TEST_LANG = "en"

# Override via: python run_evaluate_custom.py <model_path> <report_dir> <test_lang>
if len(sys.argv) >= 4:

    MODEL_PATH = sys.argv[1]
    REPORT_DIR = sys.argv[2]
    TEST_LANG = sys.argv[3]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64

# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 60)
print("SensitiveAI Evaluation - Custom Head (en)")
print("=" * 60)

test_df = load_dataframe(config.test_file)
test_df = create_label_vector(test_df)

df = test_df[test_df["language"] == TEST_LANG]

print(f"Samples : {len(df)}")
print(f"Device  : {device}")

# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading model...")

model = SensitiveCustomModel.from_pretrained(MODEL_PATH)

model.to(device)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# =====================================================
# INFERENCE
# =====================================================

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

labels = np.array(df["labels"].tolist())

probabilities = Metrics.predict_probability(all_logits)

predictions = Metrics.predict_binary(
    all_logits,
    per_class_thresholds=config.per_class_thresholds,
)

# =====================================================
# METRICS
# =====================================================

metrics = Metrics.calculate_with_thresholds(
    (all_logits, labels),
    per_class_thresholds=config.per_class_thresholds,
)

print("\n[metrics] en-custom")
print(json.dumps(metrics, indent=4))

os.makedirs(REPORT_DIR, exist_ok=True)

with open(
    os.path.join(REPORT_DIR, "metrics.json"),
    "w",
    encoding="utf-8",
) as f:

    json.dump(metrics, f, indent=4, ensure_ascii=False)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

report = classification_report(
    labels,
    predictions,
    target_names=LABELS,
    zero_division=0,
)

print("\n[classification_report] en-custom\n")
print(report)

with open(
    os.path.join(REPORT_DIR, "classification_report.txt"),
    "w",
    encoding="utf-8",
) as f:

    f.write(report)

# =====================================================
# CONFUSION MATRIX
# =====================================================

for i, label in enumerate(LABELS):

    cm = confusion_matrix(labels[:, i], predictions[:, i])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()

    plt.title(f"en-custom - {label}")

    plt.tight_layout()

    plt.savefig(os.path.join(REPORT_DIR, f"confusion_matrix_{label}.png"))

    plt.close()

# =====================================================
# ROC CURVE
# =====================================================

for i, label in enumerate(LABELS):

    fpr, tpr, _ = roc_curve(labels[:, i], probabilities[:, i])

    plt.figure()

    plt.plot(fpr, tpr)

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title(f"en-custom - {label}")

    plt.grid()

    plt.tight_layout()

    plt.savefig(os.path.join(REPORT_DIR, f"roc_curve_{label}.png"))

    plt.close()

# =====================================================
# PR CURVE
# =====================================================

for i, label in enumerate(LABELS):

    precision, recall, _ = precision_recall_curve(
        labels[:, i],
        probabilities[:, i],
    )

    plt.figure()

    plt.plot(recall, precision)

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title(f"en-custom - {label}")

    plt.grid()

    plt.tight_layout()

    plt.savefig(os.path.join(REPORT_DIR, f"pr_curve_{label}.png"))

    plt.close()

# =====================================================
# PREDICTIONS
# =====================================================

rows = []

for i in range(len(df)):

    row = {}

    for j, label in enumerate(LABELS):

        row[f"{label}_true"] = int(labels[i][j])

        row[f"{label}_pred"] = int(predictions[i][j])

        row[f"{label}_score"] = round(float(probabilities[i][j]), 4)

    rows.append(row)

prediction_df = pd.DataFrame(rows)

prediction_df["text"] = df["normalized_text"].values

prediction_df.to_csv(
    os.path.join(REPORT_DIR, "prediction_examples.csv"),
    index=False,
)

# =====================================================
# FINISH
# =====================================================

print(f"\nReports saved to : {REPORT_DIR}")
print("=" * 60)