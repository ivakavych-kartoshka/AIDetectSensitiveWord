import json
import os

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

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config.config import Config
from src.data.dataset_loader import (
    create_label_vector,
    load_dataframe,
)
from src.training.metrics import Metrics, LABELS

# =====================================================
# CONFIG
# =====================================================

config = Config()

MODEL_NAME = "microsoft/mdeberta-v3-base"
REPORT_DIR = "reports/raw-mdeberta-vi"
TEST_LANG = "vi"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64

# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 60)
print("Raw mDeBERTa-v3-base baseline (untrained) - vi")
print("=" * 60)

test_df = load_dataframe(config.test_file)
test_df = create_label_vector(test_df)

df = test_df[test_df["language"] == TEST_LANG]

print(f"Samples : {len(df)}")
print(f"Device  : {device}")

# =====================================================
# LOAD MODEL (raw, no training)
# =====================================================

print("\nLoading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=config.num_labels,
    problem_type="multi_label_classification",
)

model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

        all_logits.append(outputs.logits.cpu().numpy())

all_logits = np.concatenate(all_logits, axis=0)

labels = np.array(df["labels"].tolist())

probabilities = Metrics.predict_probability(all_logits)

predictions = Metrics.predict_binary(all_logits)

# =====================================================
# METRICS
# =====================================================

metrics = Metrics.calculate((all_logits, labels))

print("\n[metrics] raw-mdeberta-vi")
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

print("\n[classification_report] raw-mdeberta-vi\n")
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

    plt.title(f"raw-mdeberta-vi - {label}")

    plt.tight_layout()

    plt.savefig(os.path.join(REPORT_DIR, f"confusion_matrix_{label}.png"))

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
