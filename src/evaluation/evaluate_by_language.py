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
)

from transformers import AutoModelForSequenceClassification

from src.config.config import Config
from src.data.dataset_loader import (
    create_label_vector,
    load_dataframe,
    tokenize,
)
from src.training.metrics import Metrics, LABELS

# =====================================================
# CONFIG
# =====================================================

config = Config()

BASE_REPORT_DIR = "reports"

LANGUAGES = ["en", "vi"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 60)
print("SensitiveAI Evaluation By Language")
print("=" * 60)

test_df = load_dataframe(config.test_file)
test_df = create_label_vector(test_df)

print("\nLanguage distribution:")

print(test_df["language"].value_counts().to_string())

# =====================================================
# LOAD MODEL
# =====================================================

print(f"\nDevice : {device}")

model = AutoModelForSequenceClassification.from_pretrained(config.output_dir)

model.to(device)

model.eval()

# =====================================================
# EVALUATE
# =====================================================


def evaluate_language(df, language):

    report_dir = os.path.join(
        BASE_REPORT_DIR,
        language,
    )

    os.makedirs(report_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Language : {language}")
    print(f"Samples  : {len(df)}")

    print(f"{'='*60}")

    batch = tokenize(
        {
            "normalized_text": df["normalized_text"].tolist(),
        }
    )

    all_logits = []
    all_labels = []

    for i in range(len(df)):

        inputs = {
            "input_ids": torch.tensor(
                batch["input_ids"][i]
            ).unsqueeze(0).to(device),
            "attention_mask": torch.tensor(
                batch["attention_mask"][i]
            ).unsqueeze(0).to(device),
        }

        with torch.no_grad():

            outputs = model(**inputs)

        all_logits.append(outputs.logits.squeeze().cpu().numpy())

        all_labels.append(df["labels"].iloc[i])

    all_logits = np.array(all_logits)

    all_labels = np.array(all_labels)

    # =====================
    # Metrics
    # =====================

    metrics = Metrics.calculate(
        (
            all_logits,
            all_labels,
        )
    )

    print(json.dumps(metrics, indent=4))

    with open(
        os.path.join(
            report_dir,
            "metrics.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
            ensure_ascii=False,
        )

    # =====================
    # Classification Report
    # =====================

    probabilities = Metrics.predict_probability(all_logits)

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
            report_dir,
            "classification_report.txt",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        f.write(report)

    # =====================
    # Confusion Matrix
    # =====================

    for i, label in enumerate(LABELS):

        cm = confusion_matrix(
            all_labels[:, i],
            predictions[:, i],
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
        )

        disp.plot()

        plt.title(f"{language} - {label}")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                report_dir,
                f"confusion_matrix_{label}.png",
            )
        )

        plt.close()

    # =====================
    # Predictions
    # =====================

    rows = []

    for i in range(len(df)):

        row = {}

        for j, label in enumerate(LABELS):

            row[f"{label}_true"] = int(all_labels[i][j])

            row[f"{label}_pred"] = int(predictions[i][j])

            row[f"{label}_score"] = round(
                float(probabilities[i][j]),
                4,
            )

        rows.append(row)

    prediction_df = pd.DataFrame(rows)

    prediction_df["text"] = df["normalized_text"].values

    prediction_df.to_csv(
        os.path.join(
            report_dir,
            "prediction_examples.csv",
        ),
        index=False,
    )

    print(f"\nSaved reports -> {report_dir}")


for language in LANGUAGES:

    language_df = test_df[test_df["language"] == language]

    if len(language_df) == 0:

        print(f"\nNo samples for language : {language}")

        continue

    evaluate_language(language_df, language)

# =====================================================
# FINISH
# =====================================================

print("\n" + "=" * 60)
print("Evaluation By Language Finished")
print("=" * 60)