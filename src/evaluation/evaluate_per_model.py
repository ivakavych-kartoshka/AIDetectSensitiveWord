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

BASE_REPORT_DIR = "reports"

# Each model only evaluated on its own language
MODELS = [
    {
        "key": "vi",
        "name": "sensitiveai-vi",
        "path": "models/sensitiveai-vi",
        "test_lang": "vi",
    },
    {
        "key": "en",
        "name": "sensitiveai-en",
        "path": "models/sensitiveai-en",
        "test_lang": "en",
    },
    {
        "key": "v1-vi",
        "name": "sensitiveai-v1-vi",
        "path": "models/sensitiveai-v1-vi",
        "test_lang": "vi",
    },
    {
        "key": "v1-en",
        "name": "sensitiveai-v1-en",
        "path": "models/sensitiveai-v1-en",
        "test_lang": "en",
    },
    {
        "key": "v1",
        "name": "sensitiveai-v1",
        "path": "models/sensitiveai-v1",
        "test_lang": ["en", "vi"],
    },
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64

# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 60)
print("SensitiveAI Evaluation Per Model")
print("=" * 60)

test_df = load_dataframe(config.test_file)
test_df = create_label_vector(test_df)

print("\nLanguage distribution:")
print(test_df["language"].value_counts().to_string())

print(f"\nDevice : {device}")

# =====================================================
# INFERENCE
# =====================================================


def infer_model(model, tokenizer, df):

    model.eval()

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

    return np.concatenate(all_logits, axis=0)

# =====================================================
# EVALUATE ONE MODEL + LANGUAGE
# =====================================================


def evaluate(model, tokenizer, df, report_dir, title):

    os.makedirs(report_dir, exist_ok=True)

    labels = np.array(df["labels"].tolist())

    logits = infer_model(model, tokenizer, df)

    probabilities = Metrics.predict_probability(logits)

    predictions = Metrics.predict_binary(logits)

    # =====================
    # Metrics
    # =====================

    metrics = Metrics.calculate((logits, labels))

    print(f"\n[metrics] {title}")
    print(json.dumps(metrics, indent=4))

    with open(
        os.path.join(report_dir, "metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(metrics, f, indent=4, ensure_ascii=False)

    # =====================
    # Classification Report
    # =====================

    report = classification_report(
        labels,
        predictions,
        target_names=LABELS,
        zero_division=0,
    )

    print(f"\n[classification_report] {title}\n")
    print(report)

    with open(
        os.path.join(report_dir, "classification_report.txt"),
        "w",
        encoding="utf-8",
    ) as f:

        f.write(report)

    # =====================
    # Confusion Matrix
    # =====================

    for i, label in enumerate(LABELS):

        cm = confusion_matrix(labels[:, i], predictions[:, i])

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        disp.plot()

        plt.title(f"{title} - {label}")

        plt.tight_layout()

        plt.savefig(os.path.join(report_dir, f"confusion_matrix_{label}.png"))

        plt.close()

    # =====================
    # ROC Curve
    # =====================

    for i, label in enumerate(LABELS):

        fpr, tpr, _ = roc_curve(labels[:, i], probabilities[:, i])

        plt.figure()

        plt.plot(fpr, tpr)

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title(f"{title} - {label}")

        plt.grid()

        plt.tight_layout()

        plt.savefig(os.path.join(report_dir, f"roc_curve_{label}.png"))

        plt.close()

    # =====================
    # PR Curve
    # =====================

    for i, label in enumerate(LABELS):

        precision, recall, _ = precision_recall_curve(
            labels[:, i],
            probabilities[:, i],
        )

        plt.figure()

        plt.plot(recall, precision)

        plt.xlabel("Recall")

        plt.ylabel("Precision")

        plt.title(f"{title} - {label}")

        plt.grid()

        plt.tight_layout()

        plt.savefig(os.path.join(report_dir, f"pr_curve_{label}.png"))

        plt.close()

    # =====================
    # Predictions
    # =====================

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
        os.path.join(report_dir, "prediction_examples.csv"),
        index=False,
    )

    print(f"\nSaved reports -> {report_dir}")

# =====================================================
# RUN
# =====================================================

for model_info in MODELS:

    model_key = model_info["key"]
    model_path = model_info["path"]
    test_langs = model_info["test_lang"]

    if isinstance(test_langs, str):

        test_langs = [test_langs]

    if not os.path.isdir(model_path):

        print(f"\nSKIP (not found) : {model_info['name']} -> {model_path}")

        continue

    print(f"\n{'='*60}")
    print(f"Model : {model_info['name']}")
    print(f"Test  : {', '.join(test_langs)}")
    print(f"{'='*60}")

    # Only load the model once per model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.to(device)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for test_lang in test_langs:

        df = test_df[test_df["language"] == test_lang]

        print(f"\n[{test_lang}] Samples : {len(df)}")

        if len(test_langs) > 1:

            # Split model -> store per-language in a subfolder
            report_dir = os.path.join(
                BASE_REPORT_DIR,
                model_key,
                test_lang,
            )

            title = f"{model_key} ({test_lang})"

        else:

            report_dir = os.path.join(
                BASE_REPORT_DIR,
                model_key,
            )

            title = model_key

        evaluate(
            model,
            tokenizer,
            df,
            report_dir,
            title,
        )

        torch.cuda.empty_cache()

    del model

    torch.cuda.empty_cache()

# =====================================================
# FINISH
# =====================================================

print("\n" + "=" * 60)
print("Evaluation Per Model Finished")
print("=" * 60)