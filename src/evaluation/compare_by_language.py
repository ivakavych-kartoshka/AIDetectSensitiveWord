import json
import os

import numpy as np
import pandas as pd
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config.config import Config
from src.data.dataset_loader import (
    create_label_vector,
    load_dataframe,
)
from src.training.metrics import Metrics

# =====================================================
# CONFIG
# =====================================================

config = Config()

REPORT_DIR = "reports/compare"

os.makedirs(REPORT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():

    print(f"\nDevice : CUDA ({torch.cuda.get_device_name(0)})")

else:

    print("\nDevice : CPU")

# =====================================================
# MODELS TO COMPARE
# (skip if folder does not exist)
# =====================================================

MODELS = [
    {
        "key": "vi_only",
        "name": "Chiến lược 1 - từ base, chỉ tiếng Việt",
        "path": "models/sensitiveai-vi",
    },
    {
        "key": "en_only",
        "name": "Chiến lược 1 - từ base, chỉ tiếng Anh",
        "path": "models/sensitiveai-en",
    },
    {
        "key": "combined",
        "name": "Chiến lược 2 - gộp 2 ngôn ngữ (sensitiveai-v1)",
        "path": "models/sensitiveai-v1",
    },
    {
        "key": "v1_vi",
        "name": "Chiến lược 3 - từ v1, fine-tune tiếng Việt",
        "path": "models/sensitiveai-v1-vi",
    },
    {
        "key": "v1_en",
        "name": "Chiến lược 3 - từ v1, fine-tune tiếng Anh",
        "path": "models/sensitiveai-v1-en",
    },
]

TEST_LANGUAGES = ["en", "vi"]

# =====================================================
# LOAD TEST DATA BY LANGUAGE
# =====================================================

print("=" * 70)
print("SensitiveAI Compare Models By Language")
print("=" * 70)

test_df = load_dataframe(config.test_file)
test_df = create_label_vector(test_df)

language_dfs = {}

for lang in TEST_LANGUAGES:

    language_dfs[lang] = test_df[test_df["language"] == lang]

    print(f"{lang} test samples : {len(language_dfs[lang])}")

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

    batch_size = 64

    for i in range(0, len(df), batch_size):

        inputs_ids = batch["input_ids"][i : i + batch_size].to(device)
        attention = batch["attention_mask"][i : i + batch_size].to(device)

        with torch.no_grad():

            outputs = model(
                input_ids=inputs_ids,
                attention_mask=attention,
            )

        all_logits.append(outputs.logits.cpu().numpy())

    return np.concatenate(all_logits, axis=0)


def evaluate_metrics(logits, labels):

    return Metrics.calculate(
        (
            logits,
            labels,
        )
    )

# =====================================================
# RUN COMPARISON
# =====================================================

results = {}

for model_info in MODELS:

    key = model_info["key"]
    path = model_info["path"]

    if not os.path.isdir(path):

        print(f"\nSKIP (not found) : {model_info['name']} -> {path}")
        continue

    print(f"\nLoading model : {model_info['name']}")

    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(path)

    results[key] = {
        "name": model_info["name"],
        "path": path,
    }

    for lang in TEST_LANGUAGES:

        df = language_dfs[lang]

        labels = np.array(df["labels"].tolist())

        logits = infer_model(model, tokenizer, df)

        metrics = evaluate_metrics(logits, labels)

        results[key][f"test_{lang}"] = metrics

        print(f"  {lang:>4} : "f"{json.dumps(metrics)}")

    del model

    torch.cuda.empty_cache()

# =====================================================
# BUILD COMPARISON TABLE
# =====================================================

rows = []

for key, value in results.items():

    for lang in TEST_LANGUAGES:

        metrics = value.get(f"test_{lang}")

        if metrics is None:
            continue

        row = {
            "model": value["name"],
            "key": key,
            "test_language": lang,
        }

        for metric_name, m_value in metrics.items():

            row[metric_name] = m_value

        rows.append(row)

compare_df = pd.DataFrame(rows)

compare_csv = os.path.join(REPORT_DIR, "comparison.csv")

compare_df.to_csv(compare_csv, index=False)

print("\n" + "=" * 70)
print("COMPARISON TABLE")
print("=" * 70)

print(compare_df.to_string(index=False))

# =====================================================
# SAVE MARKDOWN
# =====================================================

metric_names = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]

markdown = []

markdown.append("# So sánh các mô hình theo ngôn ngữ\n")

markdown.append(
    "So sánh 3 chiến lược trên test set tiếng Anh (en) và tiếng Việt (vi).\n"
)

markdown.append("## Chiến lược")
markdown.append("- **1**: Huấn luyện riêng từng ngôn ngữ từ base.")
markdown.append("- **2**: Huấn luyện 2 ngôn ngữ gộp chung (`sensitiveai-v1`).")
markdown.append("- **3**: Dùng model gộp (v1) tiếp tục fine-tune riêng từng ngôn ngữ.\n")

for key, value in results.items():

    markdown.append(f"## {value['name']}\n")

    header = "| test_language | " + " | ".join(metric_names) + " |"
    separator = "|---|---" + "|---" * len(metric_names) + "|"

    markdown.append(header)
    markdown.append(separator)

    for lang in TEST_LANGUAGES:

        metrics = value.get(f"test_{lang}")

        if metrics is None:
            continue

        row = (
            f"| {lang} | "
            + " | ".join(str(metrics[m]) for m in metric_names)
            + " |"
        )

        markdown.append(row)

    markdown.append("")

# =====================================================
# BEST PER LANGUAGE
# =====================================================

markdown.append("## Model tốt nhất theo từng ngôn ngữ kiểm thử (theo F1)\n")

for lang in TEST_LANGUAGES:

    best_key = None
    best_f1 = -1.0

    for key, value in results.items():

        metrics = value.get(f"test_{lang}")

        if metrics is None:
            continue

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]
            best_key = key

    if best_key:

        markdown.append(
            f"- **Test tiếng {lang.upper()}**: **{results[best_key]['name']}** "
            f"(F1 = {best_f1})"
        )

# =====================================================
# CROSS-LANGUAGE MATRIX (per language, per model)
# =====================================================

markdown.append("\n## Ma trận F1 (mô hình x ngôn ngữ kiểm thử)\n")

header = "| model | " + " | ".join(f"{l.upper()}" for l in TEST_LANGUAGES) + " |"
separator = "|---|---" + "|---" * len(TEST_LANGUAGES) + "|"

markdown.append(header)
markdown.append(separator)

for key, value in results.items():

    cells = []

    for lang in TEST_LANGUAGES:

        metrics = value.get(f"test_{lang}")

        if metrics is None:

            cells.append("-")
        else:

            cells.append(str(metrics["f1"]))

    markdown.append(f"| {value['name']} | " + " | ".join(cells) + " |")

with open(
    os.path.join(REPORT_DIR, "comparison.md"),
    "w",
    encoding="utf-8",
) as f:

    f.write("\n".join(markdown))

# =====================================================
# SAVE JSON
# =====================================================

with open(
    os.path.join(REPORT_DIR, "comparison.json"),
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False,
    )

# =====================================================
# FINISH
# =====================================================

print("\n" + "=" * 70)
print("Compare By Language Finished")
print(f"Saved to : {REPORT_DIR}")
print("=" * 70)
