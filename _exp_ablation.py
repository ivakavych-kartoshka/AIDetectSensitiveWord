import json

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, average_precision_score

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config.config import Config
from src.data.dataset_loader import create_label_vector, load_dataframe
from src.models.custom_model import SensitiveCustomModel
from src.training.metrics import Metrics, LABELS

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 64

GRID = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]


def fbeta(p, r, beta=1.0):
    if p is None or r is None:
        return 0.0
    b2 = beta * beta
    d = b2 * p + r
    return 0.0 if d <= 0 else (1 + b2) * p * r / d


def load_v0(path):
    m = AutoModelForSequenceClassification.from_pretrained(path)
    m.to(device)
    m.eval()
    tok = AutoTokenizer.from_pretrained(path)
    return m, tok


def load_custom(path):
    m = SensitiveCustomModel.from_pretrained(path, num_labels=6)
    m.to(device)
    m.eval()
    tok = AutoTokenizer.from_pretrained(path)
    return m, tok


def infer(m, tok, texts):
    batch = tok(list(texts), truncation=True, padding="max_length", max_length=config.max_length, return_tensors="pt")
    chunks = []
    for i in range(0, len(texts), BATCH):
        ins = {k: v[i:i + BATCH].to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = m(**ins)
        lg = out["logits"] if isinstance(out, dict) else out.logits
        chunks.append(lg.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def tune(val_labels, val_probs):
    best = {}
    for i, label in enumerate(LABELS):
        col = val_labels[:, i]
        if col.sum() <= 0:
            best[label] = 0.5
            continue
        best_score, best_thr, best_f1 = -1.0, 0.5, -1.0
        for thr in GRID:
            pred = (val_probs[:, i] >= thr).astype(int)
            p = precision_recall_fscore_support(col, pred, average="binary", zero_division=0)[0]
            r = precision_recall_fscore_support(col, pred, average="binary", zero_division=0)[1]
            score = fbeta(p, r, 1.0)
            if score > best_score or (score == best_score and score > best_f1):
                best_score, best_thr, best_f1 = score, thr, score
        best[label] = best_thr
    return best


def evaluate(name, m, tok):
    val_df = create_label_vector(load_dataframe(config.val_file))
    val_df = val_df[val_df["language"] == "vi"].reset_index(drop=True)
    test_df = create_label_vector(load_dataframe(config.test_file))
    test_df = test_df[test_df["language"] == "vi"].reset_index(drop=True)

    val_probs = Metrics.predict_probability(infer(m, tok, val_df["normalized_text"].tolist()))
    test_probs = Metrics.predict_probability(infer(m, tok, test_df["normalized_text"].tolist()))
    val_labels = np.array(val_df["labels"].tolist())
    test_labels = np.array(test_df["labels"].tolist())

    thr = tune(val_labels, val_probs)
    thr_arr = np.array([thr[l] for l in LABELS])
    test_pred = (test_probs >= thr_arr).astype(int)
    for l, t in thr.items():
        print(f"    {l} thr={t:.2f}")
    p, r, f, _ = precision_recall_fscore_support(test_labels, test_pred, average="micro", zero_division=0)
    acc = accuracy_score(test_labels, test_pred)
    try:
        roc = roc_auc_score(test_labels, test_probs, average="micro")
    except Exception:
        roc = float("nan")
    try:
        prauc = average_precision_score(test_labels, test_probs, average="micro")
    except Exception:
        prauc = float("nan")
    pp = precision_recall_fscore_support(test_labels, test_pred, average=None, zero_division=0)
    print(f"    micro P={p:.4f} R={r:.4f} F1={f:.4f} Acc={acc:.4f} ROC={roc:.4f} PR={prauc:.4f}")
    for i, l in enumerate(LABELS):
        print(f"      {l:<12} P={pp[0][i]:.4f} R={pp[1][i]:.4f} F1={pp[2][i]:.4f} sup={int(test_labels[:, i].sum())}")
    return {"name": name, "thresholds": thr, "P": round(p,4), "R": round(r,4), "F1": round(f,4),
            "Acc": round(acc,4), "ROC": round(roc,4), "PR": round(prauc,4),
            "per_label": {l: {"P": round(pp[0][i],4), "R": round(pp[1][i],4), "F1": round(pp[2][i],4),
                              "sup": int(test_labels[:, i].sum())} for i, l in enumerate(LABELS)}}

out = {}
print("== sensitiveai-vi (standard fine-tuning) ==")
m, tok = load_v0("models/sensitiveai-vi")
out["vi"] = evaluate("vi", m, tok)
del m; torch.cuda.empty_cache()

print("== sensitiveai-v1-vi (intermediate checkpoint) ==")
m, tok = load_v0("models/sensitiveai-v1-vi")
out["v1_vi"] = evaluate("v1_vi", m, tok)
del m; torch.cuda.empty_cache()

print("== sensitiveai-vi-custom-v2 ==")
m, tok = load_custom("models/sensitiveai-vi-custom-v2")
out["custom_v2"] = evaluate("custom_v2", m, tok)
del m; torch.cuda.empty_cache()

print("== sensitiveai-vi-custom-dlr2stage ==")
m, tok = load_custom("models/sensitiveai-vi-custom-dlr2stage")
out["dlr2stage"] = evaluate("dlr2stage", m, tok)
del m; torch.cuda.empty_cache()

with open("_exp_ablation_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("saved _exp_ablation_results.json")