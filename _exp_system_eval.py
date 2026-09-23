import json
import numpy as np
import pandas as pd
import torch
import os

from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from src.config.config import Config
from src.data.dataset_loader import create_label_vector, load_dataframe
from src.models.custom_model import SensitiveCustomModel
from src.rules.rule_engine import RuleEngine
from src.training.metrics import Metrics, LABELS

config = Config()

TUNED = {"insult": 0.25, "hate_speech": 0.30, "threat": 0.5, "harassment": 0.5, "sexual": 0.5, "spam": 0.5}
FLAT = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 64

test_df = load_dataframe(config.test_file)
test_df = create_label_vector(test_df)
df = test_df[test_df["language"] == "vi"].reset_index(drop=True)
print("Rows:", len(df))

y = df[["insult", "hate_speech", "threat", "harassment", "sexual", "spam"]].values.astype(int)
y_bin = ((y[:, 0] == 1) | (y[:, 1] == 1)).astype(int)
print("Sensitive (insult|hate):", int(y_bin.sum()))

# ---------- AI inference on normalized_text ----------
model = SensitiveCustomModel.from_pretrained("models/sensitiveai-vi-custom-dlr2stage", num_labels=6)
model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("models/sensitiveai-vi-custom-dlr2stage")

probs = None
batch = tokenizer(df["normalized_text"].tolist(), truncation=True, padding="max_length",
                  max_length=config.max_length, return_tensors="pt")
chunks = []
for i in range(0, len(df), BATCH):
    ins = {k: v[i:i+BATCH].to(device) for k, v in batch.items()}
    with torch.no_grad():
        out = model(**ins)
    chunks.append(out["logits"].cpu().numpy())
logits = np.concatenate(chunks, axis=0)
probs = Metrics.predict_probability(logits)
np.save(os.path.join("_exp_ai_probs.npy"), probs)
np.save(os.path.join("_exp_y.npy"), y)

thr_arr = np.array([TUNED[l] for l in LABELS])
pred_tuned = (probs >= thr_arr).astype(int)
pred_flat = (probs >= FLAT).astype(int)

ai_tuned_bin = ((pred_tuned[:, 0]) | (pred_tuned[:, 1])).astype(int)
ai_flat_bin = ((pred_flat[:, 0]) | (pred_flat[:, 1])).astype(int)
ai_score = probs.max(axis=1)

# ---------- Rule inference ----------
rule = RuleEngine()
rule_has_sens = []
rule_score = []
for t in df["text"].tolist():
    r = rule.analyze(t)
    rule_has_sens.append(int(r["has_sensitive"]))
    rule_score.append(r["score"])
rule_has_sens = np.array(rule_has_sens)
rule_score = np.array(rule_score)

# ---------- Variants ----------
def results(name, pred):
    p, r, f, _ = precision_recall_fscore_support(y_bin, pred, average="binary", zero_division=0)
    acc = accuracy_score(y_bin, pred)
    tp = int(((y_bin == 1) & (pred == 1)).sum())
    fp = int(((y_bin == 0) & (pred == 1)).sum())
    fn = int(((y_bin == 1) & (pred == 0)).sum())
    tn = int(((y_bin == 0) & (pred == 0)).sum())
    print(f"{name:<34} P={p:.3f} R={r:.3f} F1={f:.3f} Acc={acc:.3f} TP={tp} FP={fp} FN={fn} TN={tn}")
    return {"name": name, "P": round(p,3), "R": round(r,3), "F1": round(f,3), "Acc": round(acc,3),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}

out = []
print("\n=== System-level (binary sensitive vs clean; flag = any positive) ===")
out.append(results("Rule-only (score>0)", (rule_score > 0).astype(int)))
out.append(results("Rule-only (score>=0.30)", (rule_score >= 0.30).astype(int)))
out.append(results("AI-only flat 0.5", ai_flat_bin))
out.append(results("AI-only tuned (0.25/0.30)", ai_tuned_bin))
out.append(results("Rule + AI tuned (union)", np.clip(rule_has_sens + ai_tuned_bin, 0, 1)))
out.append(results("Rule + AI flat (union)", np.clip(rule_has_sens + ai_flat_bin, 0, 1)))

# Confidence-aware hybrid: final score = max(rule_score, ai_score), decision by paper policy
fin = np.maximum(rule_score, ai_score)
out.append(results("Hybrid max-score (>=0.30)", (fin >= 0.30).astype(int)))
out.append(results("Hybrid max-score (>=0.80)", (fin >= 0.80).astype(int)))

# 3-way decision mapping on final score: <0.30 allow / <0.80 review / >=0.80 block
def decision3(s):
    if s < 0.30: return "allow"
    if s < 0.80: return "review"
    return "block"
dec = [decision3(s) for s in fin]
dec = np.array(dec)
sens = y_bin == 1
print("\n=== Decision-level (paper policy 0.30/0.80) ===")
for d in ["allow", "review", "block"]:
    m = dec == d
    print(f"{d:<8} total={int(m.sum()):>5}  sensitive={int((m & sens).sum()):>5}  non-sensitive={int((m & ~sens).sum()):>5}")
missed = int((sens & (dec == "allow")).sum())
print("Sensitive comments that passed as ALLOW:", missed, f"({missed/sens.sum():.1%} of {sens.sum()})")

with open("_exp_system_results.json", "w", encoding="utf-8") as f:
    json.dump({"rows": len(df), "sensitive": int(sens.sum()), "variants": out,
               "missed_allow": missed, "decision_counts": {d: int((dec==d).sum()) for d in ["allow","review","block"]}},
              f, indent=2, ensure_ascii=False)
print("\nsaved _exp_system_results.json")