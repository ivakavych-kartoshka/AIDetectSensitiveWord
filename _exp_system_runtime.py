import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from src.config.config import Config
from src.data.dataset_loader import create_label_vector, load_dataframe
from src.rules.rule_engine import RuleEngine
from src.training.metrics import LABELS

config = Config()

df = create_label_vector(load_dataframe(config.test_file))
df = df[df["language"] == "vi"].reset_index(drop=True)

y = np.array(df["labels"].tolist())
y_bin = ((y[:, 0] == 1) | (y[:, 1] == 1)).astype(int)

probs = np.load("reports/system_eval/_exp_ai_probs.npy")  # sigmoid probs, normalized text
rule = RuleEngine()

rule_scores = []
rule_has_sens = []
for t in df["text"].tolist():
    r = rule.analyze(t)
    rule_scores.append(r["score"])
    rule_has_sens.append(int(r["has_sensitive"]))
rule_scores = np.array(rule_scores)
rule_has_sens = np.array(rule_has_sens)

# Runtime AI: labels predicted at flat 0.5; ml_score = max confidence among predicted labels
pred_flat = (probs >= 0.5).astype(int)
ai_pred_any = (pred_flat.any(axis=1)).astype(int)
with np.errstate(invalid="ignore"):
    ml_score = np.where(pred_flat.any(axis=1), probs.max(axis=1), 0.0)

# Runtime hybrid decision (src/detector.py)
decision = np.full(len(df), "review", dtype=object)
rule_block = rule_scores >= 0.90
rule_review = rule_scores >= 0.60
decision[~rule_review] = "allow" if False else "allow"
decision = np.where(rule_block, "block", np.where(rule_review, "review", "allow"))
decision = np.where(ml_score >= 0.95, "block", decision)
decision = np.where((ml_score >= 0.80) & (decision == "allow"), "review", decision)

final_score = np.maximum(rule_scores, ml_score)
has_sensitive = np.clip(rule_has_sens + ai_pred_any, 0, 1)

print("=== Runtime decision policy (rule 0.90/0.60; AI override 0.95/0.80) ===")
for d in ["allow", "review", "block"]:
    m = decision == d
    print(f"{d:<8} total={int(m.sum()):>5}  sensitive={int((m & (y_bin==1)).sum()):>5}  "
          f"non-sensitive={int((m & (y_bin==0)).sum()):>5}")
missed = int(((decision == "allow") & (y_bin == 1)).sum())
print(f"Sensitive passed as ALLOW: {missed} / {y_bin.sum()} = {missed/y_bin.sum():.1%}")

print("\n=== Binary flag metrics (flag = decision in {review, block}) ===")
for name, pred in [
    ("flag=rule match only", rule_has_sens),
    ("flag=AI pred only (flat .5)", ai_pred_any),
    ("flag=union (rule|AI)", has_sensitive),
    ("flag=runtime decision (review|block)", (decision != "allow").astype(int)),
]:
    p, r, f, _ = precision_recall_fscore_support(y_bin, pred, average="binary", zero_division=0)
    acc = accuracy_score(y_bin, pred)
    print(f"{name:<32} P={p:.3f} R={r:.3f} F1={f:.3f} Acc={acc:.3f}  PY={int((pred==1).sum())}")

with open("reports/system_eval/runtime_decision.json", "w", encoding="utf-8") as f:
    json.dump({
        "policy": "rule block>=0.90 review>=0.60; AI override block>=0.95 review>=0.80; max(rule, ai)",
        "n": len(df), "sensitive": int(y_bin.sum()),
        "decisions": {d: int((decision == d).sum()) for d in ["allow", "review", "block"]},
        "sensitive_by_decision": {d: int(((decision == d) & (y_bin == 1)).sum()) for d in ["allow", "review", "block"]},
        "missed_allow": missed,
    }, f, indent=2, ensure_ascii=False)
print("saved reports/system_eval/runtime_decision.json")