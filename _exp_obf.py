import json
import os
import random

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer

from src.config.config import Config
from src.data.augmentation import (
    apply_homoglyph,
    apply_leetspeak,
    apply_abbreviation,
    random_case,
    add_special,
    add_zero_width,
)
from src.data.normalize import normalize_text
from src.data.dataset_loader import create_label_vector, load_dataframe
from src.models.custom_model import SensitiveCustomModel
from src.rules.rule_engine import RuleEngine
from src.training.metrics import Metrics, LABELS

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 64

# Sensitive-only rows from root clean test (vi)
test_df = load_dataframe(config.test_file)
test_df = create_label_vector(test_df)
df = test_df[test_df["language"] == "vi"].reset_index(drop=True)
sens = df[((df["insult"] == 1) | (df["hate_speech"] == 1))].reset_index(drop=True)
print("Sensitive test comments:", len(sens))

model = SensitiveCustomModel.from_pretrained("models/sensitiveai-vi-custom-dlr2stage", num_labels=6)
model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("models/sensitiveai-vi-custom-dlr2stage")

TUNED = np.array([0.25, 0.30, 0.5, 0.5, 0.5, 0.5])


def ai_any(texts):
    out = []
    batch = tokenizer(list(texts), truncation=True, padding=True, max_length=config.max_length, return_tensors="pt")
    for i in range(0, len(texts), BATCH):
        ins = {k: v[i:i + BATCH].to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**ins)["logits"].cpu().numpy()
        probs = Metrics.sigmoid(logits)
        pred = (probs >= TUNED).astype(int)
        out.append(((pred[:, 0] == 1) | (pred[:, 1] == 1)).astype(int))
    return np.concatenate(out)


def normalize_all(texts):
    return [normalize_text(t)["combined"] if isinstance(normalize_text(t), dict) else normalize_text(t) for t in texts]


rule = RuleEngine()
rule_has = []
for t in sens["text"].tolist():
    rule_has.append(int(rule.analyze(t)["has_sensitive"]))
rule_has = np.array(rule_has)

# AI on original raw text (deployment behavior) and normalized text (evaluation behavior)
ai_raw_orig = ai_any(sens["text"].tolist())
ai_norm_orig = ai_any(normalize_all(sens["text"].tolist()))

print(f"\nORIGINAL texts (N={len(sens)}):")
print(f"  Rule keyword hit        : {rule_has.sum()} ({rule_has.mean():.1%})")
print(f"  AI on raw text          : {ai_raw_orig.sum()} ({ai_raw_orig.mean():.1%})")
print(f"  AI on normalized text   : {ai_norm_orig.sum()} ({ai_norm_orig.mean():.1%})")

techniques = {
    "homoglyph": lambda t: apply_homoglyph(t),
    "leetspeak": lambda t: apply_leetspeak(t),
    "abbreviation": lambda t: apply_abbreviation(t),
    "random_case": lambda t: random_case(t),
    "special_chars": lambda t: add_special(t),
    "zero_width": lambda t: add_zero_width(t),
}

report = {"original": {"rule": int(rule_has.sum()), "ai_raw": int(ai_raw_orig.sum()),
                       "ai_norm": int(ai_norm_orig.sum()), "n": len(sens)}}


def combined(t, n=3):
    s = t
    funcs = random.sample(list(techniques.values()), n)
    random.shuffle(funcs)
    for f in funcs:
        s = f(s)
    return s


for name in list(techniques.keys()) + ["combined_3"]:
    random.seed(42)
    texts = []
    for t in sens["text"].tolist():
        if name == "combined_3":
            texts.append(combined(t))
        else:
            texts.append(techniques[name](t))
    r_has = [int(rule.analyze(t)["has_sensitive"]) for t in texts]
    r_has = np.array(r_has)
    ai_raw = ai_any(texts)
    ai_norm = ai_any(normalize_all(texts))
    rule_or_ai = ((r_has == 1) | (ai_raw == 1)).astype(int)
    print(f"\n{name:<14} (N={len(texts)}):")
    print(f"  Rule keyword hit        : {r_has.sum()} ({r_has.mean():.1%})")
    print(f"  AI on raw text          : {ai_raw.sum()} ({ai_raw.mean():.1%})")
    print(f"  AI on normalized text   : {ai_norm.sum()} ({ai_norm.mean():.1%})")
    print(f"  Hybrid (rule | ai_raw)  : {rule_or_ai.sum()} ({rule_or_ai.mean():.1%})")
    report[name] = {"rule": int(r_has.sum()), "ai_raw": int(ai_raw.sum()),
                    "ai_norm": int(ai_norm.sum()), "hybrid_rule_ai_raw": int(rule_or_ai.sum())}

with open("_exp_obf_results.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("\nsaved _exp_obf_results.json")