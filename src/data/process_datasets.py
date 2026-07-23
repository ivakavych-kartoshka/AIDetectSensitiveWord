from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from src.data.normalize import normalize_text  # Import hàm normalize

print("🚀 Bắt đầu xử lý dataset cho SensitiveAI...")

# ====================== 1. TẢI DATASET ======================
print("📥 Đang tải ViHSD (tiếng Việt)...")
vi_ds = load_dataset("uitnlp/vihsd", split="train", trust_remote_code=True)
vi_df = vi_ds.to_pandas()

print("📥 Đang tải Jigsaw Toxic (tiếng Anh)...")
en_ds = load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge")
en_df = en_ds["train"].to_pandas()

print(f"ViHSD: {len(vi_df):,} mẫu | Jigsaw: {len(en_df):,} mẫu")

# ====================== 2. XỬ LÝ ViHSD ======================
print("🔄 Đang map nhãn ViHSD...")


def vihsd_to_binary(label_id):
    return {
        "insult": 1 if label_id in (1, 2) else 0,
        "hate_speech": 1 if label_id == 2 else 0,
        "threat": 0,
        "harassment": 0,
        "sexual": 0,
        "spam": 0,
    }


vi_binary = vi_df["label_id"].apply(lambda x: pd.Series(vihsd_to_binary(x)))
vi_df = pd.concat([vi_df, vi_binary], axis=1)

vi_df = vi_df.rename(columns={"free_text": "text"})
vi_df["language"] = "vi"

vi_final = vi_df[
    [
        "text",
        "insult",
        "hate_speech",
        "threat",
        "harassment",
        "sexual",
        "spam",
        "language",
    ]
]

# ====================== 3. XỬ LÝ JIGSAW ======================
print("🔄 Đang map nhãn Jigsaw...")

en_df = en_df.rename(columns={"comment_text": "text"})
en_df["language"] = "en"

en_df["insult"] = en_df.get("toxic", 0).astype(int)
en_df["hate_speech"] = (
    en_df.get("severe_toxic", 0).astype(int) | en_df.get("identity_hate", 0).astype(int)
).astype(int)
en_df["threat"] = en_df.get("threat", 0).astype(int)
en_df["harassment"] = 0
en_df["sexual"] = en_df.get("obscene", 0).astype(int)
en_df["spam"] = 0

en_final = en_df[
    [
        "text",
        "insult",
        "hate_speech",
        "threat",
        "harassment",
        "sexual",
        "spam",
        "language",
    ]
]

# ====================== 4. GHÉP + LÀM SẠCH ======================
print("🔗 Đang ghép dataset...")

combined = pd.concat([en_final, vi_final], ignore_index=True)

# Làm sạch text
combined["text"] = combined["text"].fillna("").astype(str)
combined["text"] = combined["text"].str.replace(r"\s+", " ", regex=True).str.strip()
combined = combined[combined["text"] != ""]

# Làm sạch nhãn
label_cols = ["insult", "hate_speech", "threat", "harassment", "sexual", "spam"]
for col in label_cols:
    combined[col] = combined[col].fillna(0).astype(int)

print(f"Sau khi làm sạch: {len(combined):,} mẫu")

# ====================== 5. ÁP DỤNG NORMALIZE (RẤT QUAN TRỌNG) ======================
print("🧹 Đang áp dụng normalize_text...")


def normalize_row(row):
    result = normalize_text(row["text"])
    row["text"] = result["original_normalized"] if isinstance(result, dict) else result
    return row


combined = combined.apply(normalize_row, axis=1)

# ====================== 6. CHIA DATASET ======================
print("✂️ Đang chia train/val/test...")

train_val, test = train_test_split(
    combined, test_size=0.10, random_state=42, stratify=combined["language"]
)
train, val = train_test_split(
    train_val, test_size=0.111, random_state=42, stratify=train_val["language"]
)

print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

# ====================== 7. LƯU FILE ======================
os.makedirs("dataset", exist_ok=True)

train.to_csv("dataset/train.csv", index=False)
val.to_csv("dataset/validation.csv", index=False)
test.to_csv("dataset/test.csv", index=False)

print("\n✅ Dataset đã được xử lý và lưu thành công!")
print(f"   Train: {len(train):,} mẫu")
print(f"   Validation: {len(val):,} mẫu")
print(f"   Test: {len(test):,} mẫu")
