from transformers import AutoTokenizer, AutoModel
import os

# Đường dẫn lưu model
MODEL_NAME = "microsoft/mdeberta-v3-base"
SAVE_PATH = "models/" + MODEL_NAME

# Tạo folder nếu chưa có
os.makedirs("models", exist_ok=True)

print(f"Downloading model: {MODEL_NAME}")
print(f"Saving to: {SAVE_PATH}")

# Tải và lưu model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=SAVE_PATH)
model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=SAVE_PATH)

print("Download model successfully!")