import pandas as pd

# Đọc dataset
df = pd.read_csv("dataset/train.csv")

print("=" * 60)
print("5 dòng đầu tiên:")
print(df.head())

print("\n" + "=" * 60)
print("Tên các cột:")
print(df.columns.tolist())

print("\n" + "=" * 60)
print("Kích thước dataset:")
print(df.shape)

print("\n" + "=" * 60)
print("Kiểu dữ liệu:")
print(df.dtypes)

print("\n" + "=" * 60)
print("Số lượng giá trị NULL:")
print(df.isnull().sum())

labels = [
    "insult",
    "hate_speech",
    "threat",
    "harassment",
    "sexual",
    "spam",
]

print("\n" + "=" * 60)
print("Phân bố từng nhãn")


for label in labels:
    print(f"{label:15}: {df[label].sum()}")
    print("\n" + "=" * 60)
print("Duplicate rows")
print(df.duplicated().sum())


print("\n" + "=" * 60)
print("Empty text")
print(df["text"].isna().sum())


df["length"] = df["text"].fillna("").astype(str).apply(len)
print("\n" + "=" * 60)
print(df["length"].describe())


print("\n" + "=" * 60)
print("Unique values")

for label in [
    "insult",
    "hate_speech",
    "threat",
    "harassment",
    "sexual",
    "spam",
]:
    print(f"{label}: {df[label].unique()}")

    print("\n" + "=" * 60)

print(df.describe())
