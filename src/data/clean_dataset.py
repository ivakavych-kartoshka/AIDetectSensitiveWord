import os
import pandas as pd

from src.data.normalize import normalize_text

# ==========================================================
# CONFIG
# ==========================================================

TRAIN_FILE = "dataset/train.csv"
VAL_FILE = "dataset/validation.csv"
TEST_FILE = "dataset/test.csv"

OUTPUT_DIR = "dataset"

MAX_TEXT_LENGTH = 5000

LABEL_COLUMNS = [
    "insult",
    "hate_speech",
    "threat",
    "harassment",
    "sexual",
    "spam",
]

# ==========================================================
# CLEAN FUNCTION
# ==========================================================


def clean_dataframe(df: pd.DataFrame, dataset_name="dataset"):

    print("\n" + "=" * 60)
    print(f"Cleaning {dataset_name}")
    print("=" * 60)

    original_rows = len(df)

    # ------------------------------------------------------
    # Fill NULL
    # ------------------------------------------------------

    print("Step 1: Fill NULL text")

    df["text"] = df["text"].fillna("").astype(str)

    # ------------------------------------------------------
    # Remove Empty
    # ------------------------------------------------------

    print("Step 2: Remove empty text")

    before = len(df)

    df = df[df["text"].str.strip() != ""]

    print(f"Removed : {before-len(df)} rows")

    # ------------------------------------------------------
    # Normalize Text
    # ------------------------------------------------------

    print("Step 3: Normalize text")

    df["normalized_text"] = (
        df["text"].apply(normalize_text).apply(lambda x: x["combined"])
    )

    # ------------------------------------------------------
    # Remove Duplicate
    # ------------------------------------------------------

    print("Step 4: Remove duplicate")

    before = len(df)

    df = df.drop_duplicates(subset=["normalized_text"])

    print(f"Removed : {before-len(df)} rows")

    # ------------------------------------------------------
    # Remove Long Text
    # ------------------------------------------------------

    print("Step 5: Remove very long text")

    df["length"] = df["normalized_text"].str.len()

    before = len(df)

    df = df[df["length"] <= MAX_TEXT_LENGTH]

    print(f"Removed : {before-len(df)} rows")

    # ------------------------------------------------------
    # Reset Index
    # ------------------------------------------------------

    df = df.reset_index(drop=True)

    print("\nFinished Cleaning")

    print(f"Original rows : {original_rows}")
    print(f"Remaining rows: {len(df)}")

    return df


# ==========================================================
# REPORT
# ==========================================================


def report_dataset(df):

    print("\n" + "=" * 60)
    print("Dataset Report")
    print("=" * 60)

    print(f"Total samples : {len(df)}")

    print("\nLabel Distribution")

    for label in LABEL_COLUMNS:

        if label in df.columns:

            print(f"{label:15}: {int(df[label].sum())}")

    print("\nLanguage Distribution")

    if "language" in df.columns:

        print(df["language"].value_counts())

    print("\nText Length")

    print(df["length"].describe())

    print("\nMulti-label Distribution")

    df["label_count"] = df[LABEL_COLUMNS].sum(axis=1)

    print(df["label_count"].value_counts().sort_index())

    df.drop(columns=["label_count"], inplace=True)


# ==========================================================
# SHOW EXAMPLES
# ==========================================================


def show_examples(df, n=5):

    print("\n" + "=" * 60)
    print("Normalization Examples")
    print("=" * 60)

    samples = df.sample(min(n, len(df)), random_state=42)

    for _, row in samples.iterrows():

        print("-" * 60)

        print("Original")

        print(row["text"])

        print()

        print("Normalized")

        print(row["normalized_text"])

        print()


# ==========================================================
# SAVE
# ==========================================================


def save_dataset(df, output_file):

    df = df.drop(columns=["length"])

    df.to_csv(output_file, index=False)

    print(f"\nSaved -> {output_file}")


# ==========================================================
# PROCESS
# ==========================================================


def process(input_file, output_file):

    print("\n")
    print("#" * 70)
    print(input_file)
    print("#" * 70)

    df = pd.read_csv(input_file)

    df = clean_dataframe(df, input_file)

    report_dataset(df)

    show_examples(df)

    save_dataset(df, output_file)


# ==========================================================
# MAIN
# ==========================================================


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    process(
        TRAIN_FILE,
        os.path.join(OUTPUT_DIR, "train_clean.csv"),
    )

    process(
        VAL_FILE,
        os.path.join(OUTPUT_DIR, "validation_clean.csv"),
    )

    process(
        TEST_FILE,
        os.path.join(OUTPUT_DIR, "test_clean.csv"),
    )

    print("\n" + "=" * 60)
    print("Dataset Cleaning Completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
