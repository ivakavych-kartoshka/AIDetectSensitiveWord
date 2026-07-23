import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer

import numpy as np
from src.config.config import Config

config = Config()

# =====================================================
# LABELS
# =====================================================

LABEL_COLUMNS = [
    "insult",
    "hate_speech",
    "threat",
    "harassment",
    "sexual",
    "spam",
]

# =====================================================
# TOKENIZER
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# =====================================================
# LOAD CSV
# =====================================================


def load_dataframe(csv_path):

    df = pd.read_csv(csv_path)

    print(f"\nLoaded : {csv_path}")
    print(f"Samples: {len(df)}")

    return df


# =====================================================
# CREATE LABEL VECTOR
# =====================================================


import numpy as np


def create_label_vector(df):

    df["labels"] = df[LABEL_COLUMNS].astype(np.float32).values.tolist()

    return df


# =====================================================
# TOKENIZE
# =====================================================


def tokenize(batch):

    return tokenizer(
        batch["normalized_text"],
        truncation=True,
        padding="max_length",
        max_length=config.max_length,
    )


# =====================================================
# CONVERT TO HF DATASET
# =====================================================


def dataframe_to_dataset(df):

    dataset = Dataset.from_pandas(
        df[
            [
                "normalized_text",
                "labels",
            ]
        ]
    )

    dataset = dataset.map(
        tokenize,
        batched=True,
    )

    dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "labels",
        ],
    )

    return dataset


# =====================================================
# LOAD ALL DATASET
# =====================================================


def load_dataset():

    # ==========================
    # Load CSV
    # ==========================

    train_df = load_dataframe(config.train_file)
    val_df = load_dataframe(config.val_file)
    test_df = load_dataframe(config.test_file)

    # ==========================
    # Create label vectors
    # ==========================

    train_df = create_label_vector(train_df)
    val_df = create_label_vector(val_df)
    test_df = create_label_vector(test_df)

    # ==========================
    # Convert HuggingFace Dataset
    # ==========================

    train_dataset = dataframe_to_dataset(train_df)
    val_dataset = dataframe_to_dataset(val_df)
    test_dataset = dataframe_to_dataset(test_df)

    return train_dataset, val_dataset, test_dataset


def load_train_dataset():

    train_df = create_label_vector(load_dataframe(config.train_file))
    val_df = create_label_vector(load_dataframe(config.val_file))

    return (
        dataframe_to_dataset(train_df),
        dataframe_to_dataset(val_df),
    )


def load_test_dataset():

    test_df = create_label_vector(load_dataframe(config.test_file))

    return dataframe_to_dataset(test_df)
