from dataclasses import dataclass


@dataclass
class Config:

    # ===========================
    # Dataset
    # ===========================

    train_file = "dataset/clean/train_clean.csv"
    val_file = "dataset/clean/validation_clean.csv"
    test_file = "dataset/clean/test_clean.csv"

    # ===========================
    # Model
    # ===========================

    model_name = "microsoft/mdeberta-v3-base"

    num_labels = 6

    max_length = 256

    # ===========================
    # Training
    # ===========================

    batch_size = 8

    gradient_accumulation_steps = 4

    learning_rate = 2e-5

    num_train_epochs = 3

    weight_decay = 0.01

    warmup_ratio = 0.1

    fp16 = True

    # ===========================
    # Output
    # ===========================

    output_dir = "models/sensitiveai-v1"

    logging_steps = 100

    eval_steps = 500

    save_steps = 500

    seed = 42
