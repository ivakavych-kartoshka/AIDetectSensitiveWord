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
    # Discriminative Learning Rates (ULMFiT-style)
    # ===========================

    # Master switch: when True, the encoder is split into layers
    # and each group gets its own learning-rate multiplier.
    use_discriminative_lr = False

    # LR multiplier for the deepest/embedding layers (slowest update).
    # Layers linearly ramp from dlr_min_mult to 1.0 towards the head.
    dlr_min_mult = 0.1

    # The head (classifier / custom head) always uses the full LR.
    # Unfrozen "bottom" layers share dlr_min_mult; layer groups scale up.
    dlr_freeze_embeddings = False

    # ===========================
    # Two-stage training (freeze-then-finetune)
    # ===========================

    # Master switch: stage 1 freezes the encoder (head only learns),
    # stage 2 unfreezes everything for full fine-tuning.
    use_two_stage_training = False

    # Fraction of total training steps spent in stage 1 (frozen).
    two_stage_freeze_fraction = 0.3

    # ===========================
    # Output
    # ===========================

    output_dir = "models/sensitiveai-v1"

    logging_steps = 100

    eval_steps = 500

    save_steps = 500

    seed = 42
