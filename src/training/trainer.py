from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

from src.config.config import Config
from src.training.metrics import Metrics


class SensitiveTrainer:

    def __init__(
        self,
        model_name,
        tokenizer,
        train_dataset,
        val_dataset,
    ):

        self.config = Config()

        self.tokenizer = tokenizer

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        print("\nLoading model...")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.config.num_labels,
            problem_type="multi_label_classification",
        )

    ########################################################
    # Training Arguments
    ########################################################

    def build_training_args(self):

        return TrainingArguments(
            # ======================================================
            # Output
            # ======================================================
            output_dir=self.config.output_dir,
            overwrite_output_dir=False,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            # ======================================================
            # Evaluation
            # ======================================================
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            # ======================================================
            # Logging
            # ======================================================
            logging_strategy="steps",
            logging_steps=self.config.logging_steps,
            logging_dir="logs",
            report_to="tensorboard",
            # ======================================================
            # Optimizer
            # ======================================================
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="cosine",
            # ======================================================
            # Batch
            # ======================================================
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            # ======================================================
            # Epoch
            # ======================================================
            num_train_epochs=self.config.num_train_epochs,
            # ======================================================
            # Hardware
            # ======================================================
            fp16=self.config.fp16,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            # ======================================================
            # Misc
            # ======================================================
            seed=self.config.seed,
            remove_unused_columns=False,
        )

    ########################################################
    # Trainer
    ########################################################

    def get_trainer(self):

        trainer = Trainer(
            model=self.model,
            args=self.build_training_args(),
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=Metrics.calculate,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        return trainer
