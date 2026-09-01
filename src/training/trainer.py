from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

from transformers.trainer_pt_utils import get_parameter_names

import torch
import torch.nn as nn

from src.config.config import Config
from src.training.metrics import Metrics


class DiscriminativeTrainer(Trainer):
    """Trainer that assigns a different learning rate to each group of
    parameters (ULMFiT-style "discriminative learning rates").

    The encoder's lower layers get a smaller LR (they already encode
    general knowledge and should only be fine-tuned gently), while the
    layers closer to the output and the classification head get the full
    learning rate.
    """

    def __init__(
        self,
        *args,
        use_discriminative_lr=False,
        dlr_min_mult=0.1,
        dlr_freeze_embeddings=False,
        **kwargs,
    ):

        self.use_discriminative_lr = use_discriminative_lr
        self.dlr_min_mult = dlr_min_mult
        self.dlr_freeze_embeddings = dlr_freeze_embeddings

        super().__init__(*args, **kwargs)

    def create_optimizer(self):

        if self.optimizer is not None:

            return self.optimizer

        base_lr = float(self.args.learning_rate)

        # If DLR is disabled, fall back to the default single-group AdamW.
        if not self.use_discriminative_lr:

            return super().create_optimizer()

        decay_mult = 1.0 if self.args.weight_decay > 0.0 else 0.0

        decay_parameters = get_parameter_names(
            self.model,
            [
                nn.LayerNorm,
                nn.MultiheadAttention,
            ],
        )

        decay_parameters = [
            name for name in decay_parameters if "bias" not in name
        ]

        optimizer_grouped_parameters = self._build_dlr_groups(
            base_lr,
            decay_mult,
            decay_parameters,
        )

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args
        )

        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters,
            **optimizer_kwargs,
        )

        return self.optimizer

    def _build_dlr_groups(self, base_lr, decay_mult, decay_parameters):
        """Split the model parameters into LR groups.

        Returns a list of optimizer param dicts:
          [{"params": [...], "lr": <lr>}, ...]
        """

        min_mult = self.dlr_min_mult

        # Layers below this one are frozen (no gradient).
        freeze_embeddings = self.dlr_freeze_embeddings

        num_hidden = self._num_hidden_layers()

        groups = {}

        for name, param in self.model.named_parameters():

            # Note: frozen parameters (e.g. Two-stage stage 1) are still
            # kept in their group. Their lr is 0 naturally during
            # freezing because requires_grad=False, and they start
            # updating correctly once unfrozen (stage 2), all within
            # the SAME optimizer instance.
            decay = name in decay_parameters

            group_key, mult = self._classify_layer(
                name,
                min_mult,
                num_hidden,
                freeze_embeddings,
            )

            # Each (decay, layer-group) becomes one param group.
            key = (group_key, decay)

            if key not in groups:

                groups[key] = {
                    "params": [],
                    "lr": base_lr * mult,
                    "weight_decay": self.args.weight_decay * decay_mult
                    if decay
                    else 0.0,
                }

            groups[key]["params"].append(param)

        return list(groups.values())

    def _num_hidden_layers(self):

        if hasattr(self.model, "bert"):

            return getattr(
                self.model.bert.config, "num_hidden_layers", 12
            )

        return getattr(self.model.config, "num_hidden_layers", 12)

    def _classify_layer(self, name, min_mult, num_hidden, freeze):
        """Return (group_key, lr_multiplier) for a parameter name."""

        # --- Head / classifier: always full LR ---
        if any(k in name for k in ("head.", "classifier.", "score.", "lm_head.")):

            return ("head", 1.0)

        # --- Embeddings: smallest LR (or frozen) ---
        if "embeddings" in name or "rel_embeddings" in name:

            if freeze:

                return ("embeddings", 0.0)  # frozen -> mult 0

            return ("embeddings", min_mult)

        # --- Transformer encoder layers: linear ramp from min -> 1.0 ---
        if "encoder.layer." in name or ".layer." in name:

            idx = self._extract_layer_index(name)

            if idx is not None and num_hidden > 1:

                # Linearly interpolate between min_mult and 1.0.
                mult = min_mult + (1.0 - min_mult) * (idx / (num_hidden - 1))

                return (f"layer_{idx}", mult)

        # --- Any remaining (pooler, custom extras) : full LR ---
        return ("other", 1.0)

    def _extract_layer_index(self, name):

        import re

        match = re.search(r"layer\.(\d+)\.", name)

        if match:

            return int(match.group(1))

        return None


class SensitiveTrainer:

    def __init__(
        self,
        model_name,
        tokenizer,
        train_dataset,
        val_dataset,
        output_dir=None,
        use_custom_head=False,
        use_discriminative_lr=None,
        use_two_stage_training=None,
    ):

        self.config = Config()

        if output_dir is not None:
            self.config.output_dir = output_dir

        # Command-line overrides (None -> fall back to config default).
        if use_discriminative_lr is not None:
            self.config.use_discriminative_lr = use_discriminative_lr

        if use_two_stage_training is not None:
            self.config.use_two_stage_training = use_two_stage_training

        self.use_custom_head = use_custom_head

        self.tokenizer = tokenizer

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        print("\nLoading model...")

        if use_custom_head:

            from src.models.custom_model import SensitiveCustomModel

            self.model = SensitiveCustomModel.from_pretrained(
                model_name,
                num_labels=self.config.num_labels,
            )

        else:

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
            report_to=[],
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

        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=4),
        ]

        trainer_cls = Trainer
        trainer_kwargs = {}

        use_dlr = self.config.use_discriminative_lr

        # Two-stage callback: when enabled, it freezes the encoder at
        # the start (stage 1) and unfreezes it after a given step
        # fraction (stage 2).
        if self.config.use_two_stage_training:

            from src.training.two_stage_callback import (
                TwoStageCallback,
            )

            callbacks.append(
                TwoStageCallback(
                    freeze_first_fraction=self.config.two_stage_freeze_fraction,
                    unfreeze_at_step=None,
                )
            )

        if use_dlr:

            trainer_cls = DiscriminativeTrainer

            trainer_kwargs = {
                "use_discriminative_lr": True,
                "dlr_min_mult": self.config.dlr_min_mult,
                "dlr_freeze_embeddings": self.config.dlr_freeze_embeddings,
            }

        trainer = trainer_cls(
            model=self.model,
            args=self.build_training_args(),
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=Metrics.calculate,
            callbacks=callbacks,
            **trainer_kwargs,
        )

        return trainer
