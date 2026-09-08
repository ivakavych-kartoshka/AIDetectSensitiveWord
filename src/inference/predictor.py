import os
import time

import torch
from transformers import (
    AutoTokenizer,
)

from src.config.config import Config
from src.models.custom_model import SensitiveCustomModel
from src.training.metrics import Metrics


class Predictor:

    # Best-performing model: Vietnamese custom model trained with
    # discriminative learning rates + two-stage (freeze-then-finetune).
    MODEL_PATH = "models/sensitiveai-vi-custom-dlr2stage"

    def __init__(self, model_path=None):

        self.config = Config()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Allow overriding the model path (e.g. for testing another
        # checkpoint) without editing this file.
        self.model_path = model_path or self.MODEL_PATH

        print("=" * 60)
        print("Loading AI Model...")
        print(f"Model  : {self.model_path}")
        print(f"Device : {self.device}")

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"Model directory not found: {self.model_path}"
            )

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # The best model uses the custom multi-label head architecture.
        # It must be loaded with SensitiveCustomModel (not the default
        # AutoModelForSequenceClassification) because its state dict uses
        # custom keys (bert.* encoder + head.*).
        loader_kwargs = {}

        if self.device.type == "cuda":
            print("Loading model (FP16)...")
            loader_kwargs["torch_dtype"] = torch.float16
        else:
            print("Loading model (FP32)...")

        self.model = SensitiveCustomModel.from_pretrained(
            self.model_path,
            num_labels=self.config.num_labels,
            **loader_kwargs,
        )

        self.model.to(self.device)
        self.model.eval()

        print("AI Model Loaded Successfully")
        print("=" * 60)

    ########################################################

    def predict(self, text):

        print(f"[AI] Predicting -> {text}")

        start = time.time()

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():

            outputs = self.model(**inputs)

        # The custom model returns a dict {"loss", "logits"}.
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs.logits

        logits = logits.cpu().numpy()[0]

        result = Metrics.predict_labels(logits)

        print(f"[AI] Finished ({time.time()-start:.3f}s)")

        return result
