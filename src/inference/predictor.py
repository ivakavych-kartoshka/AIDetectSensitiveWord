import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from src.config.config import Config
from src.training.metrics import Metrics


class Predictor:

    def __init__(self):

        self.config = Config()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("=" * 60)
        print("Loading AI Model...")
        print(f"Device : {self.device}")

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.output_dir)

        print("Loading model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.output_dir
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

        logits = outputs.logits.cpu().numpy()[0]

        result = Metrics.predict_labels(logits)

        print(f"[AI] Finished ({time.time()-start:.3f}s)")

        return result
