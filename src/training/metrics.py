import json
from pathlib import Path

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

# ==========================================================
# LABELS
# ==========================================================

LABELS = [
    "insult",
    "hate_speech",
    "threat",
    "harassment",
    "sexual",
    "spam",
]

# ==========================================================
# METRICS
# ==========================================================


class Metrics:

    DEFAULT_THRESHOLD = 0.5

    # ------------------------------------------------------
    # Sigmoid
    # ------------------------------------------------------

    @staticmethod
    def sigmoid(logits):

        logits = np.asarray(logits)

        return 1.0 / (1.0 + np.exp(-logits))

    # ------------------------------------------------------
    # Probability
    # ------------------------------------------------------

    @classmethod
    def predict_probability(cls, logits):

        return cls.sigmoid(logits)

    # ------------------------------------------------------
    # Binary Prediction
    # ------------------------------------------------------

    @classmethod
    def predict_binary(
        cls,
        logits,
        threshold=None,
        per_class_thresholds=None,
    ):

        probabilities = cls.sigmoid(logits)

        if per_class_thresholds is not None:

            thresholds = np.array(
                [per_class_thresholds.get(l, cls.DEFAULT_THRESHOLD) for l in LABELS]
            )

            return (probabilities >= thresholds).astype(int)

        if threshold is None:
            threshold = cls.DEFAULT_THRESHOLD

        return (probabilities >= threshold).astype(int)

    # ------------------------------------------------------
    # Label Prediction
    # ------------------------------------------------------

    @classmethod
    def predict_labels(
        cls,
        logits,
        threshold=None,
    ):

        if threshold is None:
            threshold = cls.DEFAULT_THRESHOLD

        probabilities = cls.sigmoid(logits)

        results = []

        for label, score in zip(LABELS, probabilities):

            results.append(
                {
                    "label": label,
                    "confidence": round(float(score), 4),
                    "predicted": bool(score >= threshold),
                }
            )

        return results

    # ------------------------------------------------------
    # Highest Prediction
    # ------------------------------------------------------

    @classmethod
    def highest_prediction(cls, logits):

        probabilities = cls.sigmoid(logits)

        index = int(np.argmax(probabilities))

        return {
            "label": LABELS[index],
            "confidence": float(probabilities[index]),
        }

    # ------------------------------------------------------
    # HuggingFace Trainer Metrics
    # ------------------------------------------------------

    @classmethod
    def calculate(cls, eval_pred):

        logits, labels = eval_pred

        probabilities = cls.sigmoid(logits)

        predictions = (probabilities >= cls.DEFAULT_THRESHOLD).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="micro",
            zero_division=0,
        )

        accuracy = accuracy_score(
            labels,
            predictions,
        )

        try:

            roc_auc = roc_auc_score(
                labels,
                probabilities,
                average="micro",
            )

        except Exception:

            roc_auc = 0.0

        try:

            pr_auc = average_precision_score(
                labels,
                probabilities,
                average="micro",
            )

        except Exception:

            pr_auc = 0.0

        return {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
        }

    # ------------------------------------------------------
    # HuggingFace Trainer Metrics (per-class thresholds)
    # ------------------------------------------------------

    @classmethod
    def calculate_with_thresholds(cls, eval_pred, per_class_thresholds=None):

        logits, labels = eval_pred

        probabilities = cls.sigmoid(logits)

        if per_class_thresholds is not None:

            thresholds = np.array(
                [per_class_thresholds.get(l, cls.DEFAULT_THRESHOLD) for l in LABELS]
            )

        else:

            thresholds = cls.DEFAULT_THRESHOLD

        predictions = (probabilities >= thresholds).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="micro",
            zero_division=0,
        )

        accuracy = accuracy_score(labels, predictions)

        try:
            roc_auc = roc_auc_score(labels, probabilities, average="micro")
        except Exception:
            roc_auc = 0.0

        try:
            pr_auc = average_precision_score(labels, probabilities, average="micro")
        except Exception:
            pr_auc = 0.0

        return {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
        }

    # ------------------------------------------------------
    # Classification Report
    # ------------------------------------------------------

    @classmethod
    def classification_report(
        cls,
        logits,
        labels,
        threshold=None,
        per_class_thresholds=None,
    ):

        predictions = cls.predict_binary(
            logits,
            threshold,
            per_class_thresholds=per_class_thresholds,
        )

        return classification_report(
            labels,
            predictions,
            target_names=LABELS,
            zero_division=0,
        )

    # ------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------

    @classmethod
    def confusion_matrices(
        cls,
        logits,
        labels,
        threshold=None,
        per_class_thresholds=None,
    ):

        predictions = cls.predict_binary(
            logits,
            threshold,
            per_class_thresholds=per_class_thresholds,
        )

        matrices = {}

        for i, label in enumerate(LABELS):

            matrices[label] = confusion_matrix(
                labels[:, i],
                predictions[:, i],
            )

        return matrices

    # ------------------------------------------------------
    # Save Metrics
    # ------------------------------------------------------

    @staticmethod
    def save_metrics(
        metrics,
        output_file,
    ):

        output_file = Path(output_file)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4,
                ensure_ascii=False,
            )

    # ------------------------------------------------------
    # Pretty Print Prediction
    # ------------------------------------------------------

    @staticmethod
    def print_prediction(results):

        print("\nPrediction")
        print("=" * 55)

        for item in results:

            status = "YES" if item["predicted"] else "NO"

            print(f"{item['label']:<18}" f"{item['confidence']:.4f}" f"    {status}")

        print("=" * 55)

    # ------------------------------------------------------
    # Severity
    # ------------------------------------------------------

    @staticmethod
    def severity(confidence):

        if confidence >= 0.95:
            return "HIGH"

        if confidence >= 0.70:
            return "MEDIUM"

        return "LOW"

    # ------------------------------------------------------
    # Decision
    # ------------------------------------------------------

    @staticmethod
    def decision(confidence):

        if confidence >= 0.95:
            return "BLOCK"

        if confidence >= 0.70:
            return "REVIEW"

        return "ALLOW"
