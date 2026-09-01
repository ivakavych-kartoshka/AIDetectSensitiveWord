import argparse
import os

from src.config.config import Config
from src.data.dataset_loader import load_dataset_by_language
from src.training.trainer import SensitiveTrainer
from transformers import AutoTokenizer

config = Config()

BASE_MODEL = "models/sensitiveai-v1"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


def find_latest_checkpoint(output_dir):

    if not os.path.exists(output_dir):
        return None

    checkpoints = []

    for folder in os.listdir(output_dir):

        if folder.startswith("checkpoint-"):

            checkpoints.append(os.path.join(output_dir, folder))

    if len(checkpoints) == 0:
        return None

    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))

    return checkpoints[-1]


def finetune_language(language):

    import torch

    print("\n" + "=" * 60)
    print(f"Fine-tune from combined model for language : {language}")
    print("=" * 60)

    if torch.cuda.is_available():

        print(
            f"\nDevice : CUDA ({torch.cuda.get_device_name(0)})"
        )

    else:

        print("\nDevice : CPU")

    output_dir = os.path.join(
        "models",
        f"sensitiveai-v1-{language}",
    )

    print("\nLoading dataset...")

    train_dataset, val_dataset, _ = load_dataset_by_language([language])

    print(f"Train samples      : {len(train_dataset)}")
    print(f"Validation samples : {len(val_dataset)}")

    trainer_builder = SensitiveTrainer(
        model_name=BASE_MODEL,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
    )

    trainer = trainer_builder.get_trainer()

    checkpoint = find_latest_checkpoint(output_dir)

    if checkpoint:

        print(f"\nResume from checkpoint:")
        print(checkpoint)

        trainer.train(resume_from_checkpoint=checkpoint)

    else:

        print("\nFine-tuning from scratch (base = combined model)...")

        trainer.train()

    print("\nEvaluating best model...")

    metrics = trainer.evaluate()

    print(metrics)

    print("\nSaving model...")

    trainer.save_model(output_dir)

    tokenizer.save_pretrained(output_dir)

    print(f"\nSaved model -> {output_dir}")


def main():

    parser = argparse.ArgumentParser(
        description="Fine-tune combined model by language"
    )

    parser.add_argument(
        "--language",
        choices=["all", "en", "vi"],
        default="all",
        help="Language to fine-tune: 'all' (both), 'en', or 'vi'. Default: all",
    )

    args = parser.parse_args()

    if args.language == "all":
        languages = ["en", "vi"]
    else:
        languages = [args.language]

    for language in languages:

        finetune_language(language)

    print("\n" + "=" * 60)
    print("Fine-tune By Language Finished")
    print("=" * 60)


if __name__ == "__main__":

    main()
