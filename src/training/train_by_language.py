import argparse
import os

from src.config.config import Config
from src.data.dataset_loader import load_dataset_by_language, tokenizer
from src.training.trainer import SensitiveTrainer

config = Config()


def build_trainer(
    language,
    output_dir,
    use_custom_head=False,
    model_name=None,
    use_discriminative_lr=None,
    use_two_stage_training=None,
    use_class_weights=False,
    use_focal_loss=False,
    focal_loss_gamma=None,
    per_class_thresholds=None,
    class_weights=None,
):

    import torch

    print("\n" + "=" * 60)
    print(f"Training model for language : {language}")
    print("=" * 60)

    if torch.cuda.is_available():

        print(
            f"\nDevice : CUDA ({torch.cuda.get_device_name(0)})"
        )

    else:

        print("\nDevice : CPU")

    if model_name is None:

        model_name = config.model_name

    print(f"\nStarting from : {model_name}")

    print("\nLoading dataset...")

    train_dataset, val_dataset, _ = load_dataset_by_language([language])

    print(f"Train samples      : {len(train_dataset)}")
    print(f"Validation samples : {len(val_dataset)}")

    trainer_builder = SensitiveTrainer(
        model_name=model_name,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        use_custom_head=use_custom_head,
        use_discriminative_lr=use_discriminative_lr,
        use_two_stage_training=use_two_stage_training,
        use_class_weights=use_class_weights,
        use_focal_loss=use_focal_loss,
        focal_loss_gamma=focal_loss_gamma,
        per_class_thresholds=per_class_thresholds,
        class_weights=class_weights,
    )

    return trainer_builder


def run_training(trainer_builder, output_dir):

    trainer = trainer_builder.get_trainer()

    checkpoint = find_latest_checkpoint(output_dir)

    if checkpoint:

        print(f"\nResume from checkpoint:")
        print(checkpoint)

        trainer.train(resume_from_checkpoint=checkpoint)

    else:

        print("\nTraining from scratch...")

        trainer.train()

    print("\nEvaluating best model...")

    metrics = trainer.evaluate()

    print(metrics)

    print("\nSaving model...")

    trainer.save_model(output_dir)

    tokenizer.save_pretrained(output_dir)

    print(f"\nSaved model -> {output_dir}")


def train_language(
    language,
    use_custom_head=False,
    start_model=None,
    tag=None,
    use_discriminative_lr=None,
    use_two_stage_training=None,
    use_class_weights=False,
    use_focal_loss=False,
    focal_loss_gamma=None,
    per_class_thresholds=None,
    class_weights=None,
):

    suffix = ""

    if use_custom_head:

        suffix += "-custom"

    if tag:

        suffix += f"-{tag}"

    output_dir = os.path.join(
        "models",
        f"sensitiveai-{language}{suffix if suffix else ''}",
    )

    trainer_builder = build_trainer(
        language,
        output_dir,
        use_custom_head=use_custom_head,
        model_name=start_model,
        use_discriminative_lr=use_discriminative_lr,
        use_two_stage_training=use_two_stage_training,
        use_class_weights=use_class_weights,
        use_focal_loss=use_focal_loss,
        focal_loss_gamma=focal_loss_gamma,
        per_class_thresholds=per_class_thresholds,
        class_weights=class_weights,
    )

    run_training(trainer_builder, output_dir)


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


def main():

    from src.config.config import Config as DefaultConfig

    parser = argparse.ArgumentParser(
        description="Train SensitiveAI model by language"
    )

    parser.add_argument(
        "--language",
        choices=["all", "en", "vi"],
        default="all",
        help="Language to train: 'all' (both), 'en', or 'vi'. Default: all",
    )

    parser.add_argument(
        "--custom-head",
        action="store_true",
        help="Use the custom multi-label head architecture instead of the default classification head",
    )

    parser.add_argument(
        "--start-model",
        default=None,
        help="Path or name of the checkpoint to initialize weights from (default: base mDeBERTa)",
    )

    parser.add_argument(
        "--tag",
        default=None,
        help="Extra suffix for the output model directory",
    )

    parser.add_argument(
        "--discriminative-lr",
        action="store_true",
        help="Use ULMFiT-style discriminative learning rates (layer-wise LR)",
    )

    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Use two-stage training: freeze encoder first, then fine-tune all",
    )

    parser.add_argument(
        "--class-weights",
        action="store_true",
        help="Use class-specific pos_weight in loss to handle imbalanced labels",
    )

    parser.add_argument(
        "--class-weights-json",
        default=None,
        help="JSON dict of per-label class weights, e.g. '{\"insult\":1.2,\"hate_speech\":2.0}'. Requires --class-weights.",
    )

    parser.add_argument(
        "--focal-loss",
        action="store_true",
        help="Use Focal Loss instead of standard BCEWithLogitsLoss",
    )

    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=None,
        help="Gamma parameter for Focal Loss (default: 2.0)",
    )

    parser.add_argument(
        "--per-class-thresholds",
        action="store_true",
        help="Use per-class thresholds from config during evaluation (hate_speech=0.35, others=0.5)",
    )

    parser.add_argument(
        "--train-file",
        default=None,
        help="Optional custom training CSV (e.g. augmented dataset)",
    )

    args = parser.parse_args()

    # Override training file if provided
    if args.train_file:
        config.train_file = args.train_file
        print(f"\nTrain file    : {config.train_file}")

    # Build per_class_thresholds dict if requested
    per_class_thresholds = None

    if args.per_class_thresholds:

        per_class_thresholds = DefaultConfig().per_class_thresholds

        print(f"\nPer-class thresholds: {per_class_thresholds}")

    # Build label-specific class weights if requested
    class_weights = None

    if args.class_weights_json:

        import json

        try:

            class_weights = json.loads(args.class_weights_json)

        except json.JSONDecodeError:

            print("Invalid --class-weights-json, must be a JSON dict")
            return

        print(f"\nLabel-specific class weights: {class_weights}")

    if args.language == "all":
        languages = ["en", "vi"]
    else:
        languages = [args.language]

    for language in languages:

        train_language(
            language,
            use_custom_head=args.custom_head,
            start_model=args.start_model,
            tag=args.tag,
            use_discriminative_lr=args.discriminative_lr,
            use_two_stage_training=args.two_stage,
            use_class_weights=args.class_weights,
            use_focal_loss=args.focal_loss,
            focal_loss_gamma=args.focal_gamma,
            per_class_thresholds=per_class_thresholds,
            class_weights=class_weights,
        )

    print("\n" + "=" * 60)
    print("Training By Language Finished")
    print("=" * 60)


if __name__ == "__main__":

    main()
