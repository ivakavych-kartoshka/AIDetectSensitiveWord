import os

from src.config.config import Config
from src.data.dataset_loader import load_dataset, tokenizer
from src.training.trainer import SensitiveTrainer

config = Config()


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


############################################################

print("=" * 60)
print("SensitiveAI Training")
print("=" * 60)

print("\nLoading dataset...")

train_dataset, val_dataset, _ = load_dataset()

print(f"Train samples      : {len(train_dataset)}")
print(f"Validation samples : {len(val_dataset)}")

############################################################

trainer_builder = SensitiveTrainer(
    model_name=config.model_name,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
)

trainer = trainer_builder.get_trainer()

############################################################

checkpoint = find_latest_checkpoint(config.output_dir)

if checkpoint:

    print(f"\nResume from checkpoint:")
    print(checkpoint)

    trainer.train(resume_from_checkpoint=checkpoint)

else:

    print("\nTraining from scratch...")

    trainer.train()

############################################################

print("\nEvaluating best model...")

metrics = trainer.evaluate()

print(metrics)

############################################################

print("\nSaving model...")

trainer.save_model(config.output_dir)

tokenizer.save_pretrained(config.output_dir)

print("\nTraining Finished!")

print(f"Saved model -> {config.output_dir}")
