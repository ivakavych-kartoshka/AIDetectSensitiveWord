from data.dataset_loader import load_dataset

train_dataset, val_dataset = load_dataset()

print("=" * 60)
print(train_dataset)
print("=" * 60)

print(train_dataset[0])
