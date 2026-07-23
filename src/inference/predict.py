from src.inference.predictor import Predictor

predictor = Predictor()

print("=" * 60)
print("SensitiveAI Predictor")
print("=" * 60)

while True:

    text = input("\nInput text (q to quit): ")

    if text.lower() == "q":
        break

    results = predictor.predict(text)

    print("\nPrediction")

    print("-" * 55)

    has_sensitive = False

    for item in results:

        if item["predicted"]:

            has_sensitive = True

        print(
            f"{item['label']:<15}"
            f"{item['confidence']:.4f}"
            f"    {item['predicted']}"
        )

    print("-" * 55)

    print("Sensitive Content" if has_sensitive else "Clean Content")
