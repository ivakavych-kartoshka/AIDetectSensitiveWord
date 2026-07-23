from pprint import pprint

from src.detector import SensitiveDetector

detector = SensitiveDetector()

examples = [
    "Địt mẹ mày",
    "Tao sẽ giết mày",
    "Bú cu đi",
    "Porn video",
    "Hello everyone",
    "d!t m3 m@y",
    "Visit https://pornhub.com",
    "Email abc@gmail.com",
    "AK47",
]

for text in examples:

    print("=" * 80)

    pprint(detector.analyze(text))
