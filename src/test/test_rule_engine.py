from pprint import pprint

from src.rules.rule_engine import RuleEngine

engine = RuleEngine()

examples = [
    "Địt mẹ mày",
    "Tao sẽ giết mày",
    "Porn video",
    "Bú cu đi",
    "Liên hệ 0987654321",
    "Email abc@gmail.com",
    "Visit https://pornhub.com",
    "I will kill you",
    "Hello everyone",
]

for text in examples:

    print("=" * 80)

    print(text)

    print(engine.analyze(text))
