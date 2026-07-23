from pprint import pprint

from src.rules.regex_engine import RegexEngine

engine = RegexEngine()

examples = [
    "Liên hệ 0987654321",
    "Mail: abc@gmail.com",
    "Website https://google.com",
    "IP: 192.168.1.1",
    "CCCD 079203001234",
    "Bank 123456789012",
    "Credit card 4111111111111111",
    "Nothing here",
]

for text in examples:

    print("=" * 70)

    print(text)

    pprint(engine.summary(text))
