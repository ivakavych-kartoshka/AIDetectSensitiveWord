from pprint import pprint

from src.rules.keyword_matcher import KeywordMatcher

matcher = KeywordMatcher()

examples = [
    "Địt mẹ mày",
    "Tao sẽ giết mày",
    "AK47 là khẩu súng",
    "Porn video",
    "I will kill you",
    "Hello everyone",
    "Bú cu đi",
]

for text in examples:

    print("=" * 70)
    print("Input :", text)

    result = matcher.summary(text)

    pprint(result)
