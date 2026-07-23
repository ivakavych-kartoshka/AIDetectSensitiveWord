from pprint import pprint

from src.rules.obfuscation_detector import ObfuscationDetector

detector = ObfuscationDetector()

examples = [
    "địt mẹ",
    "d!t me",
    "d.i.t me",
    "d i t me",
    "𝐝𝐢𝐭 𝐦𝐞",
    "𝗱𝗶𝘁 𝗺𝗲",
    "l0n",
    "l.o.n",
    "hello world",
]

for text in examples:

    print("=" * 60)

    print(text)

    pprint(detector.summary(text))
