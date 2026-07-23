from pprint import pprint

from src.api import predict

examples = ["Địt mẹ mày", "Porn video", "Hello", "I will kill you", "Visit pornhub.com"]

for x in examples:

    print("=" * 80)

    pprint(predict(x))
