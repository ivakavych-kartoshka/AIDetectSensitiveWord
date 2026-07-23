import json
import re
from pathlib import Path

from src.data.normalize import normalize_text


class RegexEngine:

    def __init__(self):

        root = Path(__file__).resolve().parents[2]

        file = root / "resources" / "regex_patterns.json"

        with open(file, encoding="utf8") as f:
            self.patterns = json.load(f)

        self.compiled = {}

        self._compile()

    ##########################################################

    def _compile(self):

        for category, patterns in self.patterns.items():

            self.compiled[category] = [
                re.compile(
                    p,
                    re.IGNORECASE | re.UNICODE,
                )
                for p in patterns
            ]

    ##########################################################

    def _confidence(self, category):

        table = {
            "credit_card": 1.0,
            "bank_account": 1.0,
            "bitcoin_wallet": 1.0,
            "ethereum_wallet": 1.0,
            "email": 0.60,
            "phone": 0.60,
            "url": 0.60,
            "ipv4": 0.60,
            "ipv6": 0.60,
            "telegram": 0.70,
            "discord": 0.70,
            "facebook": 0.70,
            "zalo": 0.70,
            "youtube": 0.70,
            "instagram": 0.70,
            "tiktok": 0.70,
        }

        return table.get(category, 0.75)

    ##########################################################

    def find(self, text):

        normalized = normalize_text(text)

        text = normalized["original_normalized"]

        matches = []

        unique = set()

        for category, patterns in self.compiled.items():

            for pattern in patterns:

                for m in pattern.finditer(text):

                    key = (category, m.start(), m.end(), m.group().lower())

                    if key in unique:

                        continue

                    unique.add(key)

                    matches.append(
                        {
                            "category": category,
                            "keyword": m.group(),
                            "position": [
                                m.start(),
                                m.end(),
                            ],
                            "confidence": self._confidence(category),
                        }
                    )

        matches.sort(key=lambda x: (x["position"][0], -len(x["keyword"])))

        return matches

    ##########################################################

    def categories(self, text):

        return sorted({x["category"] for x in self.find(text)})

    ##########################################################

    def has_sensitive(self, text):

        sensitive = {
            "credit_card",
            "bank_account",
            "bitcoin_wallet",
            "ethereum_wallet",
        }

        return any(m["category"] in sensitive for m in self.find(text))

    ##########################################################

    def summary(self, text):

        matches = self.find(text)

        return {
            "has_sensitive": self.has_sensitive(text),
            "categories": sorted({m["category"] for m in matches}),
            "matches": matches,
            "metadata_count": sum(
                1
                for m in matches
                if m["category"] in {"email", "phone", "url", "ipv4", "ipv6"}
            ),
            "security_count": sum(
                1
                for m in matches
                if m["category"]
                in {"credit_card", "bank_account", "bitcoin_wallet", "ethereum_wallet"}
            ),
        }
