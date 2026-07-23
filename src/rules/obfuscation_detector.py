import re

from src.data.normalize import normalize_text


class ObfuscationDetector:

    def __init__(self):

        self.patterns = [
            (
                "split_letters",
                re.compile(
                    r"(?:[a-zA-ZÀ-ỹ][\W_]+){2,}[a-zA-ZÀ-ỹ]",
                    re.IGNORECASE,
                ),
            ),
            (
                "leet",
                re.compile(
                    r"[a-zA-Z]*[@$0134578!]+[a-zA-Z0-9]*",
                    re.IGNORECASE,
                ),
            ),
            (
                "mixed_symbol",
                re.compile(
                    r"[a-zA-Z]+(?:[_.*-]+[a-zA-Z]+)+",
                    re.IGNORECASE,
                ),
            ),
            (
                "unicode_font",
                re.compile(r"[\U0001D400-\U0001D7FF]"),
            ),
            (
                "zero_width",
                re.compile(r"[\u200B-\u200D\uFEFF]"),
            ),
            (
                "repeated_char",
                re.compile(
                    r"(.)\1{3,}",
                    re.IGNORECASE,
                ),
            ),
        ]

    ###########################################################

    def detect(self, text):

        result = []

        normalized = normalize_text(text)

        metadata = normalized.get("metadata", {})

        ########################################################
        # metadata từ normalize
        ########################################################

        for key in [
            "had_homoglyph",
            "had_leetspeak",
            "had_zero_width",
            "had_repeated_chars",
            "had_abbreviation",
        ]:

            if metadata.get(key):

                result.append(
                    {
                        "type": key,
                        "confidence": 0.95,
                    }
                )

        ########################################################
        # regex detect
        ########################################################

        for name, pattern in self.patterns:

            for m in pattern.finditer(text):

                value = m.group()

                if len(value.strip()) < 3:

                    continue

                result.append(
                    {
                        "type": name,
                        "keyword": value,
                        "position": [
                            m.start(),
                            m.end(),
                        ],
                        "confidence": 0.90,
                    }
                )

        ########################################################
        # remove duplicate
        ########################################################

        unique = {}

        for item in result:

            key = (
                item.get("type"),
                item.get("keyword", ""),
            )

            unique[key] = item

        return list(unique.values())

    ###########################################################

    def has_obfuscation(self, text):

        return len(self.detect(text)) > 0

    ###########################################################

    def summary(self, text):

        matches = self.detect(text)

        return {
            "has_obfuscation": len(matches) > 0,
            "count": len(matches),
            "matches": matches,
        }
