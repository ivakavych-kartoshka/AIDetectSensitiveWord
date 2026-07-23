from dataclasses import dataclass
import re

from src.utils.resource_manager import ResourceManager
from src.data.normalize import normalize_text

############################################################
# Match Object
############################################################


@dataclass
class KeywordMatch:

    category: str
    keyword: str
    start: int
    end: int
    confidence: float = 1.0


############################################################
# Keyword Matcher
############################################################


class KeywordMatcher:

    def __init__(self):

        self.resource = ResourceManager()

        self.whitelist = {w.lower().strip() for w in self.resource.words("whitelist")}

        self.category_keywords = {}

        self.build()

    ########################################################

    def build(self):

        ignore = {
            "homoglyph",
            "leetspeak",
            "abbreviation",
            "regex_patterns",
            "category_weight",
            "whitelist",
        }

        for category in self.resource.all_categories():

            if category in ignore:
                continue

            words = {
                w.lower().strip()
                for w in self.resource.words(category)
                if isinstance(w, str)
            }

            self.category_keywords[category] = sorted(
                words,
                key=len,
                reverse=True,
            )

    ########################################################

    def _confidence(self, keyword):

        l = len(keyword)

        if l >= 12:
            return 1.0

        if l >= 8:
            return 0.95

        if l >= 5:
            return 0.90

        return 0.85

    ########################################################

    def _search(
        self,
        text,
        category,
        keywords,
        occupied,
    ):

        matches = []

        for keyword in keywords:

            if keyword in self.whitelist:
                continue

            pattern = r"\b" + re.escape(keyword) + r"\b"

            for m in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):

                start = m.start()
                end = m.end()

                overlap = False

                for s, e in occupied:

                    if start < e and end > s:
                        overlap = True
                        break

                if overlap:
                    continue

                occupied.append((start, end))

                matches.append(
                    KeywordMatch(
                        category=category,
                        keyword=keyword,
                        start=start,
                        end=end,
                        confidence=self._confidence(keyword),
                    )
                )

        return matches

    ########################################################

    def _find_on_text(self, text):

        matches = []

        occupied = []

        for category, keywords in self.category_keywords.items():

            matches.extend(
                self._search(
                    text,
                    category,
                    keywords,
                    occupied,
                )
            )

        return matches

    ########################################################

    def find(self, text):

        normalized = normalize_text(text)

        original = normalized["original_normalized"]

        no_diacritics = normalized["no_diacritics"]

        ####################################################
        # Search original first
        ####################################################

        matches = self._find_on_text(original)

        ####################################################
        # Fallback
        ####################################################

        if not matches:

            matches = self._find_on_text(no_diacritics)

        ####################################################
        # Remove duplicate
        ####################################################

        unique = {}

        for m in matches:

            key = (
                m.category,
                m.keyword,
                m.start,
                m.end,
            )

            unique[key] = m

        return sorted(
            unique.values(),
            key=lambda x: (
                x.start,
                -(x.end - x.start),
                x.category,
            ),
        )

    ########################################################

    def categories(self, text):

        return sorted({m.category for m in self.find(text)})

    ########################################################

    def keywords(self, text):

        return sorted({m.keyword for m in self.find(text)})

    ########################################################

    def has_sensitive(self, text):

        return bool(self.find(text))

    ########################################################

    def summary(self, text):

        matches = self.find(text)

        return {
            "has_sensitive": bool(matches),
            "categories": sorted({m.category for m in matches}),
            "keywords": sorted({m.keyword for m in matches}),
            "matches": [
                {
                    "category": m.category,
                    "keyword": m.keyword,
                    "position": [
                        m.start,
                        m.end,
                    ],
                    "confidence": round(
                        m.confidence,
                        2,
                    ),
                }
                for m in matches
            ],
        }
