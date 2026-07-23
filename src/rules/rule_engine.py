import json
from pathlib import Path

from src.data.normalize import normalize_text
from src.rules.keyword_matcher import KeywordMatcher
from src.rules.regex_engine import RegexEngine
from src.rules.obfuscation_detector import ObfuscationDetector


class RuleEngine:

    def __init__(self):

        self.keyword = KeywordMatcher()

        self.regex = RegexEngine()

        self.obfuscation = ObfuscationDetector()

        self.category_weight = self._load_weight()

    ############################################################

    def _load_weight(self):

        file = (
            Path(__file__).resolve().parents[2] / "resources" / "category_weight.json"
        )

        if file.exists():

            with open(file, encoding="utf8") as f:

                return json.load(f)

        return {}

    ############################################################

    def analyze(self, text):

        normalized = normalize_text(text)

        keyword_result = self.keyword.summary(text)

        regex_result = self.regex.summary(text)

        obfuscation_result = self.obfuscation.summary(text)

        ########################################################
        # Score chỉ dựa trên keyword category
        ########################################################

        score = self.calculate_score(
            keyword_result,
            obfuscation_result,
        )

        ########################################################

        risk = self.risk_level(score)

        decision = self.decision(score)

        ########################################################

        categories = sorted(keyword_result["categories"])

        ########################################################

        matches = keyword_result["matches"] + regex_result["matches"]

        ########################################################

        return {
            "input": text,
            "normalized": normalized["combined"],
            # chỉ keyword mới tính sensitive
            "has_sensitive": len(keyword_result["matches"]) > 0,
            "score": round(score, 3),
            "risk": risk,
            "decision": decision,
            "categories": categories,
            "matches": matches,
            "keyword_result": keyword_result,
            "regex_result": regex_result,
            "obfuscation": obfuscation_result,
            "metadata": normalized["metadata"],
        }

    ############################################################

    def calculate_score(
        self,
        keyword_result,
        obfuscation_result,
    ):

        if not keyword_result["matches"]:

            return 0.0

        ####################################################
        # Lấy category nặng nhất
        ####################################################

        score = 0.0

        for category in keyword_result["categories"]:

            score = max(
                score,
                self.category_weight.get(
                    category,
                    0.50,
                ),
            )

        ####################################################
        # Nếu có obfuscation thì tăng nhẹ
        ####################################################

        if obfuscation_result["has_obfuscation"]:

            score += 0.05

        ####################################################
        # Nếu nhiều category thì tăng nhẹ
        ####################################################

        extra = max(
            0,
            len(keyword_result["categories"]) - 1,
        )

        score += min(extra * 0.05, 0.15)

        ####################################################

        return min(score, 1.0)

    ############################################################

    def risk_level(self, score):

        if score >= 0.90:

            return "critical"

        elif score >= 0.75:

            return "high"

        elif score >= 0.50:

            return "medium"

        elif score > 0:

            return "low"

        else:

            return "safe"

    ############################################################

    def decision(self, score):

        if score >= 0.90:

            return "block"

        elif score >= 0.60:

            return "review"

        elif score > 0:

            return "flag"

        else:

            return "allow"

    ############################################################

    def is_sensitive(self, text):

        return self.analyze(text)["has_sensitive"]
