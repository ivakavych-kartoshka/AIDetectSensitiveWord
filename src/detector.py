import time

from src.rules.rule_engine import RuleEngine
from src.report import ModerationReport
from src.inference.predictor import Predictor


class SensitiveDetector:

    def __init__(self):

        self.rule_engine = RuleEngine()

        # Load AI model
        self.ml_model = Predictor()

        self.report = ModerationReport()

    ############################################################

    def analyze(self, text):

        start_time = time.time()

        ########################################################
        # Rule Engine
        ########################################################

        rule_result = self.rule_engine.analyze(text)

        print("\n========== RULE ENGINE ==========")
        print(f"Input      : {text}")
        print(f"Decision   : {rule_result['decision']}")
        print(f"Risk       : {rule_result['risk']}")
        print(f"Score      : {rule_result['score']}")
        print(f"Categories : {rule_result['categories']}")

        if len(rule_result["matches"]) == 0:
            print("Matches    : None")
        else:
            print("Matches:")
            for match in rule_result["matches"]:
                print(
                    f"   - {match['keyword']} "
                    f"({match['category']}) "
                    f"confidence={match['confidence']:.2f}"
                )

        print("=================================\n")

        ########################################################
        # AI Model
        ########################################################

        print(f"[AI] Predicting -> {text}")

        ml_result = self.ml_model.predict(text)

        print("\n=========== AI MODEL ===========")

        predicted_labels = []

        ml_score = 0.0

        for item in ml_result:

            print(
                f"{item['label']:<20}"
                f"confidence={item['confidence']:.4f}   "
                f"predicted={item['predicted']}"
            )

            if item["predicted"]:
                predicted_labels.append(item["label"])
                ml_score = max(ml_score, item["confidence"])

        print("================================\n")

        ########################################################
        # Merge Rule + AI
        ########################################################

        rule_score = rule_result["score"]

        final_score = max(rule_score, ml_score)

        decision = rule_result["decision"]

        risk = rule_result["risk"]

        ########################################################
        # AI Override
        ########################################################

        if ml_score >= 0.95:

            decision = "block"

            risk = "critical"

        elif ml_score >= 0.80:

            if decision == "allow":
                decision = "review"

            if risk in ["safe", "low"]:
                risk = "high"

        ########################################################
        # Merge Categories
        ########################################################

        categories = sorted(set(rule_result["categories"] + predicted_labels))

        ########################################################
        # Sensitive
        ########################################################

        has_sensitive = rule_result["has_sensitive"] or len(predicted_labels) > 0

        ########################################################
        # Build Result
        ########################################################

        result = {
            "input": text,
            "decision": decision,
            "risk": risk,
            "score": round(final_score, 4),
            "has_sensitive": has_sensitive,
            "categories": categories,
            "matches": rule_result["matches"],
            "metadata": rule_result["metadata"],
            "obfuscation": rule_result["obfuscation"],
            "rule_result": rule_result,
            "ml_result": ml_result,
        }

        ########################################################
        # Report
        ########################################################

        result["report"] = self.report.generate(result)

        ########################################################
        # Final Log
        ########################################################

        elapsed = time.time() - start_time

        print("=========== FINAL RESULT ===========")
        print(f"Decision      : {decision}")
        print(f"Risk          : {risk}")
        print(f"Final Score   : {final_score:.4f}")
        print(f"Categories    : {categories}")
        print(f"Sensitive     : {has_sensitive}")
        print(f"Inference Time: {elapsed:.3f}s")
        print("====================================\n")

        return result

    ############################################################

    def predict(self, text):

        return self.analyze(text)["decision"]

    ############################################################

    def score(self, text):

        return self.analyze(text)["score"]

    ############################################################

    def categories(self, text):

        return self.analyze(text)["categories"]

    ############################################################

    def explain(self, text):

        result = self.analyze(text)

        return {
            "decision": result["decision"],
            "risk": result["risk"],
            "score": result["score"],
            "matched_keywords": sorted(
                {m["keyword"] for m in result["matches"] if "keyword" in m}
            ),
            "matched_categories": result["categories"],
            "recommendation": result["report"]["recommendation"],
            "summary": result["report"]["summary"],
            "ml_result": result["ml_result"],
        }
