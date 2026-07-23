from datetime import datetime


class ModerationReport:

    def __init__(self):
        pass

    ############################################################

    def generate(self, result):

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "decision": result["decision"],
            "risk": result["risk"],
            "score": result["score"],
            "has_sensitive": result["has_sensitive"],
            "summary": self.summary(result),
            "details": self.details(result),
            "recommendation": self.recommend(result),
        }

        return report

    ############################################################

    def summary(self, result):

        return {
            "category_count": len(result["categories"]),
            "match_count": len(result["matches"]),
            "categories": result["categories"],
        }

    ############################################################

    def details(self, result):

        data = []

        for m in result["matches"]:

            data.append(
                {
                    "category": m.get("category"),
                    "keyword": m.get("keyword"),
                    "confidence": m.get("confidence", 1),
                    "position": m.get("position"),
                }
            )

        return data

    ############################################################

    def recommend(self, result):

        if result["decision"] == "block":

            return "Reject content"

        elif result["decision"] == "review":

            return "Send to moderator"

        return "Allow content"
