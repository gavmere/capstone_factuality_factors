from agents.base_agent import BaseAgent
from models.political_affiliation.political_affiliation import PoliticalAffiliation


class PoliticalAffiliationAgent(BaseAgent):
    name = "political_affiliation"
    description = "Predicts the political affiliation implied by the article."

    def __init__(self):
        self.model = PoliticalAffiliation()

    def run(self, text: str):
        """
        Returns per-party probabilities and the most likely label.
        """
        probs = self.model.probability(text)

        if not probs:
            return {
                "label": None,
                "scores": {},
                "confidence": 0.0
            }

        label = max(probs, key=probs.get)
        confidence = probs[label]

        return {
            "label": label,
            "scores": probs,
            "confidence": confidence
        }
