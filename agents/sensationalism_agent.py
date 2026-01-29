from agents.base_agent import BaseAgent
from models.sensationalism.sensationalism import Sensationalism


class SensationalismAgent(BaseAgent):
    name = "sensationalism"
    description = "Assesses how exaggerated or sensational the article's framing is."

    def __init__(self):
        self.model = Sensationalism()

    def run(self, text: str):
        """
        Returns a normalized sensationalism score in [0, 1].
        """
        probs = self.model.probability(text)

        score = probs.get("sensationalism", 0.0)

        return {
            "score": score,
            "confidence": score
        }
