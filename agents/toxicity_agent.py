from agents.base_agent import BaseAgent
from models.toxicity.toxicity_model import ToxicityDetector

class ToxicityAgent(BaseAgent):
    name = "toxicity"
    description = "Detects toxicity severity in text."

    def __init__(self):
        self.detector = ToxicityDetector()

    def run(self, text: str):
        scores, label = self.detector.score(text)

        return {
            "label": label,
            "scores": scores,
            "confidence": max(scores.values())
        }
