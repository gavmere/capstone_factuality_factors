from agents.base_agent import BaseAgent
from models.sentiment_analysis.sentiment_analysis import Sentiment
from models.toxicity.toxicity_model import ToxicityDetector

class SentimentAgent(BaseAgent):
    name = "sentiment"
    description = "Detects sentiment in text."

    def __init__(self):
        self.detector = ToxicityDetector()

    def run(self, text: str):
        scores, label = self.detector.score(text)

        return {
            "label": label,
            "scores": scores,
            "confidence": max(scores.values())
        }
