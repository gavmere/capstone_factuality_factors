from models.factuality_factor import FactualityFactor
from typing import Dict


class SentimentAnalysis(FactualityFactor):
    def __init__(self):
        super().__init__(
            "Sentiment Analysis",
            "Estimates the sentiment distribution expressed within the article",
        )

    def probability(self, text: str) -> Dict[str, float]:
        return {self.name: 0.0}


if __name__ == "__main__":
    # Run sentiment analysis specific checks here
    pass

