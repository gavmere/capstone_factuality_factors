from models.factuality_factor import FactualityFactor
from typing import Dict
from models.factuality_factor import FactualityFactor
from typing import Dict
from models.sentiment_analysis.vader_model import sentiment_scores

class Sentiment(FactualityFactor):
    """
    Sentiment factuality factor using VADER (from `sentiment.ipynb`).
    Returns probabilities for `negative`, `neutral`, and `positive` based
    on VADER component scores.
    """
    def __init__(self):
        super().__init__(
            "Sentiment",
            "Probability represents the emotional tone of the text: positive, neutral, or negative."
        )

    def probability(self, text: str) -> Dict[str, float]:
        sentiment_dict = sentiment_scores(text)
        neg = float(sentiment_dict.get("neg", 0.0))
        neu = float(sentiment_dict.get("neu", 0.0))
        pos = float(sentiment_dict.get("pos", 0.0))
        total = neg + neu + pos
        if total == 0:
            return {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
        return {
            "negative": round(neg / total, 6),
            "neutral": round(neu / total, 6),
            "positive": round(pos / total, 6),
        }

if __name__ == "__main__":
    # Quick local demo similar to the notebook examples
    sentiment = Sentiment()
    print("\n1st Statement:")
    print(sentiment.probability("Geeks For Geeks is an excellent platform for CSE students."))

    print("\n2nd Statement:")
    print(sentiment.probability("Shweta played well in the match as usual."))

    print("\n3rd Statement:")
    print(sentiment.probability("I am feeling sad today."))

    #python -m models.sentiment_analysis.sentiment_analysis