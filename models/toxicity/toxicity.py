# toxicity.py
from typing import Dict
from models.factuality_factor import FactualityFactor
from models.toxicity.toxicity_model import ToxicityDetector


class Toxicity(FactualityFactor):
    def __init__(self):
        super().__init__(
            "Toxicity",
            "Detect toxic text that is more likely to be disinformed. "
            "Classifies text from friendly to super toxic."
        )
        self.model = ToxicityDetector()

    def probability(self, text: str) -> Dict[str, float]:
        """
        Returns probability distribution over toxicity classes:
        - friendly
        - neutral
        - rude
        - toxic
        - super_toxic
        """
        toxicity_scores, _ = self.model.score(text)
        return toxicity_scores


def main():
    toxicity = Toxicity()

    examples = {
        "friendly": "This is such a wonderful and helpful article! Thank you for sharing.",
        "neutral": "The conference will be held on March 15, 2026 at 10:00 AM.",
        "rude": "That's a stupid idea and you should know better.",
        "toxic": "You're an idiot if you believe this garbage!",
        "super_toxic": "I hope something terrible happens to you, you deserve it!"
    }

    for level, text in examples.items():
        scores = toxicity.probability(text)
        predicted = max(scores, key=scores.get)

        print(f"\n{level.upper()} example:")
        print("Predicted:", predicted)
        print("Scores:", scores)


if __name__ == "__main__":
    main()