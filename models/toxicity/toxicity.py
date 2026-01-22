# toxicity.py
from models.factuality_factor import FactualityFactor
from typing import Dict
from models.toxicity.toxicity_model import ToxicityDetector

class Toxicity(FactualityFactor):
    def __init__(self):
        super().__init__(
            "Toxicity",
            "Detect toxic text that is more likely to be disinformed. Classifies text from friendly to super toxic."
        )
        self.model = ToxicityDetector()

    def probability(self, text: str) -> Dict[str, float]:
        result = self.model.score(text)
        return result["probabilities"]

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
        print(f"\n{level.title()} example:")
        print(toxicity.probability(text))

if __name__ == "__main__":
    main()
