# toxicity.py
from typing import Dict
from models.factuality_factor import FactualityFactor
from models.toxicity.toxicity_model import ToxicityDetector

# Shared detector instance - initialized once at module level
_detector_instance = None

def get_detector():
    """Get or create the shared ToxicityDetector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = ToxicityDetector()
    return _detector_instance


class Toxicity(FactualityFactor):
    def __init__(self):
        super().__init__(
            "Toxicity",
            "Detect toxic content using Detoxify (unitary/toxic-bert). "
            "Provides scores for: toxicity, severe_toxicity, obscene, threat, insult, and identity_attack."
        )
        # Use shared detector instance
        self.model = get_detector()

    def probability(self, text: str) -> Dict[str, float]:
        """
        Returns probability distribution over toxicity categories:
        - toxicity: overall toxicity score
        - severe_toxicity: severe toxic content
        - obscene: obscene language
        - threat: threatening content
        - insult: insulting content
        - identity_attack: attacks on identity groups
        """
        toxicity_scores, _ = self.model.score(text)
        return toxicity_scores
    
    def categorize(self, text: str) -> str:
        """
        Categorize text into toxicity levels based on multiple toxicity dimensions.
        
        Returns one of: "Friendly", "Neutral", "Rude", "Toxic", "Super_Toxic"
        
        Logic:
        - Super_Toxic: severe_toxicity >= 0.5 OR toxicity >= 0.85
        - Toxic: toxicity >= 0.6 OR threat >= 0.6 OR identity_attack >= 0.6
        - Rude: toxicity >= 0.4 OR insult >= 0.5 OR obscene >= 0.5
        - Neutral: toxicity >= 0.15
        - Friendly: toxicity < 0.15
        """
        scores = self.probability(text)
        
        # Extract individual scores
        toxicity = scores.get("toxicity", 0.0)
        severe_toxicity = scores.get("severe_toxicity", 0.0)
        threat = scores.get("threat", 0.0)
        insult = scores.get("insult", 0.0)
        obscene = scores.get("obscene", 0.0)
        identity_attack = scores.get("identity_attack", 0.0)
        
        # Super Toxic: severe toxicity detected OR very high overall toxicity
        if severe_toxicity >= 0.5 or toxicity >= 0.85:
            return "Super_Toxic"
        
        # Toxic: high toxicity or high threat/identity attack
        if toxicity >= 0.6 or threat >= 0.6 or identity_attack >= 0.6:
            return "Toxic"
        
        # Rude: moderate toxicity or elevated insult/obscene language
        if toxicity >= 0.4 or insult >= 0.5 or obscene >= 0.5:
            return "Rude"
        
        # Neutral: mild toxicity present
        if toxicity >= 0.15:
            return "Neutral"
        
        # Friendly: very low toxicity across all categories
        return "Friendly"


def main():
    toxicity = Toxicity()

    examples = {
        "friendly": "This is such a wonderful and helpful article! Thank you for sharing.",
        "neutral": "The conference will be held on March 15, 2026 at 10:00 AM.",
        "rude": "That's a stupid idea and you should know better.",
        "toxic": "You're an idiot if you believe this garbage!",
        "severely_toxic": "I hope something terrible happens to you, you worthless piece of trash!"
    }

    print("\n=== Toxicity Detection Examples ===\n")
    for level, text in examples.items():
        scores = toxicity.probability(text)
        category = toxicity.categorize(text)
        highest_score_key = max(scores, key=scores.get)

        print(f"{level.upper()}:")
        print(f"  Text: {text}")
        print(f"  Categorized as: {category}")
        print(f"  Highest score: {highest_score_key} ({scores[highest_score_key]:.4f})")
        print(f"  All scores: {scores}")
        print()


if __name__ == "__main__":
    main()