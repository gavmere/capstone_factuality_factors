from models.factuality_factor import FactualityFactor
from typing import Dict


class SourceReputation(FactualityFactor):
    def __init__(self):
        super().__init__(
            "Source Reputation",
            "Provides the probability distribution over the source's credibility tiers",
        )

    def probability(self, text: str) -> Dict[str, float]:
        return {self.name: 0.0}


if __name__ == "__main__":
    # Run source reputation specific checks here
    pass

