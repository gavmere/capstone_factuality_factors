from models.factuality_factor import FactualityFactor
from typing import Dict


class Sensationalism(FactualityFactor):
    def __init__(self):
        super().__init__(
            "Sensationalism",
            "Assesses how exaggerated or sensational the article's framing is",
        )

    def probability(self, text: str) -> Dict[str, float]:
        return {self.name: 0.0}


if __name__ == "__main__":
    # Run sensationalism specific checks here
    pass

