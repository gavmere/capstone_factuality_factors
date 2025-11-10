from models.factuality_factor import FactualityFactor
from typing import Dict


class PoliticalAffiliation(FactualityFactor):
    def __init__(self):
        super().__init__("Political Affiliation", "Gets the political affiliation of the article")

    def probability(self, text: str) -> Dict[str, float]:
        return {self.name: 0.0}



if __name__ == "__main__":
    #Run tests here
    pass