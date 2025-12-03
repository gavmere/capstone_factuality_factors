from typing import Dict
class FactualityFactor():
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def get_name(self):
        return self.name

    def get_description(self):
        return self.description

    def probability(self, text: str) -> Dict[str, float]:
        return {self.name: 0.0}