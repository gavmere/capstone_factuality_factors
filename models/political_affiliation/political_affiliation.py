from models.factuality_factor import FactualityFactor
from typing import Dict
import os
import json
from joblib import load
import numpy as np

class PoliticalAffiliation(FactualityFactor):
    def __init__(self):
        super().__init__("Political Affiliation", "Gets the political affiliation of the article")
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, "political_affiliation.joblib")
        labels_path = os.path.join(model_dir, "party_labels.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"PoliticalAffiliation model file not found at: {model_path}"
            )
        self.model = load(model_path)

        if os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.class_labels = data.get("classes", ["Democrat", "Republican"])
        else:
            self.class_labels = ["Democrat", "Republican"]

        self.output_labels = [lbl.lower() for lbl in self.class_labels]

    def probability(self, text: str) -> Dict[str, float]:
        if text is None:
            text = ""
        text = str(text)

        proba = self.model.predict_proba([text])[0]

        return {
            label: float(p)
            for label, p in zip(self.output_labels, proba)
        }



if __name__ == "__main__":
    factor = PoliticalAffiliation()
    example_text = (
        "Biden calls for expanded climate legislation and student loan relief "
        "in a new speech."
    )
    inner = factor.probability(example_text)
    wrapped = {"political_affiliation": inner}
    print("Inner probs:", inner)
    print("Wrapped:", wrapped)