from models.factuality_factor import FactualityFactor
from typing import Dict, Any, Optional
import os
import json
import joblib

class SourceReputation(FactualityFactor):
    def __init__(self, model_path: Optional[str] = None):
        super().__init__(
            "Source Reputation",
            "Provides the probability distribution over the source's credibility tiers",
        )

        if model_path is None:
            base_dir = os.path.dirname(__file__)
            candidates = [
                os.path.join(base_dir, "source_reputation.joblib"),
                os.path.join(base_dir, "source_reputation.pkl"),
                os.path.join(base_dir, "source_reputation.ipynb]"),
            ]
        else:
            candidates = [model_path]

        self.model = None
        self.model_labels = None
        for p in candidates:
            if os.path.exists(p):
                if joblib is not None:
                    try:
                        self.model = joblib.load(p)
                        # If the model exposes classes_ (sklearn classifiers) capture them
                        self.model_labels = getattr(self.model, "classes_", None)
                    except Exception:
                        # ignore load errors and continue to fallback
                        self.model = None
                break

        # these are fabricated since we dont have access to these values
        self.source_metadata = {
            "dwayne-bohac": {"accuracy_score": 0.4, "awards": 0, "editorial_transparency": "low"},
            "scott-surovell": {"accuracy_score": 0.7, "awards": 1, "editorial_transparency": "medium"},
            "barack-obama": {"accuracy_score": 0.9, "awards": 2, "editorial_transparency": "high"},
            "blog-posting": {"accuracy_score": 0.2, "awards": 0, "editorial_transparency": "unknown"},
            "cnn": {"accuracy_score": 0.85, "awards": 2, "editorial_transparency": "high"},
            "bbc": {"accuracy_score": 0.88, "awards": 3, "editorial_transparency": "high"},
            "nytimes": {"accuracy_score": 0.9, "awards": 4, "editorial_transparency": "high"},
        }

        self.transparency_map = {"low": 0, "medium": 1, "high": 2, "unknown": -1, None: -1}

    def _heuristic_probs_from_metadata(self, originator: str) -> Dict[str, float]:
        key = originator.lower() if isinstance(originator, str) else originator
        meta = self.source_metadata.get(key)
        if not meta:
            # unknown source
            return {"unknown": 1.0}

        acc = meta.get("accuracy_score", -1)
        if acc is None or acc < 0:
            return {"unknown": 1.0}

        high_score = max(0.0, (acc - 0.6) / 0.4)  # 0 when acc<=0.6, 1 when acc>=1.0
        low_score = max(0.0, (0.6 - acc) / 0.6)   # 1 when acc=0.0, 0 when acc>=0.6
        medium_score = max(0.0, 1.0 - abs(acc - 0.5) * 2.0)

        total = high_score + medium_score + low_score
        if total <= 0:
            return {"unknown": 1.0}

        return {
            "high": round(high_score / total, 4),
            "medium": round(medium_score / total, 4),
            "low": round(low_score / total, 4),
        }

    def probability(self, text: Any) -> Dict[str, float]:
        originator = None
        if isinstance(text, dict):
            originator = text.get("source") or text.get("originator")
        elif isinstance(text, str):
            if len(text.split()) <= 3:
                originator = text

        # If we have a loaded model and the model expects numeric features, try to use it.
        if self.model is not None:
            try:
                # If input is a dict and contains numeric feature fields, use them.
                if isinstance(text, dict):
                    # Try to construct the feature vector in the same order used in training
                    feat = []
                    for f in ("source_accuracy", "source_awards", "source_transparency"):
                        feat.append(text.get(f, -1))
                    X = [feat]
                    # sklearn classifiers: predict_proba
                    proba = None
                    if hasattr(self.model, "predict_proba"):
                        probs = self.model.predict_proba(X)[0]
                        labels = None
                        if self.model_labels is not None:
                            labels = [str(l) for l in self.model_labels]
                        else:
                            labels = [str(i) for i in range(len(probs))]
                        return {labels[i]: float(probs[i]) for i in range(len(probs))}

            except Exception:
                pass

        if originator:
            return self._heuristic_probs_from_metadata(originator)

        return {"unknown": 1.0}

if __name__ == "__main__":
    # Demonstration: predict on the example article
    new_article = {
        "title": "Trump 'retired' a database tracking the most expensive weather disasters. Now it's back — and finding over $100B in losses",
        "source": "CNN",
        "statement": "The Billion-Dollar Weather and Climate Disasters Database, which the Trump administration'retired' in May, has relaunched outside of the government using the same methodology...",
    }

    factor = SourceReputation()
    print("Source Reputation prediction for article (source=CNN):")
    print(factor.probability(new_article))
