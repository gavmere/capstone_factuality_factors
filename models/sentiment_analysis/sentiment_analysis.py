from models.factuality_factor import FactualityFactor
from typing import Dict
from openai import OpenAI
import numpy as np
import os
import joblib


class SentimentAnalysis(FactualityFactor):
    def __init__(self, API_key: str):
        super().__init__("Sentiment Analysis", "Estimates the sentiment distribution expressed within the article")
        self.API_key = API_key

        # Precompute prototype embeddings lazily (on first probability call) to avoid any network call during construction
        self._prototype_embeddings = None

        self.model = None
        candidate_names = [
            'sentiment_model.joblib',
            'sentiment_model.pkl',
            'sentiment_classifier.joblib',
            'sentiment_classifier.pkl',
            'sentiment_model.sav'
        ]

        this_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(this_dir, '..', '..'))
        search_dirs = [
            os.path.join(repo_root, 'model_training_scripts', 'sentiment_analysis'),
            repo_root,
            this_dir
        ]

        for d in search_dirs:
            for name in candidate_names:
                path = os.path.join(d, name)
                if os.path.exists(path):
                    try:
                        loaded = joblib.load(path)
                        # Ensure classifier has predict_proba
                        if hasattr(loaded, 'predict_proba'):
                            self.model = loaded
                            break
                    except Exception:
                        # Ignore load errors and continue searching
                        pass
            if self.model is not None:
                break

    def _client(self):
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.API_key)

    def _get_embedding(self, text: str) -> np.ndarray:
        client = self._client()
        resp = client.embeddings.create(
            model="google/gemini-embedding-001",
            input=text,
            encoding_format="float"
        )
        return np.array(resp.data[0].embedding, dtype=float)

    def _ensure_prototype_embeddings(self):
        if self._prototype_embeddings is None:
            self._prototype_embeddings = {}
            for label, proto in self.prototypes.items():
                self._prototype_embeddings[label] = self._get_embedding(proto)

    def probability(self, text: str) -> Dict[str, float]:
        if self.model is not None:
            text_emb = self._get_embedding(text)
            try:
                probs = self.model.predict_proba([text_emb])[0]
                if hasattr(self.model, 'classes_'):
                    classes = [str(c) for c in self.model.classes_]
                    prob_map = {str(classes[i]): float(probs[i]) for i in range(len(classes))}
                    out = {}
                    for lbl in ['positive', 'neutral', 'negative']:
                        if lbl in prob_map:
                            out[lbl] = prob_map[lbl]
                    if out:
                        return out
                    return {str(i): float(p) for i, p in enumerate(probs)}
                else:
                    return {str(i): float(p) for i, p in enumerate(probs)}
            except Exception:
                pass

        self._ensure_prototype_embeddings()

        text_emb = self._get_embedding(text)

        sims = {}
        for label, p_emb in self._prototype_embeddings.items():
            denom = (np.linalg.norm(text_emb) * np.linalg.norm(p_emb))
            if denom == 0:
                sim = 0.0
            else:
                sim = float(np.dot(text_emb, p_emb) / denom)
            sims[label] = sim

        sims_array = np.array([sims['positive'], sims['neutral'], sims['negative']], dtype=float)
        logits = sims_array + 1.0
        exps = np.exp(logits)
        probs = exps / np.sum(exps)

        return {'positive': float(probs[0]), 'neutral': float(probs[1]), 'negative': float(probs[2])}


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    API_key = os.getenv("OPENROUTER_API_KEY")
    sa = SentimentAnalysis(API_key)
    examples = [
        "Trump 'retired' a database tracking the most expensive weather disasters. Now it's back — and finding over $100B in losses"
    ]
    for ex in examples:
        print(ex)
        print(sa.probability(ex))

