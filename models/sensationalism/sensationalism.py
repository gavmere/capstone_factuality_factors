from typing import Dict, List
import os
import re
import time

import numpy as np
import joblib

from google import genai
from google.genai.errors import APIError, ServerError

from models.factuality_factor import FactualityFactor
from dotenv import load_dotenv

load_dotenv()


class Sensationalism(FactualityFactor):
    """
    Sensationalism factuality factor using:
      - Gemini text embeddings ("text-embedding-004")
      - A multinomial LogisticRegression model trained on GoEmotions
        and saved as: models/sensationalism_gemini_goemotions.joblib

    Outputs:
      - A continuous score in [0,1] via self.probability(text)["sensationalism"]
        which is derived from a 0–100 internal rating.
    """

    def __init__(self):
        super().__init__(
            "Sensationalism",
            "Assesses how exaggerated or sensational the article's framing is",
        )

        # --- Gemini setup ---
        api_key = os.environ.get("AI_STUDIO_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please export it before using the Sensationalism factor."
            )

        self.gemini_model = "models/text-embedding-004"
        self.gemini_client = genai.Client(api_key=api_key)

        # --- Load trained classifier ---
        # Make sure this path matches where you saved the model in your notebook
        model_path = os.path.join(
            "models", "sensationalism", "sensationalism_gemini_goemotions.joblib"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Could not find sensationalism model at {model_path}. "
                "Did you run the training notebook and save the model?"
            )

        self.clf = joblib.load(model_path)

        # If you ever want thresholds/weights later, you can also load the meta JSON here.
        # For now we only need clf.predict_proba.

    # ---------- Internal helpers ----------

    def _embed_texts_with_gemini(
        self,
        texts: List[str],
        batch_size: int = 16,
        max_retries: int = 3,
        sleep_base: float = 1.5,
    ) -> np.ndarray:
        """
        Embed a list of texts using Gemini, with simple retry logic for 500 errors.
        Returns: np.ndarray of shape (n_samples, embedding_dim)
        """
        all_vecs: List[List[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            last_exc = None

            for attempt in range(max_retries):
                try:
                    resp = self.gemini_client.models.embed_content(
                        model=self.gemini_model,
                        contents=batch,
                    )
                    for emb in resp.embeddings:
                        all_vecs.append(emb.values)
                    break  # success → exit retry loop
                except ServerError as e:
                    last_exc = e
                    wait = sleep_base * (attempt + 1)
                    # You can replace prints with logging if you prefer
                    print(
                        f"[Sensationalism] ServerError on batch {start}:{start + len(batch)}, "
                        f"attempt {attempt + 1}/{max_retries} → retrying in {wait:.1f}s"
                    )
                    time.sleep(wait)
                except APIError as e:
                    # Non-retryable (auth/quota/etc.)
                    print("[Sensationalism] APIError (non-retryable):", e)
                    raise

            else:
                # All retries exhausted
                print(
                    f"[Sensationalism] FAILED after {max_retries} retries on batch "
                    f"{start}:{start + len(batch)}"
                )
                if last_exc is not None:
                    raise last_exc
                raise RuntimeError("Unknown error while embedding texts with Gemini.")

        return np.array(all_vecs, dtype=np.float32)

    def _sentences(self, text: str):
        """
        Lightweight sentence splitter (same idea as before).
        """
        text = (text or "").strip()
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [s for s in parts if len(s.split()) >= 2]

    def _probs_for_texts(self, texts: List[str]) -> np.ndarray:
        """
        texts -> class probabilities from the trained multinomial logistic regression.
        Returns shape (n_samples, 3), for classes [0,1,2].
        """
        X = self._embed_texts_with_gemini(texts, batch_size=16)
        probs = self.clf.predict_proba(X)  # (n_samples, 3)
        return probs

    def _rating_from_probs(self, probs_row: np.ndarray) -> float:
        """
        Convert class probabilities [p0, p1, p2] to a rating in [0,100]
        using the expected value of the class index scaled to 0–100.

        expected_class = 0*p0 + 1*p1 + 2*p2
        rating = (expected_class / 2) * 100
        """
        probs_row = np.asarray(probs_row, dtype=float)
        expected_class = float(np.dot(probs_row, np.array([0.0, 1.0, 2.0])))
        rating = (expected_class / 2.0) * 100.0
        return float(rating)

    def _score_text(self, text: str, weight_headline: float = 0.6):
        """
        Given a raw text string, treat the first sentence as 'headline' and
        the rest as body sentences. Returns:

          - headline_rating (0..100)
          - body_mean_rating (0..100)
          - final_rating (0..100) : weighted blend

        This mirrors your old behavior, but using the new 3-class model.
        """
        sents = self._sentences(text)
        if not sents:
            return 0.0, 0.0, 0.0

        headline = sents[0]
        body_sents = sents[1:]

        # Embed headline + all body sentences in one batch to minimize API calls
        all_texts = [headline] + body_sents
        probs = self._probs_for_texts(all_texts)

        # First row is headline
        headline_probs = probs[0]
        headline_rating = self._rating_from_probs(headline_probs)

        # Remaining rows are body sentences
        if len(body_sents) > 0:
            body_probs = probs[1:]
            body_ratings = [self._rating_from_probs(pr) for pr in body_probs]
            body_mean = float(np.mean(body_ratings))
        else:
            body_mean = headline_rating

        final_rating = float(
            weight_headline * headline_rating + (1.0 - weight_headline) * body_mean
        )

        return headline_rating, body_mean, final_rating

    # ---------- Public API ----------

    def probability(self, text: str) -> Dict[str, float]:
        """
        Returns:
          {"sensationalism": float} where value is in [0,1].

        Internally, the new model produces a 0–100 rating; we normalize
        by dividing by 100. This keeps the external interface consistent
        with your previous HF-based implementation.
        """
        if text is None:
            text = ""
        text = str(text)

        _, _, final_rating = self._score_text(text)
        # Map 0..100 rating → 0..1 scalar
        final_prob = max(0.0, min(1.0, final_rating / 100.0))

        return {"sensationalism": float(final_prob)}


if __name__ == "__main__":
    # Simple smoke test
    factor = Sensationalism()

    boring = (
        "Weather expected to clear up soon. "
        "Rain that has been ongoing for the past week should go away soon."
    )
    spicy = (
        "You WON'T BELIEVE what scientists found! "
        "A new paper reports a modest 2% increase in signal."
    )

    print("Boring:", factor.probability(boring))
    print("Spicy:", factor.probability(spicy))
