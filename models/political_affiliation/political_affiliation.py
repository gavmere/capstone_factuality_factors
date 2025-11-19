from models.factuality_factor import FactualityFactor
from typing import Dict, List
import os
import json
import time

from joblib import load
import numpy as np

from google import genai
from google.genai.errors import APIError, ServerError


class PoliticalAffiliation(FactualityFactor):
    def __init__(self):
        super().__init__(
            "Political Affiliation",
            "Gets the political affiliation of the article"
        )

        # --- Gemini setup ---
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please export it before using the PoliticalAffiliation factor."
            )

        self.gemini_model = "text-embedding-004"
        self.gemini_client = genai.Client(api_key=api_key)

        # --- Load trained classifier (LogisticRegression on Gemini embeddings) ---
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, "political_affiliation_gemini.joblib")
        labels_path = os.path.join(model_dir, "party_labels.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"PoliticalAffiliation model file not found at: {model_path}"
            )
        self.model = load(model_path)

        # class_labels used during training, e.g. ["Democrat", "Republican"]
        if os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.class_labels = data.get("classes", ["Democrat", "Republican"])
        else:
            self.class_labels = ["Democrat", "Republican"]

        # Output keys (lowercase) for the API
        self.output_labels = [lbl.lower() for lbl in self.class_labels]

    # ---------- Internal helpers ----------

    def _embed_texts_with_gemini(
        self,
        texts: List[str],
        batch_size: int = 8,
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
                    break  # success
                except ServerError as e:
                    last_exc = e
                    wait = sleep_base * (attempt + 1)
                    print(
                        f"[PoliticalAffiliation] ServerError on batch "
                        f"{start}:{start+len(batch)}, attempt {attempt+1}/{max_retries} "
                        f"→ retrying in {wait:.1f}s"
                    )
                    time.sleep(wait)
                except APIError as e:
                    # Non-retryable (auth/quota/etc.)
                    print("[PoliticalAffiliation] APIError (non-retryable):", e)
                    raise

            else:
                print(
                    f"[PoliticalAffiliation] FAILED after {max_retries} retries on batch "
                    f"{start}:{start+len(batch)}"
                )
                if last_exc is not None:
                    raise last_exc
                raise RuntimeError("Unknown error while embedding texts with Gemini.")

        return np.array(all_vecs, dtype=np.float32)

    # ---------- Public API ----------

    def probability(self, text: str) -> Dict[str, float]:
        """
        Returns a dict mapping the (lowercased) party labels to probabilities.

        Example:
            {"democrat": 0.72, "republican": 0.28}
        """
        if text is None:
            text = ""
        text = str(text)

        # 1. Embed the input text with Gemini
        X = self._embed_texts_with_gemini([text])  # shape (1, dim)

        # 2. Predict probabilities using the loaded LogisticRegression model
        proba = self.model.predict_proba(X)[0]  # shape (n_classes,)

        # 3. Map to label → prob
        return {
            label: float(p)
            for label, p in zip(self.output_labels, proba)
        }


if __name__ == "__main__":

    from dotenv import load_dotenv
    import os
    load_dotenv()
    gemini_key = os.environ.get("GEMINI_API_KEY")
    factor = PoliticalAffiliation()
    example_text = (
        "Biden calls for expanded climate legislation and student loan relief "
        "in a new speech."
    )
    inner = factor.probability(example_text)
    wrapped = {"political_affiliation": inner}
    print("Inner probs:", inner)
    print("Wrapped:", wrapped)
