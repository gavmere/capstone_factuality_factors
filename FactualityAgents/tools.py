from __future__ import annotations

import os
from typing import Dict

from dotenv import load_dotenv

from models.clickbait.clickbait import Clickbait
from models.headline_body_relation.headline_body_relation import HeadlineBodyRelation
from models.political_affiliation.political_affiliation import PoliticalAffiliation
from models.sensationalism.sensationalism import Sensationalism
from models.sentiment_analysis.sentiment_analysis import Sentiment
from models.toxicity.toxicity import Toxicity

load_dotenv()


def _missing_key_error(key_name: str) -> Dict[str, str]:
    return {"error": f"{key_name} environment variable is not set"}


def _coerce_floats(values: Dict[str, object]) -> Dict[str, float]:
    return {key: float(value) for key, value in values.items()}


def clickbait_score(headline: str) -> Dict[str, float]:
    """
    Predict clickbait likelihood for a headline using the trained XGBoost model.
    Returns class probabilities and a confidence value.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return _missing_key_error("OPENROUTER_API_KEY")

    model = Clickbait(api_key)
    scores = _coerce_floats(model.probability(headline))
    confidence = max(scores.values()) if scores else 0.0
    return {"scores": scores, "confidence": float(confidence)}


def headline_body_relation_score(headline: str, body: str) -> Dict[str, float]:
    """
    Compute semantic similarity between headline and body text using Gemini embeddings.
    Returns a similarity score in [0, 1].
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return _missing_key_error("OPENROUTER_API_KEY")
    model = HeadlineBodyRelation(api_key)
    result = model.probability(headline, body)
    similarity = float(result.get("similarity", 0.0))
    return {"similarity": similarity, "confidence": similarity}


def political_affiliation_score(text: str) -> Dict[str, float]:
    """
    Estimate political affiliation probabilities (e.g., democrat, republican)
    using the Gemini embedding + logistic regression classifier.
    """
    try:
        model = PoliticalAffiliation()
        return _coerce_floats(model.probability(text))
    except Exception as exc:  # pragma: no cover - defensive for runtime issues
        return {"error": str(exc)}


def sensationalism_score(text: str) -> Dict[str, float]:
    """
    Score how sensational the text is. Returns the normalized sensationalism score.
    """
    model = Sensationalism()
    probs = _coerce_floats(model.probability(text))
    score = float(probs.get("sensationalism", 0.0))
    return {"score": score, "confidence": score}


def sentiment_score(text: str) -> Dict[str, float]:
    """
    Detect sentiment using the VADER-based sentiment factor.
    Returns probabilities for negative, neutral, and positive plus the top label.
    """
    model = Sentiment()
    probs = _coerce_floats(model.probability(text))
    label = max(probs, key=probs.get)
    confidence = probs[label]
    return {"label": label, "scores": probs, "confidence": float(confidence)}


def toxicity_score(text: str) -> Dict[str, float]:
    """
    Detect toxicity using the RoBERTa multi-class classifier.
    Returns the predicted label, per-class scores, and confidence.
    """
    detector = Toxicity()
    scores = _coerce_floats(detector.probability(text))
    label = max(scores, key=scores.get) if scores else "unknown"
    confidence = scores.get(label, 0.0) if scores else 0.0
    return {"label": label, "scores": scores, "confidence": float(confidence)}
