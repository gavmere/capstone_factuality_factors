from __future__ import annotations

import os
import json
from typing import Dict, Any

from dotenv import load_dotenv

from generative_models.llm import generate
from models.clickbait.clickbait import Clickbait
from models.headline_body_relation.headline_body_relation import HeadlineBodyRelation
from models.political_affiliation.political_affiliation import PoliticalAffiliation
from models.sensationalism.sensationalism import Sensationalism
from models.sentiment_analysis.sentiment_analysis import Sentiment
from models.toxicity.toxicity import Toxicity
from .prompts import FINAL_VERACITY_PROMPT

load_dotenv()


def _missing_key_error(key_name: str) -> Dict[str, str]:
    return {"error": f"{key_name} environment variable is not set"}


def _coerce_floats(values: Dict[str, Any]) -> Dict[str, float]:
    return {
        key: float(value)
        for key, value in values.items()
        if isinstance(value, (int, float, str))
    }


def clickbait_predictive_score(headline: str) -> Dict[str, Any]:
    """
    Predict clickbait likelihood for a headline using the trained XGBoost model.
    Returns class probabilities (0: not clickbait, 1: clickbait).
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return _missing_key_error("OPENROUTER_API_KEY")

    model = Clickbait(api_key)
    scores = _coerce_floats(model.probability(headline))
    # Map "1" to clickbait score
    score = scores.get("1", scores.get(1, 0.0))
    return {"predictive_score": float(score), "all_scores": scores}


def headline_body_relation_predictive_score(headline: str, body: str) -> Dict[str, Any]:
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
    return {"predictive_score": similarity}


def political_affiliation_predictive_score(text: str) -> Dict[str, Any]:
    """
    Estimate political affiliation probabilities using Gemini embedding + logistic regression.
    Returns probabilities for democrat and republican.
    """
    try:
        model = PoliticalAffiliation()
        scores = _coerce_floats(model.probability(text))
        label = max(scores, key=scores.get)
        return {"predictive_label": label, "all_scores": scores}
    except Exception as exc:
        return {"error": str(exc)}


def sensationalism_predictive_score(text: str) -> Dict[str, Any]:
    """
    Score how sensational the text is using the trained classifier.
    """
    try:
        model = Sensationalism()
        probs = _coerce_floats(model.probability(text))
        score = float(probs.get("sensationalism", 0.0))
        return {"predictive_score": score}
    except Exception as exc:
        return {"error": str(exc)}


def sentiment_predictive_score(text: str) -> Dict[str, Any]:
    """
    Detect sentiment using the VADER-based sentiment factor.
    Returns probabilities for negative, neutral, and positive.
    """
    model = Sentiment()
    scores = _coerce_floats(model.probability(text))
    label = max(scores, key=scores.get)
    return {"predictive_label": label, "all_scores": scores}


def toxicity_predictive_score(text: str) -> Dict[str, Any]:
    """
    Detect toxicity using the RoBERTa multi-class classifier.
    Returns probabilities and categorical label.
    """
    detector = Toxicity()
    scores = _coerce_floats(detector.probability(text))
    label = detector.categorize(text)
    return {"predictive_label": label, "all_scores": scores}


def combine_scores(
    predictive_val: Any,
    generative_val: Any,
    is_numeric: bool = True,
    weight_predictive: float = 0.5,
    weight_generative: float = 0.5,
) -> Any:
    """
    Combine predictive and generative scores using weighted average or label selection.
    """
    if is_numeric:
        p = float(predictive_val) if predictive_val is not None else 0.0
        g = float(generative_val) if generative_val is not None else 0.0
        return (p * weight_predictive) + (g * weight_generative)
    else:
        # For categorical, if they match, great. If not, use generative as tie-breaker/primary
        # unless generative is missing.
        if generative_val:
            return generative_val
        return predictive_val


def _coerce_percent_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score <= 1:
        score *= 100
    return max(0.0, min(100.0, score))


def _parse_json_response(raw_response: Any) -> Dict[str, Any]:
    if raw_response is None:
        raise ValueError("LLM returned no response content.")

    response_text = str(raw_response).strip()
    if not response_text:
        raise ValueError("LLM returned an empty response.")

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    if "```" in response_text:
        for part in response_text.split("```"):
            candidate = part.strip()
            if not candidate:
                continue
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    start = response_text.find("{")
    end = response_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(response_text[start : end + 1])

    raise ValueError("LLM response did not contain valid JSON.")


def final_veracity_scoring_agent(
    title: str,
    body: str,
    clickbait_score: Any,
    headline_body_score: Any,
    sensationalism_score: Any,
    sentiment_label: str,
    toxicity_label: str,
    political_label: str,
) -> Dict[str, Any]:
    """
    Compute the final veracity assessment using the FINAL_VERACITY_PROMPT template.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return _missing_key_error("OPENROUTER_API_KEY")

    replacements = {
        "{title}": title,
        "{body}": body,
        "{clickbait_score}": str(_coerce_percent_score(clickbait_score) or "N/A"),
        "{headline_body_score}": str(
            _coerce_percent_score(headline_body_score) or "N/A"
        ),
        "{sensationalism_score}": str(
            _coerce_percent_score(sensationalism_score) or "N/A"
        ),
        "{sentiment_label}": str(sentiment_label or "N/A"),
        "{toxicity_label}": str(toxicity_label or "N/A"),
        "{political_label}": str(political_label or "N/A"),
    }

    prompt = FINAL_VERACITY_PROMPT
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)

    article = {
        "title": "",
        "source": "",
        "author": "",
        "publication_date": "",
        "content": "",
    }

    try:
        response = generate(
            api_key,
            "Return only valid JSON.",
            prompt,
            article,
            model=os.getenv("OPENROUTER_LLM_MODEL", "google/gemini-2.5-pro"),
            provider="openrouter",
        )
        return _parse_json_response(response)
    except Exception as exc:
        return {"error": str(exc)}
