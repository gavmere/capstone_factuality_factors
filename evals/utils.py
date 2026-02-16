"""
Utility functions for data processing and metrics calculation in LLM evaluation.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# Factor name mappings for normalization
FACTOR_NAME_MAPPINGS = {
    "clickbait": "Clickbait",
    "headline-body-relation": "Headline-Body-Relation",
    "headline_body_relation": "Headline-Body-Relation",
    "party affiliation": "Political Affiliation",
    "political affiliation": "Political Affiliation",
    "party affliation": "Political Affiliation",  # Handle typo in original
    "sensationalism": "Sensationalism",
    "sentiment analysis": "Sentiment Analysis",
    "sentiment": "Sentiment Analysis",
    "toxicity": "Toxicity",
}

# Expected factor names
# Note: Ground truth uses 0-100 scale for numeric factors
NUMERIC_FACTORS = ["Clickbait", "Headline-Body-Relation", "Sensationalism"]
CATEGORICAL_FACTORS = ["Political Affiliation", "Sentiment Analysis", "Toxicity"]

# Expected values for categorical factors
CATEGORICAL_VALUES = {
    "Political Affiliation": [
        "Democratic",
        "Democrat",
        "Republican",
        "Neutral",
        "Other",
    ],
    "Sentiment Analysis": ["Positive", "Negative", "Neutral"],
    "Toxicity": ["Friendly", "Neutral", "Rude", "Toxic", "Super_Toxic"],
}


def normalize_factor_name(name: str) -> str:
    """Normalize factor names to standard format."""
    name_lower = name.lower().strip()
    return FACTOR_NAME_MAPPINGS.get(name_lower, name)


def normalize_categorical_value(value: str, factor: str) -> str:
    """Normalize categorical values to match expected format."""
    if not isinstance(value, str):
        value = str(value)

    value = value.strip()
    value_lower = value.lower()

    # Handle variations
    if factor == "Political Affiliation":
        if "democrat" in value_lower or "democratic" in value_lower:
            return "Democratic"
        elif "republican" in value_lower:
            return "Republican"
        elif "neutral" in value_lower:
            return "Neutral"
        return "Other"

    if factor == "Sentiment Analysis":
        if "positive" in value_lower:
            return "Positive"
        elif "negative" in value_lower:
            return "Negative"
        return "Neutral"

    if factor == "Toxicity":
        if "friendly" in value_lower:
            return "Friendly"
        elif "super" in value_lower and "toxic" in value_lower:
            return "Super_Toxic"
        elif "toxic" in value_lower:
            return "Toxic"
        elif "rude" in value_lower:
            return "Rude"
        return "Neutral"

    return value


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Parse LLM JSON response, handling various formats and errors.

    Returns:
        Dictionary with factor names as keys and values
    """
    if not response_text or not isinstance(response_text, str):
        return {}

    response_text = response_text.strip()

    # Try to parse as JSON
    try:
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            # Extract JSON from code block
            match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL
            )
            if match:
                response_text = match.group(1)
            else:
                # Try to find JSON object
                match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if match:
                    response_text = match.group(0)

        data = json.loads(response_text)

        # Normalize keys
        normalized_data = {}
        for key, value in data.items():
            normalized_key = normalize_factor_name(key)
            normalized_data[normalized_key] = value

        return normalized_data
    except json.JSONDecodeError:
        # Try to extract key-value pairs manually
        result = {}
        for factor in NUMERIC_FACTORS + CATEGORICAL_FACTORS:
            # Look for factor name followed by colon or equals
            pattern = rf"{re.escape(factor)}[:\s=]+([^\n,}}]+)"
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip().strip("\"'")
                result[factor] = value
        return result


def convert_to_numeric(value: Any, factor: str) -> Optional[float]:
    """Convert value to numeric for numeric factors.

    Note: Ground truth uses 0-100 scale, but we normalize to 0-1 for comparison.
    """
    if pd.isna(value) or value is None:
        return None

    if isinstance(value, (int, float)):
        num_value = float(value)
        # If value is > 1, assume it's on 0-100 scale and normalize to 0-1
        if num_value > 1.0:
            num_value = num_value / 100.0
        return num_value

    if isinstance(value, str):
        # Try to extract number from string
        match = re.search(r"[\d.]+", value)
        if match:
            num_value = float(match.group())
            # If value is > 1, assume it's on 0-100 scale and normalize to 0-1
            if num_value > 1.0:
                num_value = num_value / 100.0
            return num_value

    return None


def compare_numeric_values(
    predicted: Any, ground_truth: Any, tolerance: float = 0.1
) -> Tuple[bool, float]:
    """
    Compare numeric values with tolerance.

    Returns:
        (is_match, absolute_error)
    """
    pred_num = convert_to_numeric(predicted, "numeric")
    gt_num = convert_to_numeric(ground_truth, "numeric")

    if pred_num is None or gt_num is None:
        return False, float("inf")

    abs_error = abs(pred_num - gt_num)
    is_match = abs_error <= tolerance

    return is_match, abs_error


def compare_categorical_values(predicted: Any, ground_truth: Any, factor: str) -> bool:
    """Compare categorical values with normalization."""
    if pd.isna(predicted) or pd.isna(ground_truth):
        return False

    pred_norm = normalize_categorical_value(str(predicted), factor)
    gt_norm = normalize_categorical_value(str(ground_truth), factor)

    return pred_norm == gt_norm


def calculate_metrics(
    ground_truth: List[Any],
    predictions: List[Any],
    factor: str,
    is_numeric: bool = False,
    tolerance: float = 0.1,
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for a factor.

    Returns:
        Dictionary with metrics (accuracy, precision, recall, F1, MAE, RMSE, etc.)
    """
    metrics = {}

    if is_numeric:
        # Numeric metrics
        gt_numeric = [convert_to_numeric(gt, factor) for gt in ground_truth]
        pred_numeric = [convert_to_numeric(pred, factor) for pred in predictions]

        # Filter out None values
        valid_pairs = [
            (gt, pred)
            for gt, pred in zip(gt_numeric, pred_numeric)
            if gt is not None and pred is not None
        ]

        if valid_pairs:
            gt_vals, pred_vals = zip(*valid_pairs)
            gt_vals = np.array(gt_vals)
            pred_vals = np.array(pred_vals)

            # Calculate errors
            errors = np.abs(gt_vals - pred_vals)
            metrics["mae"] = float(np.mean(errors))
            metrics["rmse"] = float(np.sqrt(np.mean(errors**2)))
            metrics["max_error"] = float(np.max(errors))
            metrics["min_error"] = float(np.min(errors))

            # Accuracy based on tolerance
            matches = errors <= tolerance
            metrics["accuracy"] = float(np.mean(matches))
            metrics["num_correct"] = int(np.sum(matches))
            metrics["num_total"] = len(valid_pairs)
        else:
            metrics["mae"] = None
            metrics["rmse"] = None
            metrics["accuracy"] = 0.0
            metrics["num_correct"] = 0
            metrics["num_total"] = 0
    else:
        # Categorical metrics
        # Normalize all values
        gt_normalized = [
            normalize_categorical_value(str(gt), factor) for gt in ground_truth
        ]
        pred_normalized = [
            normalize_categorical_value(str(pred), factor) for pred in predictions
        ]

        # Filter out None/NaN values
        valid_pairs = [
            (gt, pred)
            for gt, pred in zip(gt_normalized, pred_normalized)
            if gt and pred and gt != "nan" and pred != "nan"
        ]

        if valid_pairs:
            gt_vals, pred_vals = zip(*valid_pairs)

            # Calculate accuracy
            matches = [gt == pred for gt, pred in valid_pairs]
            metrics["accuracy"] = float(np.mean(matches))
            metrics["num_correct"] = int(np.sum(matches))
            metrics["num_total"] = len(valid_pairs)

            # Calculate precision, recall, F1
            try:
                # Get unique labels
                all_labels = sorted(set(gt_vals + pred_vals))
                precision, recall, f1, support = precision_recall_fscore_support(
                    gt_vals,
                    pred_vals,
                    labels=all_labels,
                    zero_division=0,
                    average="weighted",
                )
                metrics["precision"] = float(precision)
                metrics["recall"] = float(recall)
                metrics["f1"] = float(f1)

                # Confusion matrix
                cm = confusion_matrix(gt_vals, pred_vals, labels=all_labels)
                metrics["confusion_matrix"] = cm.tolist()
                metrics["confusion_matrix_labels"] = all_labels
            except Exception as e:
                metrics["precision"] = None
                metrics["recall"] = None
                metrics["f1"] = None
                metrics["confusion_matrix"] = None
        else:
            metrics["accuracy"] = 0.0
            metrics["num_correct"] = 0
            metrics["num_total"] = 0
            metrics["precision"] = None
            metrics["recall"] = None
            metrics["f1"] = None

    return metrics


def validate_csv_structure(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that CSV has required columns.

    Returns:
        (is_valid, list_of_missing_columns)
    """
    required_columns = ["headline"]
    optional_columns = ["body", "content", "url"]
    factor_columns = NUMERIC_FACTORS + CATEGORICAL_FACTORS

    missing = []

    # Check required columns
    if "headline" not in df.columns:
        missing.append("headline")

    # Check for body or content
    if "body" not in df.columns and "content" not in df.columns:
        missing.append("body or content")

    # Check for at least one factor column
    has_factor = any(col in df.columns for col in factor_columns)
    if not has_factor:
        missing.append("at least one factor column")

    return len(missing) == 0, missing


def get_body_column(df: pd.DataFrame) -> Optional[str]:
    """Get the body/content column name from dataframe."""
    if "body" in df.columns:
        return "body"
    elif "content" in df.columns:
        return "content"
    return None
