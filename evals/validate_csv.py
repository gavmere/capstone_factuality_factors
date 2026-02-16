#!/usr/bin/env python3
"""
Validate the Eval-Full-Test.csv file for structure and data integrity.
"""

import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any
import re

# Expected columns from the codebase
NUMERIC_FACTORS = ["Clickbait", "Headline-Body-Relation", "Sensationalism"]
CATEGORICAL_FACTORS = ["Political Affiliation", "Sentiment Analysis", "Toxicity"]

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


def normalize_column_name(col: str) -> str:
    """Normalize column names to handle variations."""
    # Remove newlines and extra whitespace
    col_clean = " ".join(col.split())
    col_lower = col_clean.lower().strip()

    # Map common variations
    if "article" in col_lower or col_lower == "url":
        return "url"
    if "content" in col_lower or "body" in col_lower:
        return "content"
    if "headline" in col_lower and "body" not in col_lower:
        return "headline"

    # Factor name mappings (check more specific ones first)
    if "headline" in col_lower and "body" in col_lower and "relation" in col_lower:
        return "Headline-Body-Relation"
    if "clickbait" in col_lower:
        return "Clickbait"
    if (
        "political" in col_lower or "party" in col_lower
    ) and "affiliation" in col_lower:
        return "Political Affiliation"
    if "sensationalism" in col_lower:
        return "Sensationalism"
    if "sentiment" in col_lower:
        return "Sentiment Analysis"
    if "toxicity" in col_lower:
        return "Toxicity"

    return col_clean


def validate_numeric_factor(value: Any, factor_name: str) -> Tuple[bool, str]:
    """Validate numeric factor value (should be 0-100)."""
    if pd.isna(value):
        return True, ""  # Missing values are OK

    try:
        # Try to extract number from string if needed
        if isinstance(value, str):
            # Remove any descriptive text, keep only numbers
            match = re.search(r"(\d+(?:\.\d+)?)", value)
            if match:
                num_value = float(match.group(1))
            else:
                return False, f"Could not extract numeric value from: {value}"
        else:
            num_value = float(value)

        if 0 <= num_value <= 100:
            return True, ""
        else:
            return False, f"Value {num_value} is outside valid range [0, 100]"
    except (ValueError, TypeError) as e:
        return False, f"Cannot convert to number: {value}"


def validate_categorical_factor(value: Any, factor_name: str) -> Tuple[bool, str]:
    """Validate categorical factor value."""
    if pd.isna(value):
        return True, ""  # Missing values are OK

    value_str = str(value).strip()
    value_lower = value_str.lower()

    valid_values = CATEGORICAL_VALUES.get(factor_name, [])

    # Check exact match or partial match
    for valid in valid_values:
        if (
            valid.lower() == value_lower
            or valid.lower() in value_lower
            or value_lower in valid.lower()
        ):
            return True, ""

    # Check for common variations
    if factor_name == "Political Affiliation":
        if any(
            term in value_lower
            for term in ["democrat", "republican", "neutral", "other"]
        ):
            return True, ""
    elif factor_name == "Sentiment Analysis":
        if any(term in value_lower for term in ["positive", "negative", "neutral"]):
            return True, ""

    return (
        False,
        f"Invalid value '{value_str}' for {factor_name}. Expected one of: {valid_values}",
    )


def validate_csv(file_path: Path) -> Dict[str, Any]:
    """Validate the CSV file and return a report."""
    issues = []
    warnings = []
    info = []

    # Try to read the CSV
    try:
        # Try reading with different encodings and quote handling
        df = pd.read_csv(
            file_path, encoding="utf-8", quotechar='"', on_bad_lines="skip"
        )
        info.append(
            f"✅ Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns"
        )
    except Exception as e:
        try:
            df = pd.read_csv(
                file_path, encoding="latin-1", quotechar='"', on_bad_lines="skip"
            )
            warnings.append(f"⚠️  Loaded with latin-1 encoding (original error: {e})")
        except Exception as e2:
            return {
                "valid": False,
                "error": f"Failed to read CSV file: {e2}",
                "issues": issues,
                "warnings": warnings,
                "info": info,
            }

    # Check basic structure
    info.append(f"Columns found: {', '.join(df.columns.tolist())}")

    # Normalize column names for checking
    normalized_cols = {col: normalize_column_name(col) for col in df.columns}

    # Check for required columns
    has_url = any(
        "url" in norm.lower() or "article" in norm.lower()
        for norm in normalized_cols.values()
    )
    has_content = any(
        "content" in norm.lower() or "body" in norm.lower()
        for norm in normalized_cols.values()
    )
    has_headline = any("headline" in norm.lower() for norm in normalized_cols.values())

    if not has_url and not has_headline:
        issues.append("Missing required column: 'url' or 'article' (for article URL)")
    else:
        info.append("✅ Found URL/article column")

    if not has_content:
        issues.append("Missing required column: 'Content' or 'body'")
    else:
        info.append("✅ Found Content/body column")

    # Check for factor columns
    found_factors = []
    for col in df.columns:
        norm_col = normalize_column_name(col)
        # Check if normalized column matches any factor
        if norm_col in NUMERIC_FACTORS + CATEGORICAL_FACTORS:
            found_factors.append((col, norm_col))
        # Also check if original column name contains factor keywords
        elif (
            "headline" in col.lower()
            and "body" in col.lower()
            and "relation" in col.lower()
        ):
            found_factors.append((col, "Headline-Body-Relation"))

    if not found_factors:
        issues.append(
            "No factor columns found. Expected: Clickbait, Political Affiliation, Sensationalism, Sentiment Analysis, Toxicity, Headline-Body-Relation"
        )
    else:
        info.append(
            f"✅ Found {len(found_factors)} factor column(s): {[f[1] for f in found_factors]}"
        )

    # Validate each factor column
    for orig_col, norm_col in found_factors:
        if norm_col in NUMERIC_FACTORS:
            # Validate numeric values
            invalid_count = 0
            invalid_examples = []
            for idx, value in enumerate(df[orig_col]):
                is_valid, error_msg = validate_numeric_factor(value, norm_col)
                if not is_valid:
                    invalid_count += 1
                    if len(invalid_examples) < 5:
                        invalid_examples.append(f"Row {idx + 2}: {error_msg}")

            if invalid_count > 0:
                issues.append(
                    f"Column '{orig_col}' ({norm_col}): {invalid_count} invalid numeric values"
                )
                issues.extend(invalid_examples[:5])
                if invalid_count > 5:
                    issues.append(f"  ... and {invalid_count - 5} more")
            else:
                info.append(
                    f"✅ Column '{orig_col}' ({norm_col}): All numeric values valid (0-100)"
                )

        elif norm_col in CATEGORICAL_FACTORS:
            # Validate categorical values
            invalid_count = 0
            invalid_examples = []
            unique_values = df[orig_col].dropna().unique()

            for idx, value in enumerate(df[orig_col]):
                is_valid, error_msg = validate_categorical_factor(value, norm_col)
                if not is_valid:
                    invalid_count += 1
                    if len(invalid_examples) < 5:
                        invalid_examples.append(f"Row {idx + 2}: {error_msg}")

            if invalid_count > 0:
                issues.append(
                    f"Column '{orig_col}' ({norm_col}): {invalid_count} invalid categorical values"
                )
                issues.extend(invalid_examples[:5])
                if invalid_count > 5:
                    issues.append(f"  ... and {invalid_count - 5} more")
            else:
                info.append(
                    f"✅ Column '{orig_col}' ({norm_col}): All categorical values valid"
                )
                info.append(
                    f"   Unique values found: {', '.join(map(str, unique_values[:10]))}"
                )

    # Check for missing values
    missing_data = df.isnull().sum()
    if missing_data.any():
        for col, count in missing_data.items():
            if count > 0:
                pct = (count / len(df)) * 100
                if pct > 50:
                    issues.append(
                        f"Column '{col}': {count} missing values ({pct:.1f}%)"
                    )
                elif pct > 10:
                    warnings.append(
                        f"Column '{col}': {count} missing values ({pct:.1f}%)"
                    )
                else:
                    info.append(f"Column '{col}': {count} missing values ({pct:.1f}%)")

    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        warnings.append(f"Found {duplicates} duplicate rows")
    else:
        info.append("✅ No duplicate rows found")

    # Summary
    is_valid = len(issues) == 0

    return {
        "valid": is_valid,
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "issues": issues,
        "warnings": warnings,
        "info": info,
        "columns": df.columns.tolist(),
        "found_factors": [f[1] for f in found_factors],
    }


def main():
    """Main validation function."""
    csv_path = Path(__file__).parent / "Eval-Full-Test.csv"

    if not csv_path.exists():
        print(f"❌ Error: File not found: {csv_path}")
        sys.exit(1)

    print(f"Validating: {csv_path}")
    print("=" * 80)

    result = validate_csv(csv_path)

    # Print results
    print("\n📊 VALIDATION RESULTS")
    print("=" * 80)

    if result["valid"]:
        print("✅ CSV FILE IS VALID")
    else:
        print("❌ CSV FILE HAS ISSUES")

    print(f"\n📈 Statistics:")
    print(f"   Rows: {result['num_rows']}")
    print(f"   Columns: {result['num_columns']}")
    print(
        f"   Factor columns found: {', '.join(result['found_factors']) if result['found_factors'] else 'None'}"
    )

    if result["info"]:
        print(f"\nℹ️  Information:")
        for msg in result["info"]:
            print(f"   {msg}")

    if result["warnings"]:
        print(f"\n⚠️  Warnings ({len(result['warnings'])}):")
        for msg in result["warnings"]:
            print(f"   {msg}")

    if result["issues"]:
        print(f"\n❌ Issues ({len(result['issues'])}):")
        for msg in result["issues"]:
            print(f"   {msg}")

    print("\n" + "=" * 80)

    if result["valid"]:
        print("✅ Validation passed!")
        sys.exit(0)
    else:
        print("❌ Validation failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
