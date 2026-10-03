from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


MISSING_TEXT = {"", "na", "n/a", "null", "none", "nan", "missing"}
IMPUTATION_STRATEGIES = {"mean", "median", "most_frequent", "constant", "knn", "regression", "iterative"}


def _profile_column_types(profile: Dict[str, Any]) -> Dict[str, str]:
    return profile.get("column_types") or profile.get("metadata", {}).get("column_types", {})


def _semantic_types(profile: Dict[str, Any]) -> Dict[str, str]:
    return profile.get("metadata", {}).get("semantic_types", {})


def _normalize_text(df: pd.DataFrame, semantic_types: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    result = df.copy()
    changes: Dict[str, Any] = {"trimmed_or_standardized": {}, "date_formats": {}, "case_normalized": []}

    for column in result.columns:
        series = result[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            text = series.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
            text = text.mask(text.str.lower().isin(MISSING_TEXT))
            changed_count = int((series.astype("string") != text).fillna(False).sum())
            semantic_type = semantic_types.get(column, "unknown")

            if semantic_type == "email":
                text = text.str.lower()
                changes["case_normalized"].append(column)
            elif semantic_type == "date" or "date" in str(column).lower() or "time" in str(column).lower():
                parsed = pd.to_datetime(text, errors="coerce")
                invalid_count = int((text.notna() & parsed.isna()).sum())
                text = parsed.dt.strftime("%Y-%m-%d").astype("string")
                changes["date_formats"][column] = {"normalized_format": "YYYY-MM-DD", "invalid_values": invalid_count}
            elif semantic_type == "unknown" and text.str.lower().nunique(dropna=True) <= min(50, max(2, int(len(text) * 0.6))):
                text = text.str.lower()
                changes["case_normalized"].append(column)

            result[column] = text
            if changed_count:
                changes["trimmed_or_standardized"][column] = changed_count

    return result, changes


def _schema_inference(df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    inferred_types = _profile_column_types(profile)
    semantic_types = _semantic_types(profile)
    profiles = profile.get("column_profiles", {})
    schema: Dict[str, Any] = {}

    for column in df.columns:
        inferred_type = inferred_types.get(column, "unknown")
        column_profile = profiles.get(column, {})
        expected_format = "YYYY-MM-DD" if semantic_types.get(column) == "date" else None
        expected_range = None
        if column_profile.get("numeric_statistics"):
            numeric_stats = column_profile["numeric_statistics"]
            expected_range = {"min": numeric_stats.get("min"), "max": numeric_stats.get("max")}
        schema[column] = {
            "expected_type": inferred_type,
            "semantic_type": semantic_types.get(column, "unknown"),
            "expected_format": expected_format,
            "expected_range": expected_range,
        }
    return schema


def _fuzzy_duplicate_candidates(df: pd.DataFrame, semantic_types: Dict[str, str], threshold: float) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    compared_pairs = 0
    max_pairs = 20000

    for column in df.columns:
        if semantic_types.get(column) != "name":
            continue
        values = df[column].dropna().astype(str).str.lower().str.strip()
        unique_values = list(dict.fromkeys(value for value in values if value))
        if len(unique_values) > 500:
            unique_values = unique_values[:500]
        for left_index, left in enumerate(unique_values):
            for right in unique_values[left_index + 1:]:
                compared_pairs += 1
                if compared_pairs > max_pairs:
                    return candidates
                similarity = SequenceMatcher(None, left, right).ratio()
                if similarity >= threshold:
                    candidates.append({"column": column, "left": left, "right": right, "similarity": round(similarity, 4)})
    return candidates


def _impute_numeric(df: pd.DataFrame, numeric_columns: List[str], strategy: str) -> Dict[str, Any]:
    filled: Dict[str, Any] = {}
    columns_with_values = [column for column in numeric_columns if df[column].notna().any()]
    all_missing = [column for column in numeric_columns if not df[column].notna().any()]
    for column in all_missing:
        df[column] = 0.0
        filled[column] = {"count": int(len(df)), "method": "constant_zero_all_missing"}

    if not columns_with_values:
        return filled

    missing_counts = {column: int(df[column].isna().sum()) for column in columns_with_values if df[column].isna().any()}
    if not missing_counts:
        return filled

    if strategy in {"knn", "regression", "iterative"} and len(df) > 1:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer, KNNImputer
        from sklearn.linear_model import LinearRegression

        matrix = df[columns_with_values].astype(float)
        if strategy == "knn":
            imputer = KNNImputer(n_neighbors=min(5, max(1, len(df) - 1)))
        else:
            estimator = LinearRegression() if strategy == "regression" else None
            imputer = IterativeImputer(estimator=estimator, max_iter=10, random_state=0)
        imputed = imputer.fit_transform(matrix)
        df.loc[:, columns_with_values] = imputed
        method = strategy
    else:
        method = strategy if strategy in {"mean", "median", "constant"} else "most_frequent"
        for column in columns_with_values:
            if not df[column].isna().any():
                continue
            if method == "mean":
                value = df[column].mean()
            elif method == "constant":
                value = 0.0
            else:
                value = df[column].median()
            df[column] = df[column].fillna(value)

    for column, count in missing_counts.items():
        filled[column] = {"count": count, "method": method}
    return filled


def _quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    cells = len(df) * len(df.columns)
    missing = int(df.isna().sum().sum())
    completeness = 100.0 if not cells else 100.0 * (cells - missing) / cells
    duplicate_count = int(df.duplicated().sum())
    uniqueness = 100.0 if len(df) == 0 else 100.0 * (len(df) - duplicate_count) / len(df)

    type_error_rates = []
    for column in df.columns:
        non_null = df[column].dropna()
        if not len(non_null):
            continue
        numeric = pd.to_numeric(non_null, errors="coerce")
        if numeric.notna().sum() and numeric.notna().sum() < len(non_null):
            type_error_rates.append(1 - numeric.notna().sum() / len(non_null))
    consistency = 100.0 * (1 - float(np.mean(type_error_rates))) if type_error_rates else 100.0
    overall = round((completeness * 0.4) + (uniqueness * 0.3) + (consistency * 0.3), 2)
    return {
        "overall": overall,
        "dimensions": {
            "completeness": round(completeness, 2),
            "uniqueness": round(uniqueness, 2),
            "type_consistency": round(consistency, 2),
        },
    }


def transform_features(
    df: pd.DataFrame,
    scaling: str = "standard",
    encode_categoricals: bool = True,
    extract_date_features: bool = True,
) -> pd.DataFrame:
    """Scale numeric values, one-hot encode categories, and derive date parts."""
    if scaling not in {"none", "standard", "minmax"}:
        raise ValueError("scaling must be 'none', 'standard', or 'minmax'")

    transformed = df.copy()
    date_columns = [column for column in transformed.columns if "date" in str(column).lower() or "time" in str(column).lower()]
    for column in date_columns:
        parsed = pd.to_datetime(transformed[column], errors="coerce")
        if extract_date_features:
            transformed[f"{column}_year"] = parsed.dt.year
            transformed[f"{column}_month"] = parsed.dt.month
            transformed[f"{column}_day"] = parsed.dt.day
            transformed[f"{column}_dayofweek"] = parsed.dt.dayofweek
            transformed = transformed.drop(columns=[column])

    numeric_columns = transformed.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        numeric = pd.to_numeric(transformed[column], errors="coerce")
        if scaling == "standard" and numeric.std() not in (0, np.nan) and pd.notna(numeric.std()):
            transformed[column] = (numeric - numeric.mean()) / numeric.std()
        elif scaling == "minmax" and numeric.max() != numeric.min():
            transformed[column] = (numeric - numeric.min()) / (numeric.max() - numeric.min())
        else:
            transformed[column] = numeric.fillna(0)

    categorical_columns = transformed.select_dtypes(include=["object", "string", "category"]).columns
    if encode_categoricals and len(categorical_columns):
        for column in categorical_columns:
            transformed[column] = transformed[column].astype("string").str.strip().str.lower()
        transformed = pd.get_dummies(transformed, columns=list(categorical_columns), dummy_na=False, dtype=int)
    return transformed


def clean_dataset(
    file_path: str | Path,
    profile: Dict[str, Any],
    output_dir: str | Path,
    imputation_strategy: str = "median",
    fuzzy_threshold: float = 0.92,
    transformations: Dict[str, Any] | None = None,
) -> Path:
    if imputation_strategy not in IMPUTATION_STRATEGIES:
        raise ValueError(f"imputation_strategy must be one of {sorted(IMPUTATION_STRATEGIES)}")
    if not 0.0 <= fuzzy_threshold <= 1.0:
        raise ValueError("fuzzy_threshold must be between 0 and 1")

    input_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(input_path)
    df = original.copy()
    semantic_types = _semantic_types(profile)
    before_score = _quality_score(df)
    schema = _schema_inference(df, profile)
    fuzzy_candidates = _fuzzy_duplicate_candidates(df, semantic_types, fuzzy_threshold)

    df, normalization_log = _normalize_text(df, semantic_types)
    exact_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    numeric_columns = [column for column, details in schema.items() if details["expected_type"] == "numeric"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column].astype("string").str.replace(",", "", regex=False), errors="coerce").astype("float64")

    date_columns = [column for column, details in schema.items() if details["semantic_type"] == "date"]
    for column in date_columns:
        parsed = pd.to_datetime(df[column], errors="coerce")
        df[column] = parsed.dt.strftime("%Y-%m-%d").astype("string")

    missing_before = {column: int(df[column].isna().sum()) for column in df.columns}
    numeric_imputations = _impute_numeric(df, numeric_columns, imputation_strategy)
    categorical_imputations: Dict[str, Any] = {}
    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        if not missing_count:
            continue
        if column in numeric_columns:
            continue
        if df[column].notna().any():
            mode = df[column].mode(dropna=True)
            fill_value = mode.iloc[0] if len(mode) else "Unknown"
        else:
            fill_value = "Unknown"
        df[column] = df[column].fillna(fill_value)
        categorical_imputations[column] = {"count": missing_count, "method": "most_frequent", "fallback": str(fill_value)}

    transformed_path = None
    transform_options = transformations or {}
    if transform_options:
        transformed = transform_features(
            df,
            scaling=transform_options.get("scaling", "standard"),
            encode_categoricals=transform_options.get("encode_categoricals", True),
            extract_date_features=transform_options.get("extract_date_features", True),
        )
        transformed_path = output_dir / "transformed_features.csv"
        transformed.to_csv(transformed_path, index=False)

    cleaned_path = output_dir / "cleaned_data.csv"
    df.to_csv(cleaned_path, index=False)
    after_score = _quality_score(df)
    invalid_counts = {
        column: int(df[column].astype("string").str.contains(r"\b(?:invalid|unknown|error)\b", case=False, na=False).sum())
        for column in df.columns
        if pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column])
    }

    cleaning_log = {
        "input_file": input_path.name,
        "output_file": cleaned_path.name,
        "schema_inference": schema,
        "actions": {
            "exact_duplicates_removed": exact_duplicates,
            "fuzzy_duplicate_candidates": fuzzy_candidates,
            "fuzzy_candidate_policy": "reported_only; candidates are not removed automatically",
            "normalization": normalization_log,
            "numeric_imputations": numeric_imputations,
            "categorical_imputations": categorical_imputations,
            "missing_values_before_imputation": missing_before,
            "textual_anomaly_counts_after_cleaning": invalid_counts,
        },
        "quality_score": {
            "before": before_score,
            "after": after_score,
            "delta": round(after_score["overall"] - before_score["overall"], 2),
        },
        "transformations": {
            "options": transform_options,
            "output_file": transformed_path.name if transformed_path else None,
        },
    }
    (output_dir / "cleaning_log.json").write_text(json.dumps(cleaning_log, indent=2, allow_nan=False), encoding="utf-8")
    return cleaned_path
