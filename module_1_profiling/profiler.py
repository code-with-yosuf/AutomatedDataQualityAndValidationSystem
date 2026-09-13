from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    column_types: Dict[str, str] = {}
    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        if non_null.empty:
            column_types[column] = "unknown"
        elif pd.api.types.is_numeric_dtype(non_null):
            column_types[column] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(non_null):
            column_types[column] = "datetime"
        else:
            column_types[column] = "string"
    return column_types


def detect_semantic_types(df: pd.DataFrame) -> Dict[str, str]:
    semantic_types: Dict[str, str] = {}
    for column in df.columns:
        lower_name = str(column).lower()
        series = df[column].dropna()

        if "email" in lower_name:
            semantic_types[column] = "email"
        elif "phone" in lower_name:
            semantic_types[column] = "phone"
        elif "name" in lower_name:
            semantic_types[column] = "name"
        elif "date" in lower_name or "time" in lower_name:
            semantic_types[column] = "date"
        elif "id" in lower_name:
            semantic_types[column] = "identifier"
        elif len(series) and series.map(lambda x: bool(re.match(r"^\d{4}-\d{2}-\d{2}$", str(x).strip()))).all():
            semantic_types[column] = "date"
        else:
            semantic_types[column] = "unknown"
    return semantic_types


def detect_mixed_type_columns(df: pd.DataFrame) -> List[str]:
    mixed_types: List[str] = []
    for column in df.columns:
        non_null = df[column].dropna()
        if non_null.empty:
            continue

        sample_types = []
        for value in non_null.head(20):
            value_str = str(value).strip()
            if value_str == "":
                continue

            try:
                float(value_str)
                kind = "numeric"
            except ValueError:
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value_str):
                    kind = "date"
                else:
                    kind = "string"

            sample_types.append(kind)

        if len(set(sample_types)) > 1:
            mixed_types.append(column)
    return mixed_types


def detect_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    return {column: int(df[column].isna().sum()) for column in df.columns}


def detect_pii_columns(df: pd.DataFrame) -> List[str]:
    pii_candidates: List[str] = []
    for column in df.columns:
        column_name = str(column).lower()
        if any(keyword in column_name for keyword in ["email", "phone", "name"]):
            pii_candidates.append(column)
    return pii_candidates


def detect_suspicious_columns(df: pd.DataFrame) -> List[str]:
    suspicious: List[str] = []
    for column in df.columns:
        series = df[column].dropna()
        if series.empty:
            suspicious.append(column)
            continue

        if series.nunique(dropna=True) == 1:
            suspicious.append(column)
            continue

        if column.lower().endswith("date") or "date" in column.lower():
            invalid_dates = series.map(lambda x: not bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(x).strip())) if pd.notna(x) else False).sum()
            if invalid_dates > 0:
                suspicious.append(column)

        elif series.map(lambda x: isinstance(x, str) and len(str(x).strip()) > 100).any():
            suspicious.append(column)

        elif column.lower() in {"age", "amount_spent"}:
            invalid_numbers = pd.to_numeric(series, errors="coerce").isna().sum()
            if invalid_numbers > 0:
                suspicious.append(column)

    return sorted(set(suspicious))


def profile_dataset(file_path: str | Path) -> Dict[str, Any]:
    df = pd.read_csv(file_path)
    column_types = infer_column_types(df)
    suspicious_columns = detect_suspicious_columns(df)

    profile = {
        "dataset_name": Path(file_path).name,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "metadata": {
            "column_types": column_types,
            "semantic_types": detect_semantic_types(df),
            "mixed_type_columns": detect_mixed_type_columns(df),
            "pii_columns": detect_pii_columns(df),
        },
        "missing_values": detect_missing_values(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "quality_checks": {
            "suspicious_columns": suspicious_columns,
            "null_percentage": round((df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2) if df.shape[0] and df.shape[1] else 0.0,
        },
        "summary": {
            "unique_values": {column: int(df[column].nunique(dropna=True)) for column in df.columns},
            "profiles": {
                "total_nulls": int(df.isna().sum().sum()),
                "total_duplicates": int(df.duplicated().sum()),
                "suspicious_columns_count": len(suspicious_columns),
            },
        },
    }
    return profile
