from __future__ import annotations

import json
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


def detect_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    return {column: int(df[column].isna().sum()) for column in df.columns}


def detect_pii_columns(df: pd.DataFrame) -> List[str]:
    pii_candidates: List[str] = []
    for column in df.columns:
        column_name = str(column).lower()
        if any(keyword in column_name for keyword in ["email", "phone", "name"]):
            pii_candidates.append(column)
    return pii_candidates


def profile_dataset(file_path: str | Path) -> Dict[str, Any]:
    df = pd.read_csv(file_path)
    profile = {
        "dataset_name": Path(file_path).name,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "column_types": infer_column_types(df),
        "missing_values": detect_missing_values(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "pii_columns": detect_pii_columns(df),
        "summary": {
            "null_percentage": round((df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2) if df.shape[0] and df.shape[1] else 0.0,
            "unique_values": {column: int(df[column].nunique(dropna=True)) for column in df.columns},
        },
    }
    return profile
