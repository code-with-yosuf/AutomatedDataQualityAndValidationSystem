from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
DATE_PATTERN = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")
MISSING_TEXT = {"", "na", "n/a", "null", "none", "nan", "missing"}


def _clean_values(series: pd.Series) -> pd.Series:
    cleaned = series.copy()
    if pd.api.types.is_object_dtype(cleaned) or pd.api.types.is_string_dtype(cleaned):
        text = cleaned.astype("string").str.strip()
        cleaned = text.mask(text.str.lower().isin(MISSING_TEXT))
    return cleaned


def _value_kind(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return "numeric"

    text = str(value).strip()
    if not text:
        return "missing"
    try:
        float(text.replace(",", ""))
        return "numeric"
    except ValueError:
        pass
    if DATE_PATTERN.fullmatch(text) or not pd.isna(pd.to_datetime(text, errors="coerce")):
        return "datetime"
    return "string"


def _infer_column_type(series: pd.Series) -> Tuple[str, int, Dict[str, int]]:
    values = _clean_values(series).dropna()
    if values.empty:
        return "unknown", 0, {}

    kinds = values.map(_value_kind)
    counts = {str(kind): int(count) for kind, count in kinds.value_counts().items()}
    dominant_kind = max(counts, key=counts.get)
    consistency_count = len(values) - counts[dominant_kind]
    return dominant_kind, consistency_count, counts


def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    return {column: _infer_column_type(df[column])[0] for column in df.columns}


def detect_semantic_types(df: pd.DataFrame) -> Dict[str, str]:
    semantic_types: Dict[str, str] = {}
    for column in df.columns:
        name = re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")
        values = _clean_values(df[column]).dropna().astype(str)
        email_ratio = values.map(lambda value: bool(EMAIL_PATTERN.fullmatch(value))).mean() if len(values) else 0
        if "email" in name or email_ratio >= 0.8:
            semantic_types[column] = "email"
        elif any(token in name for token in ("phone", "mobile", "telephone")):
            semantic_types[column] = "phone"
        elif "name" in name:
            semantic_types[column] = "name"
        elif any(token in name for token in ("date", "time", "timestamp")):
            semantic_types[column] = "date"
        elif name == "id" or name.endswith("_id") or name.startswith("id_"):
            semantic_types[column] = "identifier"
        elif len(values) and values.map(lambda value: bool(DATE_PATTERN.fullmatch(value))).mean() >= 0.8:
            semantic_types[column] = "date"
        else:
            semantic_types[column] = "unknown"
    return semantic_types


def detect_mixed_type_columns(df: pd.DataFrame) -> List[str]:
    mixed_types: List[str] = []
    for column in df.columns:
        non_null = _clean_values(df[column]).dropna()
        if non_null.empty:
            continue
        kinds = {_value_kind(value) for value in non_null}
        if len(kinds) > 1:
            mixed_types.append(column)
    return mixed_types


def detect_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    return {column: int(_clean_values(df[column]).isna().sum()) for column in df.columns}


def detect_pii_columns(df: pd.DataFrame) -> List[str]:
    pii_candidates: List[str] = []
    for column in df.columns:
        semantic_type = detect_semantic_types(df)[column]
        if semantic_type in {"email", "phone", "name"}:
            pii_candidates.append(column)
    return pii_candidates


def _column_profiles(df: pd.DataFrame, semantic_types: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    for column in df.columns:
        series = _clean_values(df[column])
        non_null = series.dropna()
        inferred_type, inconsistent_count, kind_counts = _infer_column_type(series)
        profile: Dict[str, Any] = {
            "inferred_type": inferred_type,
            "semantic_type": semantic_types[column],
            "row_count": int(len(series)),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": round(float(series.isna().mean() * 100), 2) if len(series) else 0.0,
            "unique_count": int(non_null.nunique()),
            "cardinality_percentage": round(float(non_null.nunique() / len(non_null) * 100), 2) if len(non_null) else 0.0,
            "type_counts": kind_counts,
            "inconsistent_value_count": inconsistent_count,
            "top_values": {str(key): int(value) for key, value in non_null.astype(str).value_counts().head(10).items()},
        }

        if inferred_type == "numeric":
            numeric = pd.to_numeric(non_null.astype(str).str.replace(",", "", regex=False), errors="coerce").dropna()
            if len(numeric):
                q1, q3 = numeric.quantile([0.25, 0.75])
                lower_bound = q1 - 1.5 * (q3 - q1)
                upper_bound = q3 + 1.5 * (q3 - q1)
                profile["numeric_statistics"] = {
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                    "mean": float(numeric.mean()),
                    "median": float(numeric.median()),
                    "std_dev": float(numeric.std()) if len(numeric) > 1 else 0.0,
                    "q1": float(q1),
                    "q3": float(q3),
                    "outlier_count_iqr": int(((numeric < lower_bound) | (numeric > upper_bound)).sum()),
                }
            else:
                profile["numeric_statistics"] = {}
        elif inferred_type == "datetime":
            parsed_dates = pd.to_datetime(non_null, errors="coerce")
            profile["date_statistics"] = {
                "min": parsed_dates.min().isoformat() if parsed_dates.notna().any() else None,
                "max": parsed_dates.max().isoformat() if parsed_dates.notna().any() else None,
                "invalid_count": int(parsed_dates.isna().sum()),
            }

        profiles[column] = profile
    return profiles


def _check_rules(
    df: pd.DataFrame,
    column_profiles: Dict[str, Dict[str, Any]],
    semantic_types: Dict[str, str],
) -> Dict[str, Any]:
    findings: List[Dict[str, Any]] = []
    suspicious_columns: List[str] = []
    mixed_columns = set(detect_mixed_type_columns(df))

    for column, profile in column_profiles.items():
        if profile["missing_percentage"] >= 50:
            findings.append({"column": column, "rule": "high_missingness", "severity": "warning", "count": profile["missing_count"]})
            suspicious_columns.append(column)
        if profile["inconsistent_value_count"]:
            findings.append({"column": column, "rule": "mixed_or_inconsistent_types", "severity": "warning", "count": profile["inconsistent_value_count"]})
            suspicious_columns.append(column)
        if column in mixed_columns and not profile["inconsistent_value_count"]:
            findings.append({"column": column, "rule": "mixed_types", "severity": "warning", "count": len(df[column].dropna())})
            suspicious_columns.append(column)
        if profile["unique_count"] == 1 and len(df) > 1:
            findings.append({"column": column, "rule": "constant_column", "severity": "warning", "count": len(df)})
            suspicious_columns.append(column)
        if semantic_types[column] == "email":
            invalid_count = int(sum(not bool(EMAIL_PATTERN.fullmatch(str(value).strip())) for value in df[column].dropna()))
            if invalid_count:
                findings.append({"column": column, "rule": "invalid_email_format", "severity": "error", "count": invalid_count})
                suspicious_columns.append(column)
        if semantic_types[column] == "date":
            valid_dates = pd.to_datetime(df[column], errors="coerce")
            invalid_count = int(valid_dates.isna().sum() - df[column].isna().sum())
            if invalid_count:
                findings.append({"column": column, "rule": "invalid_date_format", "severity": "error", "count": invalid_count})
                suspicious_columns.append(column)
        if semantic_types[column] in {"numeric", "identifier"} and "numeric_statistics" in profile:
            stats = profile["numeric_statistics"]
            if stats.get("outlier_count_iqr", 0):
                findings.append({"column": column, "rule": "numeric_outliers_iqr", "severity": "info", "count": stats["outlier_count_iqr"]})
                suspicious_columns.append(column)

    return {"findings": findings, "suspicious_columns": sorted(set(suspicious_columns))}


def _generate_visualizations(df: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    missing = pd.DataFrame({column: _clean_values(df[column]).isna() for column in df.columns})
    figure, axis = plt.subplots(figsize=(max(7, min(len(df.columns) * 0.55, 18)), 4.5))
    if missing.size:
        axis.imshow(missing.iloc[:1000].astype(int), aspect="auto", interpolation="nearest", cmap="YlOrRd", vmin=0, vmax=1)
        axis.set_xticks(range(len(missing.columns)), labels=missing.columns, rotation=45, ha="right")
        axis.set_ylabel("Rows (first 1,000)")
        axis.set_title("Missing-value matrix")
    else:
        axis.text(0.5, 0.5, "No rows or columns", ha="center", va="center")
    figure.tight_layout()
    path = output_dir / "missing_value_heatmap.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths["missing_heatmap"] = str(path)

    numeric_columns = [column for column in df.columns if pd.to_numeric(_clean_values(df[column]), errors="coerce").notna().sum() > 0]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for column in numeric_columns[:8]:
        values = pd.to_numeric(_clean_values(df[column]), errors="coerce").dropna()
        if len(values):
            axis.hist(values, bins=min(30, max(5, int(math.sqrt(len(values))))), alpha=0.45, label=str(column))
            plotted = True
    if plotted:
        axis.set_title("Numeric distributions and outliers")
        axis.legend(loc="best", fontsize="small")
    else:
        axis.text(0.5, 0.5, "No numeric columns", ha="center", va="center")
    figure.tight_layout()
    path = output_dir / "outlier_distributions.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths["outlier_distribution"] = str(path)

    numeric_frame = pd.DataFrame({column: pd.to_numeric(_clean_values(df[column]), errors="coerce") for column in numeric_columns})
    corr = numeric_frame.corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(max(6, min(len(corr.columns) * 0.8, 14)), max(4, min(len(corr.columns) * 0.7, 12))))
    if not corr.empty:
        image = axis.imshow(corr.fillna(0), vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=45, ha="right")
        axis.set_yticks(range(len(corr.index)), labels=corr.index)
        axis.set_title("Numeric correlation heatmap")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    else:
        axis.text(0.5, 0.5, "No numeric columns", ha="center", va="center")
    figure.tight_layout()
    path = output_dir / "correlation_heatmap.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths["correlation_heatmap"] = str(path)

    return paths


def profile_dataset(file_path: str | Path, output_dir: str | Path | None = None) -> Dict[str, Any]:
    file_path = Path(file_path)
    df = pd.read_csv(file_path)
    semantic_types = detect_semantic_types(df)
    column_types = infer_column_types(df)
    column_profiles = _column_profiles(df, semantic_types)
    rule_results = _check_rules(df, column_profiles, semantic_types)
    missing_values = detect_missing_values(df)
    missing_total = sum(missing_values.values())
    cell_count = len(df) * len(df.columns)
    numeric_columns = [column for column in df.columns if column_profiles[column]["inferred_type"] == "numeric"]
    numeric_frame = pd.DataFrame({column: pd.to_numeric(_clean_values(df[column]), errors="coerce") for column in numeric_columns})
    correlations = numeric_frame.corr(numeric_only=True).round(6).where(lambda values: values.notna(), None).to_dict()
    visualization_dir = Path(output_dir) if output_dir is not None else file_path.parent / "profiling_visualizations"
    visualizations = _generate_visualizations(df, visualization_dir)
    pii_columns = detect_pii_columns(df)

    return {
        "schema_version": "1.0",
        "dataset_name": file_path.name,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "metadata": {
            "column_types": column_types,
            "semantic_types": semantic_types,
            "mixed_type_columns": detect_mixed_type_columns(df),
            "pii_columns": pii_columns,
        },
        "column_profiles": column_profiles,
        "missing_values": missing_values,
        "duplicate_rows": int(df.duplicated().sum()),
        "profiling": {
            "cardinality": {column: column_profiles[column]["unique_count"] for column in df.columns},
            "distributions": {column: column_profiles[column]["top_values"] for column in df.columns},
            "correlation_matrix": correlations,
        },
        "quality_checks": {
            "suspicious_columns": rule_results["suspicious_columns"],
            "findings": rule_results["findings"],
            "null_percentage": round(missing_total / cell_count * 100, 2) if cell_count else 0.0,
        },
        "visualizations": visualizations,
        "summary": {
            "total_nulls": int(missing_total),
            "total_duplicates": int(df.duplicated().sum()),
            "suspicious_columns_count": len(rule_results["suspicious_columns"]),
            "pii_columns_count": len(pii_columns),
        },
    }
