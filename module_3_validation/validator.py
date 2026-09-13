from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def validate_dataset(file_path: str | Path) -> Dict[str, Any]:
    df = pd.read_csv(file_path)
    issues: List[str] = []

    if df.empty:
        issues.append("Dataset is empty.")

    for column in df.columns:
        if df[column].isnull().all():
            issues.append(f"Column '{column}' is entirely null.")

    numeric_columns = df.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        if (df[column] < 0).any():
            issues.append(f"Column '{column}' contains negative values.")

    if "email" in str(df.columns).lower():
        email_cols = [col for col in df.columns if 'email' in col.lower()]
        for col in email_cols:
            invalid = df[col].dropna().str.contains(r"@", na=False)
            if (~invalid).any():
                issues.append(f"Column '{col}' contains invalid email values.")

    return {
        "dataset_name": Path(file_path).name,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "passed_checks": len(issues) == 0,
    }
