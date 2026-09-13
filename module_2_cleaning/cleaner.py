from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def clean_dataset(file_path: str | Path, profile: Dict[str, Any], output_dir: str | Path) -> Path:
    df = pd.read_csv(file_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].astype(str).str.strip()
        df[column] = df[column].replace({"nan": None, "NaN": None, "None": None, "": None})

    df = df.drop_duplicates()
    numeric_columns = [col for col, dtype in profile["column_types"].items() if dtype == "numeric"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].replace({"None": None, "nan": None})
        df[column] = df[column].fillna(method="ffill")

    cleaned_path = output_dir / "cleaned_data.csv"
    df.to_csv(cleaned_path, index=False)
    return cleaned_path
