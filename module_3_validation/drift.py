from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .rules import make_dataset_finding, numeric_share

MIN_DRIFT_ROWS = 20
PSI_MODERATE = 0.1
PSI_HIGH = 0.25


def _psi(reference_share: np.ndarray, current_share: np.ndarray, ref_n: int, cur_n: int) -> float:
    ref = np.clip(reference_share, 1e-4, None)
    cur = np.clip(current_share, 1e-4, None)
    raw = float(np.sum((cur - ref) * np.log(cur / ref)))
    # Without drift PSI is still about (bins - 1) * (1/ref_n + 1/cur_n); subtract that sampling noise.
    return max(0.0, raw - (len(ref) - 1) * (1 / ref_n + 1 / cur_n))


def _numeric_psi(reference: pd.Series, current: pd.Series) -> Optional[float]:
    ref = reference.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    cur = current.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return None
    bins = int(np.clip(min(len(ref), len(cur)) // 10, 3, 10))
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    inner = edges[1:-1]
    ref_share = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=len(edges) - 1) / len(ref)
    cur_share = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=len(edges) - 1) / len(cur)
    return _psi(ref_share, cur_share, len(ref), len(cur))


def _categorical_psi(reference: pd.Series, current: pd.Series) -> Optional[float]:
    ref = reference.dropna().astype(str).str.strip().str.lower()
    cur = current.dropna().astype(str).str.strip().str.lower()
    if ref.empty or cur.empty:
        return None
    categories = sorted(set(ref) | set(cur))
    ref_share = ref.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    cur_share = cur.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    return _psi(ref_share, cur_share, len(ref), len(cur))


def detect_drift(
    df: pd.DataFrame,
    reference: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compare column distributions with a reference dataset.

    Without a reference the first and second half of rows are compared for information only; row order is often
    arbitrary, so that mode never produces findings.
    """
    if reference is None:
        half = len(df) // 2
        if half < MIN_DRIFT_ROWS:
            return {"status": "skipped", "reason": f"needs a reference dataset or at least {2 * MIN_DRIFT_ROWS} rows"}, []
        ref, cur, mode = df.iloc[:half], df.iloc[half:], "first_half_vs_second_half"
    else:
        ref, cur, mode = reference, df, "reference_dataset"
        if len(ref) < MIN_DRIFT_ROWS or len(cur) < MIN_DRIFT_ROWS:
            return {"status": "skipped", "reason": f"each side needs at least {MIN_DRIFT_ROWS} rows"}, []

    findings: List[Dict[str, Any]] = []
    columns: Dict[str, Any] = {}

    missing = sorted(str(c) for c in set(ref.columns) - set(cur.columns))
    added = sorted(str(c) for c in set(cur.columns) - set(ref.columns))
    if missing or added:
        findings.append(make_dataset_finding(
            "schema_drift", None, "medium",
            f"Columns differ from reference (missing: {missing or 'none'}, new: {added or 'none'})",
        ))

    for column in [c for c in cur.columns if c in ref.columns]:
        ref_numeric, ref_share = numeric_share(ref[column])
        cur_numeric, cur_share = numeric_share(cur[column])
        if ref_share >= 0.8 and cur_share >= 0.8:
            psi, kind = _numeric_psi(ref_numeric, cur_numeric), "numeric"
        else:
            unique = ref[column].nunique()
            if unique > 50 or (ref[column].notna().sum() and unique / ref[column].notna().sum() > 0.5):
                continue
            psi, kind = _categorical_psi(ref[column], cur[column]), "categorical"
        if psi is None:
            continue

        level = "high" if psi >= PSI_HIGH else "moderate" if psi >= PSI_MODERATE else "none"
        columns[str(column)] = {"type": kind, "psi": round(psi, 4), "drift": level}
        if level != "none":
            findings.append(make_dataset_finding(
                "data_drift", column, "high" if level == "high" else "medium",
                f"Column '{column}' distribution drifted from the reference (PSI={psi:.3f})",
            ))

    return {
        "status": "ok",
        "mode": mode,
        "thresholds": {"moderate": PSI_MODERATE, "high": PSI_HIGH},
        "columns": columns,
    }, findings if reference is not None else []
