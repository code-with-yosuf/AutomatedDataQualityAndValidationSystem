from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .rules import FREE_FORM_ROLES, numeric_share

MIN_ANOMALY_ROWS = 10
DETECTORS = ("isolation_forest", "local_outlier_factor", "autoencoder")
Contamination = Union[str, float]


def numeric_feature_matrix(df: pd.DataFrame, roles: Dict[Any, Optional[str]]) -> Tuple[np.ndarray, List[Any]]:
    columns: Dict[Any, pd.Series] = {}
    for column in df.columns:
        if roles.get(column) in FREE_FORM_ROLES:
            continue
        numeric, share = numeric_share(df[column])
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        if share < 0.8 or numeric.nunique() < 2:
            continue
        columns[column] = numeric.fillna(numeric.median())
    if not columns:
        return np.empty((len(df), 0)), []
    matrix = StandardScaler().fit_transform(pd.DataFrame(columns).to_numpy(dtype=float))
    return matrix, list(columns)


def supports(name: str, X: np.ndarray) -> Optional[str]:
    """Return a reason the detector cannot run on X, or None when it can."""
    if name == "autoencoder" and X.shape[1] < 2:
        return "autoencoder needs at least 2 numeric features"
    return None


def _robust_flags(values: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median)) * 1.4826
    return (values - median) / mad > threshold if mad > 0 else np.zeros(len(values), dtype=bool)


def run_detector(name: str, X: np.ndarray, contamination: Contamination = "auto", seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Return (anomaly score where higher is more anomalous, boolean flags).

    With contamination="auto", Isolation Forest's own cutoff flags a large share of normal rows, so scores are
    thresholded with a robust z-score instead.
    """
    n, d = X.shape
    if name == "isolation_forest":
        model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
        flags = model.fit_predict(X) == -1
        scores = -model.score_samples(X)
        return scores, _robust_flags(scores) if contamination == "auto" else flags

    if name == "local_outlier_factor":
        model = LocalOutlierFactor(n_neighbors=max(1, min(20, n - 1)), contamination=contamination)
        flags = model.fit_predict(X) == -1
        return -model.negative_outlier_factor_, flags

    if name == "autoencoder":
        width = max(2, d * 2)
        model = MLPRegressor(hidden_layer_sizes=(width, max(1, d // 2), width), activation="tanh", max_iter=800, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X, X)
        errors = ((X - model.predict(X)) ** 2).mean(axis=1)
        if contamination == "auto":
            flags = _robust_flags(np.log(errors + 1e-12))
        else:
            flags = errors >= np.quantile(errors, 1 - float(contamination))
        return errors, flags

    raise ValueError(f"Unknown detector '{name}'")


def detect_anomalies(
    df: pd.DataFrame,
    roles: Dict[Any, Optional[str]],
    contamination: Contamination = "auto",
    seed: int = 42,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Return (report, per-row vote share, consensus mask, feature matrix)."""
    n = len(df)
    X, features = numeric_feature_matrix(df, roles)
    report: Dict[str, Any] = {"status": "skipped", "features": [str(c) for c in features], "methods": {}}
    row_scores = np.zeros(n)
    consensus = np.zeros(n, dtype=bool)

    if n < MIN_ANOMALY_ROWS:
        report["reason"] = f"fewer than {MIN_ANOMALY_ROWS} rows"
        return report, row_scores, consensus, X
    if not features:
        report["reason"] = "no numeric features"
        return report, row_scores, consensus, X

    votes = np.zeros(n)
    ran = 0
    for name in DETECTORS:
        reason = supports(name, X)
        if reason:
            report["methods"][name] = {"status": "skipped", "reason": reason}
            continue
        _, flags = run_detector(name, X, contamination, seed)
        votes += flags
        ran += 1
        report["methods"][name] = {
            "status": "ok",
            "flagged_count": int(flags.sum()),
            "flagged_rows": [int(i) for i in np.flatnonzero(flags)[:100]],
        }

    min_votes = min(2, ran)
    consensus = votes >= min_votes
    row_scores = np.where(consensus, votes / ran, 0.0)
    report.update(
        status="ok",
        consensus={
            "min_votes": min_votes,
            "flagged_count": int(consensus.sum()),
            "flagged_rows": [int(i) for i in np.flatnonzero(consensus)[:100]],
        },
    )
    return report, row_scores, consensus, X
