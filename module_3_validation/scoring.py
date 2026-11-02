from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SEVERITY_WEIGHT = {"high": 1.0, "medium": 0.5, "low": 0.2}
STRUCTURAL_POINTS = {"high": 10.0, "medium": 5.0, "low": 2.0}
STRUCTURAL_CAP = 30.0
ERROR_CATEGORIES = {"rule", "ai_error", "nlp_classifier"}
ERROR_WEIGHT, ANOMALY_WEIGHT = 0.6, 0.25


def score_dataset(
    n: int,
    findings: List[Dict[str, Any]],
    anomaly_row_scores: Optional[np.ndarray],
) -> Tuple[Dict[str, float], np.ndarray]:
    """Return score summary plus each row's worst error weight.

    Error severity is the severity-weighted share of rows with an error. Anomaly severity is the average share of
    detectors flagging a row. Findings without row masks (drift, mislabeled or empty columns) subtract fixed points.
    """
    row_weight = np.zeros(n)
    structural = 0.0
    for finding in findings:
        if finding.get("category") not in ERROR_CATEGORIES | {"drift"}:
            continue
        mask = finding.get("_mask")
        if mask is None:
            structural += STRUCTURAL_POINTS[finding["severity"]]
        else:
            row_weight = np.maximum(row_weight, mask * SEVERITY_WEIGHT[finding["severity"]])

    error_severity = 100 * float(row_weight.mean()) if n else 0.0
    anomaly_severity = 100 * float(anomaly_row_scores.mean()) if anomaly_row_scores is not None and n else 0.0
    structural = min(structural, STRUCTURAL_CAP)
    health = float(np.clip(100 - ERROR_WEIGHT * error_severity - ANOMALY_WEIGHT * anomaly_severity - structural, 0, 100))
    return {
        "error_severity": round(error_severity, 2),
        "anomaly_severity": round(anomaly_severity, 2),
        "structural_penalty": round(structural, 2),
        "health_score": round(health, 2),
    }, row_weight
