from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .anomaly import detect_anomalies
from .drift import detect_drift
from .error_detection import detect_errors
from .evaluation import evaluate_detectors
from .nlp_classifier import classify_columns
from .rules import column_roles, make_finding, validate_rules
from .scoring import score_dataset

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _tag(findings: List[Dict[str, Any]], category: str) -> List[Dict[str, Any]]:
    for finding in findings:
        finding["category"] = category
    return findings


def _public(finding: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in finding.items() if key != "_mask"}


def validate_dataset(
    file_path: Union[str, Path],
    profile: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    reference: Optional[Union[str, Path, pd.DataFrame]] = None,
    rules: Optional[Dict[str, Any]] = None,
    labels: Optional[Sequence[Any]] = None,
    contamination: Union[str, float] = "auto",
    pass_threshold: float = 80.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Validate a cleaned CSV and return the validation report.

    `rules` may hold `numeric_ranges` ({column: (min, max)}), `allowed_values` ({column: [...]}) and
    `postcode_countries`. `labels` marks known-bad rows so detectors are evaluated on real data.
    """
    path = Path(file_path)
    df = pd.read_csv(path, dtype=str)
    n = len(df)
    report: Dict[str, Any] = {"dataset_name": path.name, "row_count": int(n), "column_count": int(len(df.columns))}

    if n == 0 or len(df.columns) == 0:
        report.update(
            status="fail",
            passed_checks=False,
            issues=["Dataset is empty."],
            scores={"error_severity": 100.0, "anomaly_severity": 0.0, "structural_penalty": 0.0, "health_score": 0.0, "pass_threshold": pass_threshold},
            findings=[],
        )
        return report

    if labels is not None and len(labels) != n:
        raise ValueError("labels must have one entry per row")
    if isinstance(reference, (str, Path)):
        reference = pd.read_csv(reference, dtype=str)

    roles = column_roles(list(df.columns), profile)
    classification, classifier_findings = classify_columns(df, roles)
    drift_report, drift_findings = detect_drift(df, reference)
    anomaly_report, anomaly_scores, consensus, features = detect_anomalies(df, roles, contamination, seed)

    findings = (
        _tag(validate_rules(df, roles, rules), "rule")
        + _tag(detect_errors(df, roles), "ai_error")
        + _tag(classifier_findings, "nlp_classifier")
        + _tag(drift_findings, "drift")
    )
    anomaly_finding = make_finding("anomalous_rows", None, "medium", consensus, "Rows flagged as anomalous by multiple detectors", n)
    if anomaly_finding:
        findings += _tag([anomaly_finding], "anomaly")

    scores, row_error_weight = score_dataset(n, findings, anomaly_scores if anomaly_report["status"] == "ok" else None)
    scores["pass_threshold"] = pass_threshold

    extra = None
    if labels is not None:
        combined = np.maximum(row_error_weight, anomaly_scores)
        extra = {"validator_combined": (combined, combined >= 0.5)}
    evaluation = evaluate_detectors(
        features, labels, contamination, seed,
        Path(output_dir) if output_dir is not None else None, extra,
    )

    findings.sort(key=lambda f: (SEVERITY_ORDER[f["severity"]], f.get("category", ""), f["check"]))
    healthy = scores["health_score"] >= pass_threshold
    report.update(
        status="pass" if healthy else "fail",
        passed_checks=healthy,
        issues=[f["message"] for f in findings],
        scores=scores,
        findings=[_public(f) for f in findings],
        column_classification=classification,
        anomaly_detection=anomaly_report,
        drift=drift_report,
        evaluation=evaluation,
    )
    return report
