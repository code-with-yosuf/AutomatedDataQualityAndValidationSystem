from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import auc, roc_curve

from .anomaly import DETECTORS, run_detector, supports

MIN_EVAL_ROWS = 30
MAX_ROC_POINTS = 100


def inject_anomalies(X: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Shift a few rows by 5-8 standard deviations on up to two features; return (X, labels)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    count = max(3, round(0.05 * n))
    rows = rng.choice(n, size=count, replace=False)
    shifted = X.copy()
    for row in rows:
        for col in rng.choice(d, size=min(d, 2), replace=False):
            shifted[row, col] += rng.choice([-1, 1]) * rng.uniform(5, 8) * (X[:, col].std() or 1.0)
    labels = np.zeros(n, dtype=bool)
    labels[rows] = True
    return shifted, labels


def _metrics(labels: np.ndarray, scores: np.ndarray, flags: np.ndarray) -> Dict[str, Any]:
    tp = int((labels & flags).sum())
    fp = int((~labels & flags).sum())
    fn = int((labels & ~flags).sum())
    tn = int((~labels & ~flags).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr, tpr, _ = roc_curve(labels, scores)
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), MAX_ROC_POINTS)).astype(int))
    return {
        "confusion_matrix": {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(float(auc(fpr, tpr)), 4),
        "roc_curve": {"fpr": [round(float(v), 4) for v in fpr[keep]], "tpr": [round(float(v), 4) for v in tpr[keep]]},
    }


def _plot_roc(results: Dict[str, Dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 5))
    for name, metrics in results.items():
        curve = metrics["roc_curve"]
        axis.plot(curve["fpr"], curve["tpr"], label=f"{name} (AUC={metrics['roc_auc']:.2f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC curves")
    axis.legend(loc="lower right", fontsize="small")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def evaluate_detectors(
    X: np.ndarray,
    labels: Optional[Sequence[Any]] = None,
    contamination: Union[str, float] = "auto",
    seed: int = 42,
    output_dir: Optional[Union[str, Path]] = None,
    extra: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, Any]:
    """Score detectors against known bad rows, or against injected anomalies when no labels exist.

    `extra` maps a name to (scores, flags) for other row-level signals and is only used with provided labels.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] == 0 or X.shape[0] < MIN_EVAL_ROWS:
        return {"status": "skipped", "reason": f"needs at least {MIN_EVAL_ROWS} rows and one numeric feature"}

    if labels is None:
        X_eval, y = inject_anomalies(X, seed)
        mode = "synthetic_injection"
    else:
        y = np.asarray(labels).astype(bool)
        if len(y) != len(X):
            raise ValueError("labels must have one entry per row")
        if y.all() or not y.any():
            return {"status": "skipped", "reason": "labels must contain both good and bad rows"}
        X_eval, mode = X, "provided_labels"

    results: Dict[str, Dict[str, Any]] = {}
    for name in DETECTORS:
        if supports(name, X_eval):
            continue
        scores, flags = run_detector(name, X_eval, contamination, seed)
        results[name] = _metrics(y, scores, flags)
    if mode == "provided_labels":
        for name, (scores, flags) in (extra or {}).items():
            results[name] = _metrics(y, np.asarray(scores, dtype=float), np.asarray(flags, dtype=bool))

    report: Dict[str, Any] = {
        "status": "ok",
        "mode": mode,
        "rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "detectors": results,
    }
    if output_dir is not None and results:
        path = Path(output_dir) / "roc_curves.png"
        _plot_roc(results, path)
        report["roc_plot"] = str(path)
    return report
