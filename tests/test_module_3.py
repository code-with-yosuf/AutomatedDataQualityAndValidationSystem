import json

import numpy as np
import pandas as pd
import pytest

from module_3_validation.anomaly import detect_anomalies, numeric_feature_matrix
from module_3_validation.drift import detect_drift
from module_3_validation.error_detection import detect_errors
from module_3_validation.evaluation import evaluate_detectors
from module_3_validation.nlp_classifier import classify_columns
from module_3_validation.rules import column_roles, validate_rules
from module_3_validation.validator import validate_dataset


def checks(findings):
    return {(f["check"], f["column"]) for f in findings}


def test_rules_flag_invalid_email_phone_postcode_range_and_categories():
    df = pd.DataFrame(
        {
            "email": ["a@x.com", "not-an-email", "b@x.com", "c@x.com"],
            "phone": ["555-123-4567", "12", "555-222-3333", "555-444-5555"],
            "zip": ["12345", "SW1A 1AA", "ABCDE", "90210"],
            "age": [30, 150, -2, 41],
            "status": ["active", "Active", "active", "inactive"] * 1,
        }
    )
    roles = column_roles(list(df.columns))

    found = checks(validate_rules(df, roles, {"allowed_values": {"status": ["active", "inactive"]}}))

    assert ("invalid_email", "email") in found
    assert ("invalid_phone", "phone") in found
    assert ("invalid_postcode", "zip") in found
    assert ("numeric_out_of_range", "age") in found
    assert ("value_not_allowed", "status") in found


def test_rules_flag_inconsistent_category_spelling():
    df = pd.DataFrame({"segment": ["retail"] * 6 + ["Retail", "wholesale"] * 2})

    found = checks(validate_rules(df, column_roles(["segment"])))

    assert ("categorical_inconsistency", "segment") in found


def test_ml_detectors_flag_injected_outliers():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(50, 5, 200), "b": rng.normal(100, 10, 200)})
    df.loc[[10, 77, 150], ["a", "b"]] = [[500, 900], [-300, -400], [450, 800]]

    report, row_scores, consensus, _ = detect_anomalies(df.astype(str), column_roles(["a", "b"]))

    assert report["status"] == "ok"
    assert set(report["methods"]) == {"isolation_forest", "local_outlier_factor", "autoencoder"}
    assert consensus[[10, 77, 150]].all()
    assert consensus.sum() <= 15
    assert row_scores[[10, 77, 150]].min() > 0


def test_anomaly_detection_skips_tiny_datasets():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).astype(str)

    report, *_ = detect_anomalies(df, column_roles(["a", "b"]))

    assert report["status"] == "skipped"


def test_feature_matrix_ignores_identifier_and_text_columns():
    df = pd.DataFrame({"user_id": ["1", "2", "3"], "score": ["1", "5", "9"], "city": ["a", "b", "c"]})

    _, columns = numeric_feature_matrix(df, column_roles(list(df.columns)))

    assert columns == ["score"]


def test_classifier_detects_mislabeled_column_and_off_type_values():
    emails = [f"user{i}@example.com" for i in range(20)]
    phones = [f"555-{200 + i}-{1000 + i}" for i in range(20)]
    df = pd.DataFrame({"email": phones, "contact": emails})
    df.loc[3, "contact"] = "555-310-4422"
    roles = {"email": "email", "contact": None}

    classification, findings = classify_columns(df, roles)

    assert classification["email"]["predicted_class"] == "phone"
    assert classification["email"]["status"] == "mislabeled"
    assert classification["contact"]["predicted_class"] == "email"
    assert ("mislabeled_column", "email") in checks(findings)
    off_type = [f for f in findings if f["check"] == "semantic_inconsistency" and f["column"] == "contact"]
    assert off_type and off_type[0]["example_rows"] == [3]


def test_classifier_accepts_correctly_named_columns():
    df = pd.DataFrame(
        {
            "email": ["zed@corp.org", "mia@mail.com", "ola@site.net"],
            "phone": ["(555) 123-4567", "555-987-6543", "+1 555 222 3333"],
            "signup_date": ["2024-01-05", "2023-11-30", "2022-07-19"],
        }
    )

    classification, findings = classify_columns(df, column_roles(list(df.columns)))

    assert all(entry["status"] == "ok" for entry in classification.values())
    assert not [f for f in findings if f["check"] == "mislabeled_column"]


def test_error_detection_flags_impossible_and_suspicious_values():
    df = pd.DataFrame(
        {
            "order_id": ["A1", "A2", "A2", "A4"],
            "order_date": ["2024-01-01", "2999-01-01", "1850-05-05", "2024-03-03"],
            "name": ["Alice", "test", "Bob", "Carol"],
            "phone": ["555-123-4567", "0000000000", "555-222-3333", "555-444-5555"],
            "empty": [None] * 4,
        }
    )

    found = checks(detect_errors(df, column_roles(list(df.columns))))

    assert ("duplicate_identifier", "order_id") in found
    assert ("future_date", "order_date") in found
    assert ("implausible_date", "order_date") in found
    assert ("suspicious_pattern", "name") in found
    assert ("suspicious_pattern", "phone") in found
    assert ("empty_column", "empty") in found


def test_error_detection_flags_end_before_start_and_extreme_values():
    df = pd.DataFrame(
        {
            "start_date": ["2024-01-01", "2024-02-01"] * 6,
            "end_date": ["2024-01-05", "2024-01-15"] * 6,
            "amount": [10, 11, 9, 10, 12, 10, 9, 11, 10, 10, 11, 9_999_999],
        }
    ).astype(str)

    found = checks(detect_errors(df, column_roles(list(df.columns))))

    assert ("end_before_start", "end_date") in found
    assert ("extreme_value", "amount") in found


def test_drift_detects_shifted_numeric_and_categorical_columns():
    rng = np.random.default_rng(1)
    reference = pd.DataFrame({"amount": rng.normal(100, 10, 200), "tier": ["gold"] * 150 + ["silver"] * 50}).astype(str)
    current = pd.DataFrame({"amount": rng.normal(160, 10, 200), "tier": ["gold"] * 40 + ["silver"] * 160}).astype(str)

    report, findings = detect_drift(current, reference)

    assert report["columns"]["amount"]["drift"] == "high"
    assert report["columns"]["tier"]["drift"] == "high"
    assert {f["check"] for f in findings} == {"data_drift"}


def test_drift_is_quiet_for_stable_data_and_skips_small_data():
    rng = np.random.default_rng(2)
    stable = pd.DataFrame({"amount": rng.normal(100, 10, 400)}).astype(str)

    report, findings = detect_drift(stable.iloc[200:], stable.iloc[:200])
    skipped, _ = detect_drift(stable.head(10))

    assert report["columns"]["amount"]["drift"] == "none" and not findings
    assert skipped["status"] == "skipped"


def test_evaluation_reports_metrics_and_roc_plot_for_injected_anomalies(tmp_path):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 4))

    report = evaluate_detectors(X, output_dir=tmp_path)

    assert report["mode"] == "synthetic_injection"
    for metrics in report["detectors"].values():
        assert set(metrics["confusion_matrix"]) == {"true_positive", "false_positive", "true_negative", "false_negative"}
        assert 0 <= metrics["precision"] <= 1 and 0 <= metrics["recall"] <= 1
    assert report["detectors"]["isolation_forest"]["roc_auc"] > 0.9
    assert (tmp_path / "roc_curves.png").stat().st_size > 0


def test_evaluation_with_provided_labels_includes_combined_validator():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(100, 3))
    labels = np.zeros(100, dtype=bool)
    labels[:5] = True
    X[:5] += 8

    report = evaluate_detectors(X, labels, extra={"validator_combined": (labels.astype(float), labels)})

    assert report["mode"] == "provided_labels"
    assert report["detectors"]["validator_combined"]["confusion_matrix"]["true_positive"] == 5
    assert report["detectors"]["isolation_forest"]["recall"] >= 0.8


def test_evaluation_skips_small_data_and_single_class_labels():
    assert evaluate_detectors(np.zeros((5, 2)))["status"] == "skipped"
    assert evaluate_detectors(np.random.default_rng(0).normal(size=(40, 2)), [False] * 40)["status"] == "skipped"


def make_dataset(rows=120, seed=5):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "customer_id": [f"C-{1000 + i}" for i in range(rows)],
            "email": [f"user{i}@example.com" for i in range(rows)],
            "age": rng.integers(20, 70, rows),
            "amount_spent": rng.normal(200, 30, rows).round(2),
            "visits": rng.normal(12, 3, rows).round(1),
        }
    )


def test_validate_dataset_clean_data_passes_and_report_is_json_safe(tmp_path):
    path = tmp_path / "clean.csv"
    make_dataset().to_csv(path, index=False)

    report = validate_dataset(path, output_dir=tmp_path / "viz")

    json.dumps(report, allow_nan=False)
    assert report["status"] == "pass" and report["passed_checks"] is True
    assert report["scores"]["health_score"] >= 90
    assert {"issues", "row_count", "column_count", "dataset_name"} <= set(report)
    assert report["anomaly_detection"]["status"] == "ok"
    assert report["evaluation"]["mode"] == "synthetic_injection"
    assert (tmp_path / "viz" / "roc_curves.png").exists()


def test_validate_dataset_dirty_data_fails_with_lower_health(tmp_path):
    clean = make_dataset()
    dirty = clean.copy()
    dirty["email"] = dirty["email"].astype(object)
    dirty.loc[:59, "email"] = "broken"
    dirty.loc[:39, "age"] = 999
    clean_path, dirty_path = tmp_path / "clean.csv", tmp_path / "dirty.csv"
    clean.to_csv(clean_path, index=False)
    dirty.to_csv(dirty_path, index=False)

    good = validate_dataset(clean_path)
    bad = validate_dataset(dirty_path)

    assert bad["status"] == "fail"
    assert bad["scores"]["health_score"] < good["scores"]["health_score"]
    assert bad["scores"]["error_severity"] > good["scores"]["error_severity"]
    assert bad["issues"] and all(f["severity"] in {"high", "medium", "low"} for f in bad["findings"])


def test_validate_dataset_evaluates_against_provided_labels(tmp_path):
    df = make_dataset()
    labels = np.zeros(len(df), dtype=bool)
    labels[:6] = True
    df.loc[:5, ["amount_spent", "visits"]] = [[5000, 300]] * 6
    path = tmp_path / "labelled.csv"
    df.to_csv(path, index=False)

    report = validate_dataset(path, labels=labels)

    assert report["evaluation"]["mode"] == "provided_labels"
    assert report["evaluation"]["detectors"]["validator_combined"]["recall"] >= 0.8


def test_validate_dataset_uses_reference_for_drift_and_custom_ranges(tmp_path):
    reference = make_dataset(seed=6)
    current = make_dataset(seed=7)
    current["amount_spent"] += 150
    reference.to_csv(tmp_path / "reference.csv", index=False)
    current.to_csv(tmp_path / "current.csv", index=False)

    report = validate_dataset(
        tmp_path / "current.csv",
        reference=tmp_path / "reference.csv",
        rules={"numeric_ranges": {"visits": (0, 10)}},
    )

    assert report["drift"]["columns"]["amount_spent"]["drift"] == "high"
    assert ("data_drift", "amount_spent") in checks(report["findings"])
    assert ("numeric_out_of_range", "visits") in checks(report["findings"])


def test_validate_dataset_handles_empty_input_and_bad_labels(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("a,b\n", encoding="utf-8")
    path = tmp_path / "data.csv"
    make_dataset(20).to_csv(path, index=False)

    assert validate_dataset(empty)["status"] == "fail"
    with pytest.raises(ValueError):
        validate_dataset(path, labels=[True, False])
