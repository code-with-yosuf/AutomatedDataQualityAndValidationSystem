import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import pipeline
from module_1_profiling.profiler import profile_dataset
from module_2_cleaning.cleaner import clean_dataset
from module_3_validation.validator import validate_dataset
from module_4_automation import parse_config, run_pipeline

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample_customer_data.csv"
OUTLIER_ROWS = [10, 77, 150]
DUPLICATE_COUNT = 12
BASE_ROWS = 240

FIRST = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry", "Irene", "Jack"]
LAST = ["Johnson", "Smith", "Davis", "Lee", "Brown", "White", "Green", "Miller", "Wilson", "Moore"]
CITIES = ["New York", "Chicago", "Seattle", "Miami", "Boston", "Denver"]
NOTES = ["Active customer", "Needs follow-up", "High value", "Verified", "Prefers email", "New signup"]


def build_frame(seed=11, defects=True, amount_shift=0.0):
    """Customer data with known defects at known rows; returns BASE_ROWS rows plus DUPLICATE_COUNT copies."""
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 1)
    df = pd.DataFrame(
        {
            "customer_id": [f"C-{2000 + i}" for i in range(BASE_ROWS)],
            "name": [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(BASE_ROWS)],
            "email": [f"user{i}@example.com" for i in range(BASE_ROWS)],
            "age": rng.integers(21, 70, BASE_ROWS).astype(object),
            "city": rng.choice(CITIES, BASE_ROWS).astype(object),
            "signup_date": [(start + timedelta(days=int(d))).isoformat() for d in rng.integers(0, 700, BASE_ROWS)],
            "amount_spent": (rng.normal(200, 40, BASE_ROWS) + amount_shift).round(2),
            "phone": [f"555-{rng.integers(200, 999)}-{rng.integers(1000, 9999)}" for _ in range(BASE_ROWS)],
            "notes": rng.choice(NOTES, BASE_ROWS).astype(object),
        }
    )
    if not defects:
        return df

    pool = [i for i in range(BASE_ROWS) if i not in OUTLIER_ROWS]
    rng.shuffle(pool)
    chunks, offset = [], 0
    for size in (15, 6, 10, 8, 5, 5, DUPLICATE_COUNT):
        chunks.append(pool[offset:offset + size])
        offset += size
    missing_age, invalid_age, missing_amount, missing_city, bad_date, messy_email, duplicated = chunks

    df.loc[missing_age, "age"] = None
    df.loc[invalid_age, "age"] = "unknown"
    df.loc[missing_amount, "amount_spent"] = np.nan
    df.loc[missing_city, "city"] = None
    df.loc[bad_date, "signup_date"] = "not-a-date"
    df.loc[messy_email, "email"] = df.loc[messy_email, "email"].map(lambda value: f"  {value.upper()} ")
    df.loc[OUTLIER_ROWS, "amount_spent"] = [50000.0, 48000.0, 52000.0]
    return pd.concat([df, df.iloc[duplicated]], ignore_index=True)


def config_for(csv_path, out_dir, **sections):
    return parse_config({"input": {"path": str(csv_path)}, "output": {"dir": str(out_dir)}, **sections}, base_dir=csv_path.parent)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def messy_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("messy") / "customers.csv"
    build_frame().to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def messy_run(messy_csv, tmp_path_factory):
    out = tmp_path_factory.mktemp("messy_out")
    return run_pipeline(config_for(messy_csv, out)), out


def test_profiling_finds_the_planted_defects(messy_run):
    result, out = messy_run
    profile = load(out / "profiling_report.json")

    assert result.status == "success"
    assert profile["row_count"] == BASE_ROWS + DUPLICATE_COUNT
    assert profile["duplicate_rows"] == DUPLICATE_COUNT
    assert {"age", "signup_date"} <= set(profile["metadata"]["mixed_type_columns"])
    assert {"name", "email", "phone"} <= set(profile["metadata"]["pii_columns"])
    assert profile["missing_values"]["city"] == 8
    assert all(Path(path).is_file() for path in profile["visualizations"].values())


def test_cleaning_repairs_what_profiling_found(messy_run):
    _, out = messy_run
    cleaned = pd.read_csv(out / "cleaned_data.csv")
    log = load(out / "cleaning_log.json")

    assert len(cleaned) == BASE_ROWS
    assert cleaned.isna().sum().sum() == 0
    assert (cleaned["email"] == cleaned["email"].str.strip().str.lower()).all()
    assert pd.to_datetime(cleaned["signup_date"], format="%Y-%m-%d", errors="raise").notna().all()
    assert pd.api.types.is_numeric_dtype(cleaned["age"])
    assert log["actions"]["exact_duplicates_removed"] == DUPLICATE_COUNT
    assert log["quality_score"]["delta"] > 0
    assert log["quality_score"]["after"]["overall"] > log["quality_score"]["before"]["overall"]


def test_validation_consumes_the_cleaned_data(messy_run):
    _, out = messy_run
    report = load(out / "validation_report.json")
    findings = {(f["check"], f["column"]): f for f in report["findings"]}

    assert report["row_count"] == BASE_ROWS
    assert findings[("extreme_value", "amount_spent")]["example_rows"] == OUTLIER_ROWS
    assert not [f for f in report["findings"] if f["check"] in {"invalid_email", "invalid_date", "duplicate_rows"}]
    assert report["anomaly_detection"]["status"] == "ok"
    assert set(OUTLIER_ROWS) <= set(report["anomaly_detection"]["consensus"]["flagged_rows"])
    assert report["evaluation"]["mode"] == "synthetic_injection"
    assert Path(report["evaluation"]["roc_plot"]).is_file()


def test_summary_agrees_with_the_individual_reports(messy_run):
    result, out = messy_run
    summary = load(out / "end_to_end_pipeline_summary.json")
    validation = load(out / "validation_report.json")
    profile = load(out / "profiling_report.json")
    cleaning = load(out / "cleaning_log.json")
    metrics = {stage["name"]: stage["metrics"] for stage in summary["stages"]}

    assert [stage["name"] for stage in summary["stages"]] == ["profiling", "cleaning", "validation"]
    assert summary["validation_health_score"] == validation["scores"]["health_score"] == metrics["validation"]["health_score"]
    assert summary["validation_status"] == validation["status"]
    assert metrics["profiling"]["rows"] == profile["row_count"]
    assert metrics["cleaning"]["quality_delta"] == cleaning["quality_score"]["delta"]
    assert summary["issues_found"] == validation["issues"]
    assert all(Path(path).exists() for path in result.artifacts.values())


def test_runs_are_reproducible(messy_run, messy_csv, tmp_path):
    _, first = messy_run

    run_pipeline(config_for(messy_csv, tmp_path / "again"))

    assert (first / "cleaned_data.csv").read_bytes() == (tmp_path / "again" / "cleaned_data.csv").read_bytes()
    reports = [load(directory / "validation_report.json") for directory in (first, tmp_path / "again")]
    for report in reports:
        report["evaluation"].pop("roc_plot")
    assert reports[0] == reports[1]


def test_modules_compose_without_the_orchestrator(messy_csv, tmp_path):
    profile = profile_dataset(messy_csv, tmp_path / "viz")
    cleaned_path = clean_dataset(messy_csv, profile, tmp_path)
    report = validate_dataset(cleaned_path, profile=profile)

    assert Path(cleaned_path).name == "cleaned_data.csv"
    assert report["row_count"] == BASE_ROWS
    assert report["column_classification"]["email"]["predicted_class"] == "email"
    assert report["column_classification"]["phone"]["predicted_class"] == "phone"


def test_shifted_data_is_flagged_as_drift_against_a_reference(tmp_path):
    reference, current = tmp_path / "reference.csv", tmp_path / "current.csv"
    build_frame(seed=21, defects=False).to_csv(reference, index=False)
    build_frame(seed=11, amount_shift=150.0).to_csv(current, index=False)

    result = run_pipeline(config_for(current, tmp_path / "out", validation={"reference": str(reference)}))

    report = load(tmp_path / "out" / "validation_report.json")
    assert report["drift"]["mode"] == "reference_dataset"
    assert report["drift"]["columns"]["amount_spent"]["drift"] == "high"
    assert ("data_drift", "amount_spent") in {(f["check"], f["column"]) for f in report["findings"]}
    assert report["scores"]["structural_penalty"] >= 10
    assert result.status == "success"


def test_unreadable_input_fails_cleanly_and_still_writes_a_summary(tmp_path, capsys):
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")

    code = pipeline.main(["--input", str(empty), "--output", str(tmp_path / "out"), "--log-level", "ERROR"])

    summary = load(tmp_path / "out" / "end_to_end_pipeline_summary.json")
    assert code == pipeline.EXIT_PIPELINE_FAILED
    assert [stage["status"] for stage in summary["stages"]] == ["failed", "skipped", "skipped"]
    assert summary["status"] == "failed" and summary["pipeline_ready"] is False
    assert "Error:" in capsys.readouterr().out


@pytest.mark.skipif(not SAMPLE.is_file(), reason="run 'dvc pull' to fetch the DVC-tracked sample data")
def test_shipped_sample_runs_and_reports_its_dvc_version(tmp_path):
    code = pipeline.main(["--input", str(SAMPLE), "--output", str(tmp_path / "out"), "--log-level", "ERROR"])

    summary = load(tmp_path / "out" / "end_to_end_pipeline_summary.json")
    tracked = yaml.safe_load((ROOT / "data" / "sample_customer_data.csv.dvc").read_text(encoding="utf-8"))
    assert code == pipeline.EXIT_OK
    assert summary["input_data_version"] == tracked["outs"][0]["md5"]
