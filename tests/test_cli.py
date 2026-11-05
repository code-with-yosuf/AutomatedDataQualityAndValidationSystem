import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "customers.csv"
    pd.DataFrame(
        {
            "customer_id": [f"C-{1000 + i}" for i in range(12)],
            "email": [f"user{i}@example.com" for i in range(12)],
            "age": [20 + i for i in range(12)],
            "amount_spent": [100.0 + 7 * i for i in range(12)],
        }
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def strict_config(tmp_path):
    """A config under which the dataset always fails validation."""
    path = tmp_path / "strict.yaml"
    path.write_text("validation:\n  rules:\n    numeric_ranges:\n      age: [0, 1]\n", encoding="utf-8")
    return path


def test_cli_processes_input_into_output_directory(dataset, tmp_path, capsys):
    out = tmp_path / "results"

    code = pipeline.main(["--input", str(dataset), "--output", str(out), "--log-level", "ERROR"])

    assert code == pipeline.EXIT_OK
    assert "Status:           success" in capsys.readouterr().out
    for name in ("profiling_report.json", "cleaned_data.csv", "cleaning_log.json", "validation_report.json", "end_to_end_pipeline_summary.json"):
        assert (out / name).is_file(), name


def test_cli_log_level_override_controls_log_file(dataset, tmp_path):
    out = tmp_path / "results"

    pipeline.main(["--input", str(dataset), "--output", str(out), "--log-level", "error"])

    assert "INFO" not in (out / "pipeline.log").read_text(encoding="utf-8")


def test_cli_missing_input_exits_with_config_error(tmp_path, capsys):
    code = pipeline.main(["--input", str(tmp_path / "nope.csv"), "--output", str(tmp_path / "out")])

    assert code == pipeline.EXIT_CONFIG_ERROR
    assert "Input file not found" in capsys.readouterr().err


def test_cli_missing_dvc_tracked_input_suggests_dvc_pull(tmp_path, capsys):
    (tmp_path / "data.csv.dvc").write_text("outs:\n- md5: abc\n  path: data.csv\n", encoding="utf-8")

    code = pipeline.main(["--input", str(tmp_path / "data.csv"), "--output", str(tmp_path / "out")])

    assert code == pipeline.EXIT_CONFIG_ERROR
    assert "dvc pull" in capsys.readouterr().err


def test_cli_invalid_config_exits_with_config_error(dataset, tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("cleaning: {typo: 1}\n", encoding="utf-8")

    code = pipeline.main(["--config", str(bad), "--input", str(dataset)])

    assert code == pipeline.EXIT_CONFIG_ERROR
    assert "Unknown key" in capsys.readouterr().err


def test_cli_fail_on_validation_controls_exit_code(dataset, strict_config, tmp_path):
    args = ["--config", str(strict_config), "--input", str(dataset), "--output", str(tmp_path / "out"), "--log-level", "ERROR"]

    assert pipeline.main(args) == pipeline.EXIT_OK
    assert pipeline.main(args + ["--fail-on-validation"]) == pipeline.EXIT_PIPELINE_FAILED


def test_cli_reports_dvc_input_version(dataset, tmp_path, capsys):
    (tmp_path / "customers.csv.dvc").write_text("outs:\n- md5: 0123abcd\n  path: customers.csv\n", encoding="utf-8")

    pipeline.main(["--input", str(dataset), "--output", str(tmp_path / "out"), "--log-level", "ERROR"])

    assert "Input version:    0123abcd" in capsys.readouterr().out


def test_cli_runs_as_a_script_with_the_documented_arguments(dataset, tmp_path):
    out = tmp_path / "results"

    done = subprocess.run(
        [sys.executable, "pipeline.py", "--input", str(dataset), "--output", str(out)],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300,
    )

    assert done.returncode == 0, done.stderr
    assert "Status:           success" in done.stdout
    assert "Stage 'validation' finished" in done.stderr
    assert (out / "validation_report.json").is_file()


def test_legacy_run_pipeline_entrypoint_still_works(dataset, tmp_path):
    done = subprocess.run(
        [sys.executable, "run_pipeline.py", "--input", str(dataset), "--output", str(tmp_path / "out"), "--log-level", "ERROR"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300,
    )

    assert done.returncode == 0, done.stderr
