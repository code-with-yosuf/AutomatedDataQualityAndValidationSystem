import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from module_4_automation import ConfigError, load_config, parse_config, run_pipeline
from module_4_automation import orchestrator
from module_4_automation.logging_setup import LOGGER_NAME, pipeline_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def write_dataset(path, rows=12):
    pd.DataFrame(
        {
            "customer_id": [f"C-{1000 + i}" for i in range(rows)],
            "email": [f"user{i}@example.com" for i in range(rows)],
            "age": [20 + i for i in range(rows)],
            "amount_spent": [100.0 + 7 * i for i in range(rows)],
        }
    ).to_csv(path, index=False)
    return path


def make_config(tmp_path, **sections):
    raw = {"input": {"path": str(write_dataset(tmp_path / "data.csv"))}, "output": {"dir": str(tmp_path / "out")}, **sections}
    return parse_config(raw, base_dir=tmp_path)


def test_shipped_config_loads_and_resolves_paths_against_base_dir():
    config = load_config(PROJECT_ROOT / "config" / "pipeline.yaml", base_dir=PROJECT_ROOT)

    assert config.input_path == PROJECT_ROOT / "data" / "sample_customer_data.csv"
    assert config.output_dir == PROJECT_ROOT / "outputs"
    assert config.cleaning.imputation_strategy == "median"
    assert config.validation.contamination == "auto"
    assert config.logging.file == "pipeline.log"


def test_empty_config_uses_defaults(tmp_path):
    config = parse_config(None, base_dir=tmp_path)

    assert config.input_path is None
    assert config.output_dir == tmp_path / "outputs"
    assert config.validation.pass_threshold == 80.0


def test_config_accepts_explicit_values(tmp_path):
    config = parse_config(
        {
            "cleaning": {"imputation_strategy": "knn", "fuzzy_threshold": 0.8, "transformations": {"scaling": "minmax"}},
            "validation": {"contamination": 0.05, "seed": 7, "rules": {"numeric_ranges": {"age": [0, 90]}}, "reference": "ref.csv"},
            "logging": {"level": "debug", "file": None},
        },
        base_dir=tmp_path,
    )

    assert config.cleaning.transformations == {"scaling": "minmax"}
    assert config.validation.contamination == 0.05
    assert config.validation.reference == tmp_path / "ref.csv"
    assert config.logging.level == "DEBUG" and config.logging.file is None


@pytest.mark.parametrize(
    "raw",
    [
        {"unknown": 1},
        {"cleaning": {"typo": 1}},
        {"cleaning": {"imputation_strategy": "magic"}},
        {"cleaning": {"fuzzy_threshold": 1.5}},
        {"cleaning": {"transformations": {"scaling": "huge"}}},
        {"validation": {"contamination": 0.9}},
        {"validation": {"pass_threshold": True}},
        {"validation": {"seed": 1.5}},
        {"validation": {"fail_on_validation": "yes"}},
        {"validation": {"rules": {"postcode_countries": ["xx"]}}},
        {"validation": {"rules": {"numeric_ranges": {"age": [1]}}}},
        {"logging": {"level": "LOUD"}},
        {"input": "data.csv"},
    ],
)
def test_invalid_config_is_rejected(raw, tmp_path):
    with pytest.raises(ConfigError):
        parse_config(raw, base_dir=tmp_path)


def test_load_config_reports_missing_and_malformed_files(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("input: [unclosed", encoding="utf-8")

    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(bad)


def test_pipeline_logging_writes_levels_to_file_and_cleans_up(tmp_path):
    log_file = tmp_path / "logs" / "run.log"
    logger = logging.getLogger(LOGGER_NAME)

    with pipeline_logging("INFO", log_file) as log:
        log.debug("hidden")
        log.info("info line")
        log.warning("warning line")
        log.error("error line")

    text = log_file.read_text(encoding="utf-8")
    assert "INFO" in text and "WARNING" in text and "ERROR" in text
    assert "hidden" not in text
    assert logger.handlers == []


def test_pipeline_run_writes_all_artifacts_and_summary(tmp_path):
    config = make_config(tmp_path)

    result = run_pipeline(config)

    assert result.status == "success" and result.pipeline_ready is True
    assert [stage.name for stage in result.stages] == ["profiling", "cleaning", "validation"]
    assert all(stage.status == "success" and stage.duration_seconds >= 0 for stage in result.stages)
    assert result.profiling_status == "completed" and result.validation_status == "pass"
    for name in ("profiling_report", "cleaned_data", "cleaning_log", "validation_report", "pipeline_summary", "log"):
        assert Path(result.artifacts[name]).is_file(), name

    summary = json.loads((tmp_path / "out" / "end_to_end_pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == result.run_id and summary["schema_version"] == "1.1"
    assert summary["config"]["cleaning"]["imputation_strategy"] == "median"
    assert summary["stages"][2]["metrics"]["health_score"] == result.validation_health_score

    log_text = Path(result.artifacts["log"]).read_text(encoding="utf-8")
    assert "Stage 'validation' finished" in log_text and result.run_id in log_text
    assert logging.getLogger(LOGGER_NAME).handlers == []


def test_stage_failure_is_logged_and_later_stages_are_skipped(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(orchestrator, "clean_dataset", broken)

    result = run_pipeline(make_config(tmp_path))

    assert result.status == "failed" and result.pipeline_ready is False
    assert [stage.status for stage in result.stages] == ["success", "failed", "skipped"]
    assert "RuntimeError: disk full" in result.stages[1].error
    assert result.validation_status == "not_run" and result.cleaned_file is None
    log_text = Path(result.artifacts["log"]).read_text(encoding="utf-8")
    assert "ERROR" in log_text and "disk full" in log_text and "skipped" in log_text
    assert (tmp_path / "out" / "end_to_end_pipeline_summary.json").is_file()


@pytest.mark.parametrize("fail_on_validation, expected_status", [(False, "success"), (True, "failed")])
def test_failed_validation_only_fails_the_run_when_configured(tmp_path, fail_on_validation, expected_status):
    config = make_config(
        tmp_path,
        validation={"rules": {"numeric_ranges": {"age": [0, 1]}}, "fail_on_validation": fail_on_validation},
    )

    result = run_pipeline(config)

    assert result.validation_status == "fail" and result.pipeline_ready is False
    assert result.status == expected_status
    assert (result.error is not None) == fail_on_validation


def test_log_file_can_be_disabled(tmp_path):
    result = run_pipeline(make_config(tmp_path, logging={"file": None}))

    assert "log" not in result.artifacts
    assert not (tmp_path / "out" / "pipeline.log").exists()


def test_input_data_version_is_read_from_the_dvc_file(tmp_path):
    data = tmp_path / "data.csv"
    (tmp_path / "data.csv.dvc").write_text("outs:\n- md5: cf15\n  size: 1\n  path: data.csv\n", encoding="utf-8")

    assert orchestrator.input_data_version(data) == "cf15"


@pytest.mark.parametrize("content", [None, "outs: [unclosed", "outs: []", "outs:\n- md5: x\n  path: other.csv\n", "- not a mapping"])
def test_input_data_version_is_none_when_untracked_or_unreadable(tmp_path, content):
    if content is not None:
        (tmp_path / "data.csv.dvc").write_text(content, encoding="utf-8")

    assert orchestrator.input_data_version(tmp_path / "data.csv") is None


def test_run_summary_records_input_data_version(tmp_path):
    config = make_config(tmp_path)
    (tmp_path / "data.csv.dvc").write_text("outs:\n- md5: 0123abcd\n  path: data.csv\n", encoding="utf-8")

    result = run_pipeline(config)

    summary = json.loads((tmp_path / "out" / "end_to_end_pipeline_summary.json").read_text(encoding="utf-8"))
    assert result.input_data_version == summary["input_data_version"] == "0123abcd"


def test_missing_input_or_reference_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="No input"):
        run_pipeline(parse_config({}, base_dir=tmp_path))
    with pytest.raises(ConfigError, match="Input file not found"):
        run_pipeline(parse_config({"input": {"path": "nope.csv"}}, base_dir=tmp_path))
    with pytest.raises(ConfigError, match="Reference file not found"):
        run_pipeline(make_config(tmp_path, validation={"reference": "nope.csv"}))
