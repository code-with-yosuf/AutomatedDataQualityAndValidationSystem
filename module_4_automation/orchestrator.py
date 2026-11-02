from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Tuple

from module_1_profiling.profiler import profile_dataset
from module_2_cleaning.cleaner import clean_dataset
from module_3_validation.validator import validate_dataset

from .config import ConfigError
from .logging_setup import pipeline_logging
from .schemas import PipelineConfig, PipelineResult, StageResult, jsonable

SUMMARY_FILE = "end_to_end_pipeline_summary.json"
MAX_LOGGED_FINDINGS = 10

StageOutput = Tuple[Dict[str, str], Dict[str, Any]]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False), encoding="utf-8")


def _profiling_stage(config: PipelineConfig, state: Dict[str, Any], log: logging.Logger) -> StageOutput:
    visualizations = config.output_dir / "profiling_visualizations"
    profile = profile_dataset(config.input_path, visualizations)
    report_path = config.output_dir / "profiling_report.json"
    _write_json(report_path, profile)
    state["profile"] = profile

    suspicious = profile["quality_checks"]["suspicious_columns"]
    mixed = profile["metadata"]["mixed_type_columns"]
    pii = profile["metadata"]["pii_columns"]
    if suspicious:
        log.warning("Suspicious columns: %s", ", ".join(suspicious))
    if mixed:
        log.warning("Mixed-type columns: %s", ", ".join(mixed))
    if pii:
        log.warning("Potential PII columns: %s", ", ".join(pii))
    if profile["duplicate_rows"]:
        log.warning("Input contains %d duplicate row(s)", profile["duplicate_rows"])

    return (
        {"profiling_report": str(report_path), "profiling_visualizations": str(visualizations)},
        {
            "rows": profile["row_count"],
            "columns": profile["column_count"],
            "suspicious_columns": suspicious,
            "mixed_type_columns": mixed,
            "pii_columns": pii,
            "null_percentage": profile["quality_checks"]["null_percentage"],
        },
    )


def _cleaning_stage(config: PipelineConfig, state: Dict[str, Any], log: logging.Logger) -> StageOutput:
    cleaned_path = clean_dataset(
        config.input_path,
        state["profile"],
        config.output_dir,
        imputation_strategy=config.cleaning.imputation_strategy,
        fuzzy_threshold=config.cleaning.fuzzy_threshold,
        transformations=config.cleaning.transformations,
    )
    state["cleaned_path"] = cleaned_path
    log_path = config.output_dir / "cleaning_log.json"
    cleaning_log = json.loads(log_path.read_text(encoding="utf-8"))

    quality = cleaning_log["quality_score"]
    fuzzy = cleaning_log["actions"]["fuzzy_duplicate_candidates"]
    if fuzzy:
        log.warning("%d fuzzy duplicate candidate(s) need manual review (not removed)", len(fuzzy))
    if quality["delta"] < 0:
        log.warning("Cleaning lowered the quality score by %.2f", -quality["delta"])
    log.info("Quality score %.2f -> %.2f (%+.2f)", quality["before"]["overall"], quality["after"]["overall"], quality["delta"])

    outputs = {"cleaned_data": str(cleaned_path), "cleaning_log": str(log_path)}
    transformed = cleaning_log["transformations"]["output_file"]
    if transformed:
        outputs["transformed_features"] = str(config.output_dir / transformed)
    return outputs, {
        "exact_duplicates_removed": cleaning_log["actions"]["exact_duplicates_removed"],
        "fuzzy_duplicate_candidates": len(fuzzy),
        "quality_before": quality["before"]["overall"],
        "quality_after": quality["after"]["overall"],
        "quality_delta": quality["delta"],
    }


def _validation_stage(config: PipelineConfig, state: Dict[str, Any], log: logging.Logger) -> StageOutput:
    options = config.validation
    visualizations = config.output_dir / "validation_visualizations"
    report = validate_dataset(
        state["cleaned_path"],
        profile=state["profile"],
        output_dir=visualizations,
        reference=options.reference,
        rules=options.rules or None,
        contamination=options.contamination,
        pass_threshold=options.pass_threshold,
        seed=options.seed,
    )
    report_path = config.output_dir / "validation_report.json"
    _write_json(report_path, report)
    state["validation"] = report

    findings = report["findings"]
    severities = {level: sum(f["severity"] == level for f in findings) for level in ("high", "medium", "low")}
    for finding in [f for f in findings if f["severity"] == "high"][:MAX_LOGGED_FINDINGS]:
        log.warning("Validation: %s", finding["message"])
    for section in ("anomaly_detection", "drift", "evaluation"):
        if report.get(section, {}).get("status") == "skipped":
            log.info("Validation %s skipped: %s", section.replace("_", " "), report[section].get("reason"))

    scores = report["scores"]
    log.log(
        logging.INFO if report["status"] == "pass" else logging.WARNING,
        "Validation %s: health score %.2f (threshold %.2f), %d finding(s)",
        report["status"], scores["health_score"], scores["pass_threshold"], len(findings),
    )

    outputs = {"validation_report": str(report_path)}
    if visualizations.exists():
        outputs["validation_visualizations"] = str(visualizations)
    return outputs, {
        "status": report["status"],
        "health_score": scores["health_score"],
        "error_severity": scores["error_severity"],
        "anomaly_severity": scores["anomaly_severity"],
        "findings": len(findings),
        "findings_by_severity": severities,
    }


STAGES: List[Tuple[str, Callable[[PipelineConfig, Dict[str, Any], logging.Logger], StageOutput]]] = [
    ("profiling", _profiling_stage),
    ("cleaning", _cleaning_stage),
    ("validation", _validation_stage),
]


def _execute(name: str, action: Callable[[], StageOutput], log: logging.Logger) -> StageResult:
    log.info("Stage '%s' started", name)
    start = perf_counter()
    try:
        outputs, metrics = action()
    except Exception as exc:
        log.exception("Stage '%s' failed", name)
        return StageResult(name, "failed", round(perf_counter() - start, 3), error=f"{type(exc).__name__}: {exc}")
    duration = round(perf_counter() - start, 3)
    log.info("Stage '%s' finished in %.2fs", name, duration)
    return StageResult(name, "success", duration, outputs, metrics)


def _check_inputs(config: PipelineConfig) -> None:
    if config.input_path is None:
        raise ConfigError("No input dataset configured (set input.path)")
    if not config.input_path.is_file():
        raise ConfigError(f"Input file not found: {config.input_path}")
    if config.validation.reference is not None and not config.validation.reference.is_file():
        raise ConfigError(f"Reference file not found: {config.validation.reference}")


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run profiling, cleaning and validation, and write the run summary.

    Stage failures are recorded in the result rather than raised; a failed stage skips the stages after it.
    Raises ConfigError when the input or reference file is missing.
    """
    _check_inputs(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(config.logging.file) if config.logging.file else None
    log_path = None if log_file is None else (log_file if log_file.is_absolute() else config.output_dir / log_file)

    started = datetime.now(timezone.utc)
    clock = perf_counter()
    run_id = f"{started:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    state: Dict[str, Any] = {}
    stages: List[StageResult] = []
    error = None

    with pipeline_logging(config.logging.level, log_path) as log:
        log.info("Run %s started: input=%s output=%s", run_id, config.input_path, config.output_dir)

        for name, action in STAGES:
            if error:
                log.warning("Stage '%s' skipped because an earlier stage failed", name)
                stages.append(StageResult(name, "skipped"))
                continue
            stage = _execute(name, lambda action=action: action(config, state, log), log)
            stages.append(stage)
            if stage.status == "failed":
                error = f"Stage '{name}' failed: {stage.error}"

        validation = state.get("validation")
        validation_passed = bool(validation) and validation["status"] == "pass"
        if not error and validation and not validation_passed and config.validation.fail_on_validation:
            error = f"Validation failed: health score {validation['scores']['health_score']} below {config.validation.pass_threshold}"
            log.error(error)

        artifacts: Dict[str, str] = {}
        for stage in stages:
            artifacts.update(stage.outputs)
        if log_path is not None:
            artifacts["log"] = str(log_path)
        artifacts["pipeline_summary"] = str(config.output_dir / SUMMARY_FILE)

        by_name = {stage.name: stage for stage in stages}
        result = PipelineResult(
            run_id=run_id,
            status="failed" if error else "success",
            started_at=started.isoformat(timespec="seconds"),
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            duration_seconds=round(perf_counter() - clock, 3),
            input_file=config.input_path.name,
            output_dir=str(config.output_dir),
            cleaned_file=state["cleaned_path"].name if "cleaned_path" in state else None,
            profiling_status={"success": "completed"}.get(by_name["profiling"].status, by_name["profiling"].status),
            validation_status=validation["status"] if validation else "not_run",
            validation_health_score=validation["scores"]["health_score"] if validation else None,
            issues_found=validation["issues"] if validation else [],
            pipeline_ready=not error and validation_passed,
            stages=stages,
            artifacts=artifacts,
            config=jsonable(config),
            error=error,
        )
        _write_json(config.output_dir / SUMMARY_FILE, result.to_dict())
        log.log(
            logging.ERROR if error else logging.INFO,
            "Run %s %s in %.2fs (pipeline_ready=%s)", run_id, result.status, result.duration_seconds, result.pipeline_ready,
        )
    return result
