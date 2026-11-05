from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

SCHEMA_VERSION = "1.1"


@dataclass(frozen=True)
class CleaningConfig:
    imputation_strategy: str = "median"
    fuzzy_threshold: float = 0.92
    transformations: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ValidationConfig:
    pass_threshold: float = 80.0
    contamination: Union[str, float] = "auto"
    seed: int = 42
    reference: Optional[Path] = None
    rules: Dict[str, Any] = field(default_factory=dict)
    fail_on_validation: bool = False


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    # Relative paths resolve against the output directory; None disables the log file.
    file: Optional[str] = "pipeline.log"


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline input schema."""

    input_path: Optional[Path] = None
    output_dir: Path = Path("outputs")
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


@dataclass
class StageResult:
    name: str
    status: str  # success | failed | skipped
    duration_seconds: float = 0.0
    outputs: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Pipeline output schema, written to end_to_end_pipeline_summary.json."""

    run_id: str
    status: str  # success | failed
    started_at: str
    finished_at: str
    duration_seconds: float
    input_file: str
    output_dir: str
    cleaned_file: Optional[str]
    profiling_status: str  # completed | failed | skipped
    validation_status: str  # pass | fail | not_run
    validation_health_score: Optional[float]
    issues_found: List[str]
    pipeline_ready: bool
    stages: List[StageResult]
    artifacts: Dict[str, str]
    config: Dict[str, Any]
    error: Optional[str] = None
    input_data_version: Optional[str] = None  # md5 recorded in the input's DVC file, if tracked
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return jsonable(self)


def jsonable(value: Any) -> Any:
    """Convert dataclasses, paths, tuples and numpy scalars into JSON-safe values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value
