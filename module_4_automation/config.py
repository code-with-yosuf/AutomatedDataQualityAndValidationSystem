from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from module_2_cleaning.cleaner import IMPUTATION_STRATEGIES
from module_3_validation.rules import POSTCODE_PATTERNS

from .logging_setup import LEVELS
from .schemas import CleaningConfig, LoggingConfig, PipelineConfig, ValidationConfig

TOP_LEVEL_KEYS = {"input", "output", "cleaning", "validation", "logging"}
SCALING_OPTIONS = {"none", "standard", "minmax"}
TRANSFORMATION_KEYS = {"scaling", "encode_categoricals", "extract_date_features"}
RULE_KEYS = {"numeric_ranges", "allowed_values", "postcode_countries"}


class ConfigError(ValueError):
    pass


def _section(raw: Dict[str, Any], name: str, allowed: set) -> Dict[str, Any]:
    section = raw.get(name)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ConfigError(f"'{name}' must be a mapping")
    unknown = set(section) - allowed
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{name}': {sorted(unknown)}")
    return section


def _number(value: Any, name: str, low: float, high: float, low_open: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{name}' must be a number")
    if (value <= low if low_open else value < low) or value > high:
        raise ConfigError(f"'{name}' must be between {low} and {high}, got {value}")
    return float(value)


def _path(value: Any, name: str, base_dir: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigError(f"'{name}' must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _transformations(value: Any) -> Optional[Dict[str, Any]]:
    if not value:
        return None
    if not isinstance(value, dict):
        raise ConfigError("'cleaning.transformations' must be a mapping")
    unknown = set(value) - TRANSFORMATION_KEYS
    if unknown:
        raise ConfigError(f"Unknown key(s) in 'cleaning.transformations': {sorted(unknown)}")
    if value.get("scaling", "standard") not in SCALING_OPTIONS:
        raise ConfigError(f"'cleaning.transformations.scaling' must be one of {sorted(SCALING_OPTIONS)}")
    for key in ("encode_categoricals", "extract_date_features"):
        if key in value and not isinstance(value[key], bool):
            raise ConfigError(f"'cleaning.transformations.{key}' must be true or false")
    return dict(value)


def _rules(value: Any) -> Dict[str, Any]:
    if not value:
        return {}
    if not isinstance(value, dict):
        raise ConfigError("'validation.rules' must be a mapping")
    unknown = set(value) - RULE_KEYS
    if unknown:
        raise ConfigError(f"Unknown key(s) in 'validation.rules': {sorted(unknown)}")

    for column, bounds in (value.get("numeric_ranges") or {}).items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ConfigError(f"'validation.rules.numeric_ranges.{column}' must be [min, max]")
    for column, allowed in (value.get("allowed_values") or {}).items():
        if not isinstance(allowed, (list, tuple)):
            raise ConfigError(f"'validation.rules.allowed_values.{column}' must be a list")
    countries = value.get("postcode_countries") or []
    if set(countries) - set(POSTCODE_PATTERNS):
        raise ConfigError(f"'validation.rules.postcode_countries' must be a subset of {sorted(POSTCODE_PATTERNS)}")
    return dict(value)


def parse_config(raw: Optional[Dict[str, Any]], base_dir: Union[str, Path, None] = None) -> PipelineConfig:
    """Validate a raw mapping; relative paths resolve against base_dir (default: current directory)."""
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError("Configuration must be a mapping")
    unknown = set(raw) - TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"Unknown top-level key(s): {sorted(unknown)}")
    base = Path(base_dir) if base_dir is not None else Path.cwd()

    input_section = _section(raw, "input", {"path"})
    output_section = _section(raw, "output", {"dir"})
    cleaning = _section(raw, "cleaning", {"imputation_strategy", "fuzzy_threshold", "transformations"})
    validation = _section(raw, "validation", {"pass_threshold", "contamination", "seed", "reference", "rules", "fail_on_validation"})
    logging_section = _section(raw, "logging", {"level", "file"})

    defaults = (CleaningConfig(), ValidationConfig(), LoggingConfig())

    strategy = cleaning.get("imputation_strategy", defaults[0].imputation_strategy)
    if strategy not in IMPUTATION_STRATEGIES:
        raise ConfigError(f"'cleaning.imputation_strategy' must be one of {sorted(IMPUTATION_STRATEGIES)}")

    contamination = validation.get("contamination", defaults[1].contamination)
    if contamination != "auto":
        contamination = _number(contamination, "validation.contamination", 0.0, 0.5, low_open=True)

    seed = validation.get("seed", defaults[1].seed)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError("'validation.seed' must be an integer")

    fail_on_validation = validation.get("fail_on_validation", defaults[1].fail_on_validation)
    if not isinstance(fail_on_validation, bool):
        raise ConfigError("'validation.fail_on_validation' must be true or false")

    level = str(logging_section.get("level", defaults[2].level)).upper()
    if level not in LEVELS:
        raise ConfigError(f"'logging.level' must be one of {list(LEVELS)}")

    log_file = logging_section.get("file", defaults[2].file)
    if log_file is not None and not (isinstance(log_file, str) and log_file.strip()):
        raise ConfigError("'logging.file' must be a file name or null")

    reference = validation.get("reference")
    return PipelineConfig(
        input_path=_path(input_section["path"], "input.path", base) if input_section.get("path") else None,
        output_dir=_path(output_section.get("dir", "outputs"), "output.dir", base),
        cleaning=CleaningConfig(
            imputation_strategy=strategy,
            fuzzy_threshold=_number(cleaning.get("fuzzy_threshold", defaults[0].fuzzy_threshold), "cleaning.fuzzy_threshold", 0.0, 1.0),
            transformations=_transformations(cleaning.get("transformations")),
        ),
        validation=ValidationConfig(
            pass_threshold=_number(validation.get("pass_threshold", defaults[1].pass_threshold), "validation.pass_threshold", 0.0, 100.0),
            contamination=contamination,
            seed=seed,
            reference=_path(reference, "validation.reference", base) if reference else None,
            rules=_rules(validation.get("rules")),
            fail_on_validation=fail_on_validation,
        ),
        logging=LoggingConfig(level=level, file=log_file),
    )


def load_config(path: Union[str, Path], base_dir: Union[str, Path, None] = None) -> PipelineConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file is not valid YAML: {exc}") from exc
    return parse_config(raw, base_dir)
