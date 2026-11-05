"""Command-line entry point: python pipeline.py --input data.csv --output results/"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import List, Optional

from module_4_automation import ConfigError, PipelineResult, load_config, run_pipeline
from module_4_automation.logging_setup import LEVELS

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "pipeline.yaml"
EXIT_OK, EXIT_PIPELINE_FAILED, EXIT_CONFIG_ERROR = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Profile, clean and validate a CSV dataset.",
        epilog="Exit codes: 0 success, 1 pipeline failed, 2 configuration or input error.",
    )
    parser.add_argument("--input", metavar="CSV", help="dataset to process (overrides input.path)")
    parser.add_argument("--output", metavar="DIR", help="directory for results (overrides output.dir)")
    parser.add_argument(
        "--config", metavar="YAML",
        help=f"configuration file (default: {DEFAULT_CONFIG.relative_to(ROOT)}, relative to the project root); "
        "relative paths inside a custom config resolve against the current directory",
    )
    parser.add_argument("--reference", metavar="CSV", help="baseline dataset for drift detection (overrides validation.reference)")
    parser.add_argument("--log-level", type=str.upper, choices=LEVELS, help="overrides logging.level")
    parser.add_argument("--fail-on-validation", action="store_true", help="exit with 1 when validation fails")
    return parser


def _print_summary(result: PipelineResult) -> None:
    print(f"Status:           {result.status}")
    print(f"Pipeline ready:   {result.pipeline_ready}")
    print(f"Validation:       {result.validation_status} (health score {result.validation_health_score})")
    if result.input_data_version:
        print(f"Input version:    {result.input_data_version}")
    if result.error:
        print(f"Error:            {result.error}")
    print("Artifacts:")
    for name, path in result.artifacts.items():
        print(f"  {name}: {path}")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config) if args.config else load_config(DEFAULT_CONFIG, base_dir=ROOT)
        validation, logging_config = config.validation, config.logging
        if args.reference:
            validation = dataclasses.replace(validation, reference=Path(args.reference).resolve())
        if args.fail_on_validation:
            validation = dataclasses.replace(validation, fail_on_validation=True)
        if args.log_level:
            logging_config = dataclasses.replace(logging_config, level=args.log_level)
        config = dataclasses.replace(
            config,
            input_path=Path(args.input).resolve() if args.input else config.input_path,
            output_dir=Path(args.output).resolve() if args.output else config.output_dir,
            validation=validation,
            logging=logging_config,
        )
        result = run_pipeline(config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    _print_summary(result)
    return EXIT_OK if result.status == "success" else EXIT_PIPELINE_FAILED


if __name__ == "__main__":
    sys.exit(main())
