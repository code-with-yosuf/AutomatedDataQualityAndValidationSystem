from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def run_automation(input_file: str | Path, cleaned_path: str | Path, profiling_report: Dict[str, Any], validation_report: Dict[str, Any], output_dir: str | Path) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "input_file": str(Path(input_file).name),
        "cleaned_file": str(Path(cleaned_path).name),
        "profiling_status": "completed" if profiling_report.get("row_count", 0) > 0 else "failed",
        "validation_status": validation_report.get("status", "unknown"),
        "issues_found": validation_report.get("issues", []),
        "pipeline_ready": validation_report.get("status") == "pass",
    }

    return summary
