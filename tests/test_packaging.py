import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
IMPORT_TO_DISTRIBUTION = {"sklearn": "scikit-learn", "yaml": "PyYAML"}


def dockerfile_copy_sources():
    sources = []
    for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and parts[0].upper() == "COPY":
            sources.extend(part for part in parts[1:-1] if not part.startswith("--"))
    return sources


def runtime_packages():
    return sorted(path.name for path in ROOT.glob("module_*") if path.is_dir())


def declared_requirements():
    names = set()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-")):
            names.add(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].lower().replace("_", "-"))
    return names


def imported_third_party_modules():
    local = set(runtime_packages()) | {"pipeline"}
    found = set()
    for source in [ROOT / "pipeline.py", *ROOT.glob("module_*/**/*.py")]:
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            found.update(name.split(".")[0] for name in names)
    return {name for name in found if name not in sys.stdlib_module_names and name not in local}


def test_dockerfile_copies_every_runtime_package_and_nothing_else_heavy():
    copied = set(dockerfile_copy_sources())

    assert {"requirements.txt", "pipeline.py", "config", *runtime_packages()} <= copied
    assert not copied & {"tests", "data", "outputs", ".venv", ".git"}


def test_requirements_cover_every_third_party_import():
    declared = declared_requirements()
    needed = {IMPORT_TO_DISTRIBUTION.get(name, name).lower().replace("_", "-") for name in imported_third_party_modules()}

    assert needed, "expected the scan to find third-party imports"
    assert needed <= declared, f"missing from requirements.txt: {sorted(needed - declared)}"


def test_dev_requirements_include_runtime_requirements_and_test_tools():
    lines = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8").split()

    assert "-r" in lines and "requirements.txt" in lines
    assert any(line.startswith("pytest") for line in lines)


def test_dockerignore_keeps_the_build_context_small():
    ignored = set((ROOT / ".dockerignore").read_text(encoding="utf-8").split())

    assert {".venv", ".git", "outputs", "data", "tests"} <= ignored


@pytest.mark.integration
def test_pipeline_runs_from_a_copy_containing_only_the_image_files(dataset, tmp_path):
    image_root = tmp_path / "image"
    image_root.mkdir()
    for source in dockerfile_copy_sources():
        origin, target = ROOT / source, image_root / source
        if origin.is_dir():
            shutil.copytree(origin, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(origin, target)

    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    done = subprocess.run(
        [sys.executable, "pipeline.py", "--input", str(dataset), "--output", str(tmp_path / "results")],
        cwd=image_root, env=env, capture_output=True, text=True, timeout=300,
    )

    assert done.returncode == 0, done.stderr
    assert (tmp_path / "results" / "validation_report.json").is_file()
