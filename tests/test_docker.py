import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.docker

ROOT = Path(__file__).resolve().parent.parent
IMAGE = "dq-pipeline:test"


def docker(*args, timeout=900):
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


@pytest.fixture(scope="module")
def image():
    if shutil.which("docker") is None or docker("info", timeout=60).returncode != 0:
        pytest.skip("Docker is not installed or its daemon is not running")
    build = docker("build", "-t", IMAGE, str(ROOT))
    assert build.returncode == 0, build.stderr
    yield IMAGE
    docker("rmi", "-f", IMAGE)


def run_container(image, *run_args, command=()):
    user = ["--user", f"{os.getuid()}:{os.getgid()}"] if hasattr(os, "getuid") else []
    return docker("run", "--rm", *user, *run_args, image, *command, timeout=600)


def test_container_processes_a_mounted_dataset(image, dataset, tmp_path):
    results = tmp_path / "results"
    results.mkdir()

    done = run_container(
        image,
        "-v", f"{dataset.parent}:/data:ro",
        "-v", f"{results}:/results",
        command=["--input", f"/data/{dataset.name}", "--output", "/results"],
    )

    assert done.returncode == 0, done.stderr
    assert "Status:           success" in done.stdout
    for name in ("profiling_report.json", "cleaned_data.csv", "validation_report.json", "end_to_end_pipeline_summary.json"):
        assert (results / name).is_file(), name


def test_container_default_command_reports_a_missing_dataset(image):
    done = run_container(image)

    assert done.returncode == 2
    assert "Input file not found: /data/input.csv" in done.stderr


def test_container_prints_cli_help(image):
    done = run_container(image, command=["--help"])

    assert done.returncode == 0
    assert "--fail-on-validation" in done.stdout
