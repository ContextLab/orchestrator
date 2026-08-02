"""Every pipeline in examples/supported/ is executable product behaviour.

`examples/` held 111 pipelines of which 3 compiled. They were the project's
primary documentation surface and almost none of them worked, so a reader had
no way to tell which parts of the syntax were real.

`examples/supported/` is the answer: a small set that is *tested as product
behaviour*. Every file here must validate, must execute through both the CLI
and the Python API, and the two must agree as whole normalised documents. A
pipeline that cannot do all of that does not belong in this directory.

Adding a file to examples/supported/ without an entry in EXPECTATIONS fails,
so the set cannot grow silently past its coverage.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_infrastructure import create_test_orchestrator

pytestmark = [pytest.mark.contract, pytest.mark.e2e]

SUPPORTED_DIR = Path(__file__).parent.parent / "examples" / "supported"
EXAMPLES = sorted(SUPPORTED_DIR.glob("*.yaml"))

# Exit code 1 is not a failure of the example: 06 deliberately contains a step
# that fails, and the run is supposed to report that honestly.
EXPECTATIONS = {
    "01_hello_filesystem.yaml": {
        "exit_code": 0,
        "artifacts": ["output/greeting.txt"],
        "outputs": {"greeting_text": "hello, world"},
    },
    "02_parallel_fanout_fanin.yaml": {
        "exit_code": 0,
        "artifacts": ["output/alpha.txt", "output/beta.txt", "output/joined.txt"],
        "artifact_contents": {"output/joined.txt": "alpha-value+beta-value"},
    },
    "04_conditions.yaml": {
        "exit_code": 0,
        "artifacts": ["output/long.txt", "output/short.txt"],
    },
    "06_failure_policy.yaml": {
        "exit_code": 1,
        "artifacts": ["output/before.txt", "output/after.txt"],
        "failed_steps": ["flaky"],
    },
    "07_templates_and_outputs.yaml": {
        "exit_code": 0,
        "artifacts": ["output/header.md", "output/report.md"],
    },
}


def _run_cli(args, cwd):
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(Path(__file__).parent.parent / "src") + os.pathsep + env.get("PYTHONPATH", "")
    )
    env.pop("ANTHROPIC_API_KEY", None)
    env["ORCHESTRATOR_AUTO_INSTALL"] = "0"
    return subprocess.run(
        [sys.executable, "-m", "orchestrator.cli", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _normalise(payload):
    """Drop the fields that legitimately differ between two runs."""
    payload = json.loads(json.dumps(payload))
    payload["execution_id"] = "<execution-id>"
    for key in ("started_at", "completed_at", "duration"):
        payload[key] = None
    for step in payload.get("steps", {}).values():
        for key in ("started_at", "completed_at", "duration"):
            step[key] = None
        # Retry timing varies, and a step's value can embed an absolute path
        # that differs between the CLI and API working directories.
        step.pop("value", None)
    payload.pop("outputs", None)
    return payload


def test_the_supported_directory_is_not_empty():
    assert EXAMPLES, f"no examples found in {SUPPORTED_DIR}"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_every_supported_example_has_declared_expectations(example):
    """Coverage cannot lag behind the directory."""
    assert example.name in EXPECTATIONS, (
        f"{example.name} is in examples/supported/ but has no entry in "
        f"EXPECTATIONS, so nothing asserts what it should do"
    )


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_every_supported_example_validates(example, tmp_path):
    result = _run_cli(["validate", str(example)], cwd=tmp_path)

    assert result.returncode == 0, (
        f"{example.name} does not compile:\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_every_supported_example_runs_through_the_cli(example, tmp_path):
    expected = EXPECTATIONS[example.name]

    result = _run_cli(["run", str(example)], cwd=tmp_path)

    assert result.returncode == expected["exit_code"], (
        f"{example.name} exited {result.returncode}, expected "
        f"{expected['exit_code']}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )

    for relative in expected["artifacts"]:
        assert (tmp_path / relative).is_file(), (
            f"{example.name} did not produce {relative}"
        )

    for relative, content in expected.get("artifact_contents", {}).items():
        assert (tmp_path / relative).read_text().strip() == content

    payload = json.loads(result.stdout[result.stdout.index("{"):])
    for step_id in expected.get("failed_steps", []):
        assert payload["steps"][step_id]["success"] is False, (
            f"{example.name}: step {step_id} was expected to fail"
        )


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_every_supported_example_runs_through_the_python_api(example, tmp_path):
    import asyncio

    expected = EXPECTATIONS[example.name]
    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        orchestrator = create_test_orchestrator()
        result = asyncio.run(
            orchestrator.execute_yaml(yaml_content=example.read_text(), context={})
        )
    finally:
        os.chdir(previous)

    assert result.success is (expected["exit_code"] == 0), (
        f"{example.name}: success={result.success} disagrees with the "
        f"expected exit code {expected['exit_code']}"
    )
    for relative in expected["artifacts"]:
        assert (tmp_path / relative).is_file()

    for name, value in expected.get("outputs", {}).items():
        assert result.outputs.get(name) == value, (
            f"{example.name}: declared output {name!r} was "
            f"{result.outputs.get(name)!r}, expected {value!r}"
        )


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_the_cli_and_the_api_agree_on_every_supported_example(example, tmp_path):
    """Whole normalised documents, not selected nested values."""
    import asyncio

    cli_dir = tmp_path / "cli"
    api_dir = tmp_path / "api"
    cli_dir.mkdir()
    api_dir.mkdir()

    cli_result = _run_cli(["run", str(example)], cwd=cli_dir)
    from_cli = json.loads(cli_result.stdout[cli_result.stdout.index("{"):])

    previous = Path.cwd()
    os.chdir(api_dir)
    try:
        from_api = asyncio.run(
            create_test_orchestrator().execute_yaml(
                yaml_content=example.read_text(), context={}
            )
        ).to_dict()
    finally:
        os.chdir(previous)

    assert _normalise(from_cli) == _normalise(from_api), (
        f"{example.name}: the CLI and the Python API disagree"
    )
