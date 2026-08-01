"""End-to-end acceptance tests for the golden pipelines.

These are the executable form of the acceptance specifications in
docs/adr/0001-product-contract.md. They are hermetic: no network, no API keys,
no Docker, no model provider. They exercise the real compiler, the real
executor and the real filesystem tool, and they run the pipelines through
*both* supported surfaces -- the CLI and the Python API -- because the contract
requires the two to agree.

Nothing here is mocked. A failure means the advertised path is broken.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parent / "golden"
BASIC = GOLDEN_DIR / "basic.yaml"
CONTROL_FLOW = GOLDEN_DIR / "control_flow.yaml"
FAILURE = GOLDEN_DIR / "failure.yaml"

pytestmark = [pytest.mark.e2e]


def _run_cli(args, cwd):
    """Invoke the CLI as a subprocess, the way a user would."""
    env = dict(os.environ)
    src = str(Path(__file__).parent.parent / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    # Keep the run hermetic regardless of the developer's shell.
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


async def _run_api(pipeline_path, context, cwd):
    """Execute a pipeline through the Python API, tool-only (no models)."""
    from orchestrator.control_systems.tool_integrated_control_system import (
        ToolIntegratedControlSystem,
    )
    from orchestrator.orchestrator import Orchestrator

    previous = Path.cwd()
    os.chdir(cwd)
    try:
        orchestrator = Orchestrator(control_system=ToolIntegratedControlSystem())
        try:
            return await orchestrator.execute_yaml_file(str(pipeline_path), context)
        finally:
            await orchestrator.shutdown()
    finally:
        os.chdir(previous)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pipeline",
    [BASIC, CONTROL_FLOW, FAILURE],
    ids=["basic", "control_flow", "failure"],
)
def test_golden_pipeline_validates(pipeline, tmp_path):
    """Every golden pipeline compiles and reports its task graph."""
    result = _run_cli(["validate", str(pipeline)], cwd=tmp_path)
    assert result.returncode == 0, (
        f"validate failed for {pipeline.name}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "is valid" in result.stdout


def test_validate_accepts_what_run_accepts(tmp_path):
    """`validate` must not reject a pipeline that `run` executes successfully.

    Regression: the tool validator inspected only `parameters:` when checking a
    tool's required arguments, so a step written in the documented two-field
    form (`tool:` + `action:`) was reported as missing a required `action`
    parameter. It surfaced only when no parameter contained a template --
    templates downgraded the finding to a warning -- so a literal-path pipeline
    failed `validate` while `run` succeeded.
    """
    pipeline = tmp_path / "literal.yaml"
    pipeline.write_text(
        "id: literal\n"
        "name: Literal Path Pipeline\n"
        "steps:\n"
        "  - id: write_it\n"
        "    tool: filesystem\n"
        "    action: write\n"
        "    parameters:\n"
        "      path: \"./out/literal.txt\"\n"
        "      content: \"no templates here\"\n"
    )

    validated = _run_cli(["validate", str(pipeline)], cwd=tmp_path)
    executed = _run_cli(["run", str(pipeline)], cwd=tmp_path)

    assert executed.returncode == 0, f"run failed:\n{executed.stdout}\n{executed.stderr}"
    assert validated.returncode == 0, (
        "validate rejected a pipeline that run executed successfully:\n"
        f"{validated.stdout}\n{validated.stderr}"
    )
    assert (tmp_path / "out" / "literal.txt").read_text() == "no templates here"


def test_validate_rejects_malformed_pipeline(tmp_path):
    """A pipeline that cannot compile exits 2, not 0."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("id: bad\nsteps:\n  - id: x\n    tool: no_such_tool_at_all\n    action: nope\n")
    result = _run_cli(["validate", str(bad)], cwd=tmp_path)
    assert result.returncode == 2, (
        f"expected exit 2 for an invalid pipeline, got {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# basic: sequential steps, templating, typed outputs
# ---------------------------------------------------------------------------

def test_basic_pipeline_via_cli(tmp_path):
    result = _run_cli(["run", str(BASIC), "-i", "greeting=hello"], cwd=tmp_path)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    produced = tmp_path / "golden_out" / "greeting.txt"
    assert produced.is_file(), "pipeline did not write its output file"
    # The `greeting` input must reach the template.
    assert produced.read_text() == "hello world"

    # stdout must be the typed result document.
    payload = json.loads(result.stdout[result.stdout.index("{"):])
    steps = payload["steps"]
    assert steps["write_greeting"]["success"] is True
    assert steps["read_back"]["success"] is True
    # The dependent step read back exactly what the first step wrote.
    assert steps["read_back"]["value"]["result"]["content"] == "hello world"


@pytest.mark.asyncio
async def test_basic_pipeline_via_api(tmp_path):
    results = await _run_api(BASIC, {"greeting": "hello"}, tmp_path)
    assert results["read_back"]["result"]["content"] == "hello world"
    assert (tmp_path / "golden_out" / "greeting.txt").read_text() == "hello world"


@pytest.mark.asyncio
async def test_cli_and_api_agree(tmp_path):
    """The contract requires both surfaces to produce the same result."""
    cli_dir = tmp_path / "cli"
    api_dir = tmp_path / "api"
    cli_dir.mkdir()
    api_dir.mkdir()

    cli_result = _run_cli(["run", str(BASIC), "-i", "greeting=hello"], cwd=cli_dir)
    assert cli_result.returncode == 0, cli_result.stderr
    cli_payload = json.loads(cli_result.stdout[cli_result.stdout.index("{"):])

    api_result = await _run_api(BASIC, {"greeting": "hello"}, api_dir)

    # Whole documents, not selected values: a difference anywhere is caught.
    def _normalise(payload):
        payload = json.loads(json.dumps(payload))
        payload["execution_id"] = "<execution-id>"
        for key in ("started_at", "completed_at", "duration"):
            payload[key] = None
        for step in payload["steps"].values():
            for key in ("started_at", "completed_at", "duration"):
                step[key] = None
        return payload

    assert _normalise(cli_payload) == _normalise(api_result.to_dict())
    assert (
        cli_payload["steps"]["read_back"]["value"]["result"]["content"]
        == "hello world"
    )
    assert (cli_dir / "golden_out" / "greeting.txt").read_text() == (
        api_dir / "golden_out" / "greeting.txt"
    ).read_text()


# ---------------------------------------------------------------------------
# control flow: parallel fan-out, dependency ordering, cross-step data flow
# ---------------------------------------------------------------------------

def test_control_flow_pipeline_via_cli(tmp_path):
    result = _run_cli(["run", str(CONTROL_FLOW)], cwd=tmp_path)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    out = tmp_path / "golden_out"
    assert (out / "alpha.txt").read_text() == "alpha-value"
    assert (out / "beta.txt").read_text() == "beta-value"

    # The join step ran after both branches and interpolated both outputs.
    joined = (out / "joined.txt").read_text()
    assert joined == "alpha-value+beta-value", (
        f"cross-step template interpolation failed; got {joined!r}"
    )


def test_failed_step_propagates_to_exit_code(tmp_path):
    """A step that fails must NOT produce a successful run.

    The executor records a step failure in that step's result and carries on,
    so the process previously exited 0 even though `read_missing` reported
    `"success": false`. A caller (shell script, CI job, parent pipeline) would
    have been told the pipeline succeeded.
    """
    result = _run_cli(["run", str(FAILURE)], cwd=tmp_path)

    assert result.returncode == 1, (
        f"expected exit 1 for a pipeline with a failing step, got "
        f"{result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # The failing step is named, so the operator knows which one broke.
    assert "read_missing" in result.stderr

    payload = json.loads(result.stdout[result.stdout.index("{"):])
    assert payload["success"] is False
    steps = payload["steps"]
    assert steps["read_missing"]["success"] is False
    assert steps["read_missing"]["error"]
    # The step before the failure still ran and is reported honestly.
    assert steps["write_first"]["success"] is True
    assert (tmp_path / "golden_out" / "before_failure.txt").is_file()


def test_successful_pipeline_still_exits_zero(tmp_path):
    """Guard against the failure check misfiring on healthy pipelines."""
    result = _run_cli(["run", str(BASIC), "-i", "greeting=fine"], cwd=tmp_path)
    assert result.returncode == 0, result.stderr


def test_unresolved_template_does_not_reach_output(tmp_path):
    """Template references must resolve, not be written through literally.

    Guards a regression class that recurred across several refactors: an
    unresolved `{{ step.field }}` silently landing in the output file instead of
    the value it names.
    """
    result = _run_cli(["run", str(CONTROL_FLOW)], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    joined = (tmp_path / "golden_out" / "joined.txt").read_text()
    assert "{{" not in joined and "}}" not in joined


# ---------------------------------------------------------------------------
# CLI contract details
# ---------------------------------------------------------------------------

def test_input_flag_parses_json_scalars(tmp_path):
    """-i values parse as JSON when possible so types survive."""
    result = _run_cli(["run", str(BASIC), "-i", 'greeting="quoted"'], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "golden_out" / "greeting.txt").read_text() == "quoted world"


def test_malformed_input_flag_is_rejected(tmp_path):
    result = _run_cli(["run", str(BASIC), "-i", "no_equals_sign"], cwd=tmp_path)
    assert result.returncode != 0
    assert "key=value" in (result.stderr + result.stdout)


def test_output_file_option(tmp_path):
    target = tmp_path / "results.json"
    result = _run_cli(["run", str(BASIC), "-o", str(target)], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads(target.read_text())
    assert payload["steps"]["read_back"]["success"] is True
