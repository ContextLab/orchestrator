"""The PipelineResult contract.

`execute_pipeline` used to return a bare `{step_id: value}` dict -- except when
the pipeline declared `outputs:`, when it returned
`{"steps": ..., "outputs": ...}` instead. Two shapes from one method, told
apart only by inspecting keys. Everything else the run knew (timing, model
identity, retries, skips, the order things ran in) was computed and dropped.

These tests pin the typed result, and pin that the CLI and the Python API agree
on it as *data* rather than on a handful of nested values.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.core.pipeline_result import PipelineResult, StepResult
from orchestrator.core.task import Task, TaskStatus
from tests.test_infrastructure import create_test_orchestrator

pytestmark = [pytest.mark.contract, pytest.mark.e2e]

TOOL_PIPELINE = """
id: result_contract_pipeline
name: Result Contract Pipeline
parameters:
  out_dir:
    type: string
    default: "./result_out"
steps:
  - id: alpha
    tool: filesystem
    action: write
    parameters:
      path: "{{ out_dir }}/alpha.txt"
      content: "alpha-value"
  - id: read_alpha
    tool: filesystem
    action: read
    parameters:
      path: "{{ out_dir }}/alpha.txt"
    dependencies:
      - alpha
outputs:
  saved: "{{ read_alpha.result.content }}"
"""

FAILING_PIPELINE = """
id: failing_result_pipeline
name: Failing Result Pipeline
steps:
  - id: read_missing
    tool: filesystem
    action: read
    on_failure: continue
    parameters:
      path: "/nonexistent/definitely/not/here.txt"
"""


def _run_api(yaml_content, cwd, context=None):
    import asyncio

    previous = Path.cwd()
    os.chdir(cwd)
    try:
        orchestrator = create_test_orchestrator()
        return asyncio.run(
            orchestrator.execute_yaml(yaml_content=yaml_content, context=context or {})
        )
    finally:
        os.chdir(previous)


def _run_cli(pipeline_path, cwd):
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(Path(__file__).parent.parent / "src") + os.pathsep + env.get("PYTHONPATH", "")
    )
    env.pop("ANTHROPIC_API_KEY", None)
    env["ORCHESTRATOR_AUTO_INSTALL"] = "0"
    return subprocess.run(
        [sys.executable, "-m", "orchestrator.cli", "run", str(pipeline_path)],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


# ---------------------------------------------------------------------------
# shape
# ---------------------------------------------------------------------------

def test_the_result_is_typed_and_still_indexes_by_step_id(tmp_path):
    """The compatibility promise: `result["step_id"]` is unchanged.

    Thousands of existing assertions index the result directly. A typed result
    that broke them would have to be introduced by rewriting them all, so the
    type is a Mapping and the trace arrives alongside.
    """
    result = _run_api(TOOL_PIPELINE, tmp_path)

    assert isinstance(result, PipelineResult)
    assert "alpha" in result and "read_alpha" in result
    assert result["read_alpha"] == result.steps["read_alpha"].value


def test_declared_outputs_no_longer_change_the_return_shape(tmp_path):
    """Outputs are a field, not a different shape.

    Previously a pipeline with `outputs:` returned {"steps":…, "outputs":…}
    and one without returned {step_id: …}, so a caller could not index a
    result without first checking which it had been handed. A step named
    `outputs` collided with the second shape outright.
    """
    result = _run_api(TOOL_PIPELINE, tmp_path)

    assert result.outputs, "declared outputs must be resolved"
    assert result.outputs["saved"] == "alpha-value"
    # The steps are still addressable exactly as in a pipeline with no outputs.
    assert sorted(result) == ["alpha", "read_alpha"]


def test_overall_status_and_success(tmp_path):
    result = _run_api(TOOL_PIPELINE, tmp_path)

    assert result.success is True
    assert result.status == "completed"
    assert result.pipeline_id == "result_contract_pipeline"
    assert result.execution_id


# ---------------------------------------------------------------------------
# the trace
# ---------------------------------------------------------------------------

def test_execution_order_and_levels_reflect_the_dependency_graph(tmp_path):
    result = _run_api(TOOL_PIPELINE, tmp_path)

    assert result.execution_order.index("alpha") < result.execution_order.index("read_alpha")
    assert result.execution_levels == (("alpha",), ("read_alpha",)), (
        f"a dependent step must occupy a later level: {result.execution_levels}"
    )


def test_every_step_carries_timing(tmp_path):
    result = _run_api(TOOL_PIPELINE, tmp_path)

    for step in result.steps.values():
        assert step.started_at is not None, f"{step.id} has no start time"
        assert step.completed_at is not None, f"{step.id} has no completion time"
        assert step.duration is not None and step.duration >= 0
    assert result.duration is not None and result.duration >= 0


def test_the_tool_that_ran_a_step_is_recorded(tmp_path):
    result = _run_api(TOOL_PIPELINE, tmp_path)

    assert result.steps["alpha"].tool == "filesystem"


def test_the_model_that_answered_a_step_is_recorded(tmp_path):
    model_pipeline = """
id: model_result_pipeline
name: Model Result Pipeline
steps:
  - id: think
    action: generate
    parameters:
      prompt: "summarise this"
"""
    result = _run_api(model_pipeline, tmp_path)

    assert result.steps["think"].model == "test-model", (
        "the selected model's identity must survive into the trace"
    )


def test_a_failed_step_carries_a_structured_error(tmp_path):
    """Not a stringified recovery policy -- the reason.

    Every failed step used to record the error handler's decision --
    {"action": "retry", "delay": 5.0, ...} -- in place of its error.
    """
    result = _run_api(FAILING_PIPELINE, tmp_path)

    step = result.steps["read_missing"]
    assert step.success is False
    # `status` records whether the task finished; `success` whether it worked.
    # A tool that returns {"success": False} without raising finishes, so the
    # two differ here -- and reading only `status` is what let a failing
    # pipeline report success.
    assert step.status == TaskStatus.COMPLETED.value
    assert step.error, "a failed step must say why"
    assert step.error_type, "and of what kind"
    assert "retry" not in str(step.error_type).lower()

    assert result.success is False
    assert result.status == "failed"
    assert [s.id for s in result.failed_steps] == ["read_missing"]


# ---------------------------------------------------------------------------
# serialisation, and CLI/API equivalence
# ---------------------------------------------------------------------------

def test_to_dict_is_json_serialisable_without_coercion(tmp_path):
    """`json.dumps(..., default=str)` hid unserialisable values by stringifying
    them, so the JSON was neither stable nor round-trippable."""
    result = _run_api(TOOL_PIPELINE, tmp_path)

    encoded = json.dumps(result.to_dict())  # no `default=` -- must not need one
    assert json.loads(encoded)["pipeline_id"] == "result_contract_pipeline"


def test_normalized_drops_everything_run_specific(tmp_path):
    """Two runs of the same pipeline must normalise to the same data."""
    first = _run_api(TOOL_PIPELINE, tmp_path)
    second = _run_api(TOOL_PIPELINE, tmp_path)

    assert first.normalized() == second.normalized(), (
        "two runs of one pipeline disagree after normalisation"
    )


def test_the_cli_and_the_python_api_produce_the_same_result(tmp_path):
    """The contract that makes the CLI and the library one product.

    Compared as whole normalised documents, not by fishing out selected nested
    values -- that is what lets a difference anywhere be caught.
    """
    cli_dir = tmp_path / "cli"
    api_dir = tmp_path / "api"
    cli_dir.mkdir()
    api_dir.mkdir()

    pipeline_path = cli_dir / "pipeline.yaml"
    pipeline_path.write_text(TOOL_PIPELINE)

    completed = _run_cli(pipeline_path, cli_dir)
    assert completed.returncode == 0, (
        f"CLI run failed:\n{completed.stdout}\n{completed.stderr}"
    )
    from_cli = json.loads(completed.stdout)

    from_api = _run_api(TOOL_PIPELINE, api_dir).to_dict()

    # Normalise both the same way the type does.
    for payload in (from_cli, from_api):
        payload["execution_id"] = "<execution-id>"
        for key in ("started_at", "completed_at", "duration"):
            payload[key] = None
        for step in payload["steps"].values():
            for key in ("started_at", "completed_at", "duration"):
                step[key] = None

    assert from_cli == from_api, (
        "the CLI and the Python API disagree about the same pipeline"
    )


# ---------------------------------------------------------------------------
# the type in isolation
# ---------------------------------------------------------------------------

def test_step_result_from_a_skipped_task():
    task = Task(id="s", name="s", action="generate")
    task.skip("dependency failed")

    step = StepResult.from_task(task)

    assert step.status == TaskStatus.SKIPPED.value
    assert step.success is False


def test_step_result_reports_retries_beyond_the_first_attempt():
    task = Task(id="s", name="s", action="generate")
    task.fail(RuntimeError("boom"))
    task.fail(RuntimeError("boom again"))

    step = StepResult.from_task(task)

    assert step.retries == 1, (
        "Task.retry_count counts the initial failure too, so it is not the "
        f"number of retries: got retry_count={task.retry_count}"
    )
    assert step.error == "boom again"
    assert step.error_type == "RuntimeError"


def test_a_result_with_no_steps_is_still_a_mapping():
    result = PipelineResult(
        pipeline_id="p", execution_id="e", status="completed", success=True
    )

    assert len(result) == 0
    assert list(result) == []
    assert result.to_dict()["steps"] == {}
