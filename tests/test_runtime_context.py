"""One run, one answer about itself.

`{{ execution.timestamp }}` had seven independent implementations in four
formats, and `_execute_level` rebuilt it at every level of the graph, so a
single run answered its own question differently each time:

    step one   ->   2026-08-02T20:01:55.182681
    step two   ->   2026-08-02T20:01:55.184368

Meanwhile `validate` rejected the expression outright -- 59 references across
32 catalogue pipelines, every one of which ran correctly and failed
validation. The namespace was simultaneously too permissive (the data-flow
validator accepted `execution.anything`, plus `pipeline`, `context` and `env`,
none of which anything populates) and, through the template validator, too
strict.

These tests pin the contract from both sides: what the namespace offers, that
it offers the same thing throughout a run, and that everything else is
refused.
"""

import json
import os
import subprocess
import sys
from datetime import timezone
from pathlib import Path

import pytest

from orchestrator.core.runtime_context import (
    EXECUTION_FIELDS,
    RUNTIME_NAMESPACE,
    RuntimeContext,
    execution_namespace_for,
)

pytestmark = [pytest.mark.contract]

REPO = Path(__file__).resolve().parent.parent


def _cli(command, pipeline, cwd, *extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("ANTHROPIC_API_KEY", None)
    env["ORCHESTRATOR_AUTO_INSTALL"] = "0"
    return subprocess.run(
        [sys.executable, "-m", "orchestrator.cli", command, str(pipeline), *extra],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=300,
    )


def _pipeline(*expressions):
    steps = "\n".join(
        f"""  - id: step_{i}
    tool: filesystem
    action: write
    parameters:
      path: "./out_{i}.txt"
      content: "{expression}"
"""
        + (f"    dependencies:\n      - step_{i - 1}\n" if i else "")
        for i, expression in enumerate(expressions)
    )
    return f"id: rc\nname: RC\nsteps:\n{steps}"


# ---------------------------------------------------------------------------
# The schema
# ---------------------------------------------------------------------------

def test_the_declared_fields_are_the_fields_produced():
    """A schema the validator trusts must match what the runtime renders.

    If they disagree, validation either rejects a field that works or accepts
    one that does not -- the two failure modes this whole namespace had.
    """
    produced = set(RuntimeContext.create("probe").as_template_namespace())
    assert produced == EXECUTION_FIELDS, (
        f"declared_only={sorted(EXECUTION_FIELDS - produced)}, "
        f"produced_only={sorted(produced - EXECUTION_FIELDS)}"
    )


def test_the_run_id_is_carried_through():
    assert RuntimeContext.create("run-77").as_template_namespace()["id"] == "run-77"


def test_a_run_without_an_id_still_gets_one():
    namespace = RuntimeContext.create(None).as_template_namespace()
    assert namespace["id"]


def test_started_at_is_utc():
    """Local time makes two machines' stamps incomparable, and goes backwards
    across a daylight-saving change."""
    assert RuntimeContext.create("x").started_at.tzinfo is timezone.utc
    assert RuntimeContext.create("x").as_template_namespace()["timestamp"].endswith(
        "+00:00"
    )


def test_timestamp_is_started_at_under_its_older_name():
    """The same instant read once, not the clock read twice."""
    namespace = RuntimeContext.create("x").as_template_namespace()
    assert namespace["timestamp"] == namespace["started_at"]


def test_the_namespace_survives_json():
    """A run's context is checkpointed as JSON.

    Caching the `RuntimeContext` object itself made every checkpointed run
    fail with "Object of type RuntimeContext is not JSON serializable", which
    is the sort of thing that only shows up when a run is long enough to
    checkpoint.
    """
    context = {"execution_id": "abc"}
    execution_namespace_for(context)
    json.dumps(context)  # must not raise


# ---------------------------------------------------------------------------
# One run, one answer
# ---------------------------------------------------------------------------

def test_asking_twice_gives_the_same_answer():
    context = {"execution_id": "abc"}
    assert execution_namespace_for(context) == execution_namespace_for(context)


def test_separate_runs_get_separate_answers():
    first = execution_namespace_for({"execution_id": "a"})
    second = execution_namespace_for({"execution_id": "b"})
    assert first["id"] != second["id"]


@pytest.mark.e2e
def test_every_step_of_a_run_reports_the_same_timestamp(tmp_path):
    """The defect, end to end.

    Two steps, one run. Before this, `_execute_level` rebuilt the namespace
    per level and they differed by milliseconds -- enough for anything naming
    an output file by timestamp to write two.
    """
    pipeline = tmp_path / "p.yaml"
    pipeline.write_text(
        _pipeline("{{ execution.timestamp }}", "{{ execution.timestamp }}")
    )

    result = _cli("run", pipeline, tmp_path)
    assert result.returncode == 0, f"{result.stdout[-800:]}{result.stderr[-800:]}"

    first = (tmp_path / "out_0.txt").read_text()
    second = (tmp_path / "out_1.txt").read_text()
    assert first, "the first step wrote nothing"
    assert first == second, (
        f"one run reported two different start times: {first!r} then {second!r}"
    )


# ---------------------------------------------------------------------------
# What validation accepts, the runtime renders -- and the reverse
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.parametrize("field", sorted(EXECUTION_FIELDS))
def test_every_declared_field_both_validates_and_renders(field, tmp_path):
    """A field in the schema must work all the way to the file.

    `execution.timestamp` used to run correctly and fail validation; the
    point of a shared schema is that the two cannot disagree again.
    """
    pipeline = tmp_path / "p.yaml"
    pipeline.write_text(_pipeline(f"{{{{ execution.{field} }}}}"))

    validated = _cli("validate", pipeline, tmp_path)
    ran = _cli("run", pipeline, tmp_path)

    assert validated.returncode == 0, (
        f"execution.{field} is in the schema and was rejected: "
        f"{validated.stdout[-500:]}"
    )
    assert ran.returncode == 0, f"execution.{field} did not run: {ran.stdout[-500:]}"
    assert (tmp_path / "out_0.txt").read_text().strip(), (
        f"execution.{field} rendered as nothing"
    )


@pytest.mark.e2e
@pytest.mark.parametrize(
    "expression",
    [
        # A typo in a real field. Rendering this as an empty string would put
        # a blank where a timestamp belongs and report success.
        "{{ execution.strated_at }}",
        "{{ execution.bogus }}",
        # Namespaces the data-flow validator used to accept, populated by
        # nothing at all.
        "{{ pipeline.name }}",
        "{{ context.foo }}",
        "{{ env.HOME }}",
    ],
)
def test_what_cannot_render_is_refused(expression, tmp_path):
    pipeline = tmp_path / "p.yaml"
    pipeline.write_text(_pipeline(expression))

    validated = _cli("validate", pipeline, tmp_path)
    ran = _cli("run", pipeline, tmp_path)

    assert ran.returncode != 0, f"{expression} ran; the runtime populates it after all"
    assert validated.returncode != 0, (
        f"{expression} cannot render but validates, so the failure waits until "
        f"run time"
    )


@pytest.mark.parametrize(
    "reference", ["pipeline.name", "context.foo", "env.HOME", "execution.bogus"]
)
def test_the_data_flow_validator_refuses_them_on_its_own(reference):
    """Asked directly, not through a whole pipeline.

    Through the CLI these are rejected either way, because the *template*
    validator does not know the names -- so re-adding them here breaks no
    end-to-end test, and the permissiveness sat unnoticed for exactly that
    reason. This asks the line that changed.
    """
    from orchestrator.validation.data_flow_validator import DataFlowValidator

    result = DataFlowValidator()._validate_variable_reference(
        reference,
        task_id="some_task",
        parameter_name="content",
        task_schemas={},
        pipeline_inputs={},
    )
    assert result["valid"] is False, (
        f"{reference} is accepted as a runtime namespace but nothing populates it"
    )


def test_an_unknown_field_names_the_ones_that_exist():
    """An error that lists the alternatives is the difference between a fix
    and a guess."""
    from orchestrator.validation.data_flow_validator import DataFlowValidator

    result = DataFlowValidator()._validate_variable_reference(
        "execution.strated_at",
        task_id="some_task",
        parameter_name="content",
        task_schemas={},
        pipeline_inputs={},
    )
    assert result["valid"] is False
    assert "started_at" in result["message"]


def test_the_namespace_name_is_stated_once():
    assert RUNTIME_NAMESPACE == "execution"


@pytest.mark.e2e
def test_the_run_the_template_sees_is_the_run_the_result_reports(tmp_path):
    """`{{ execution.id }}` and the returned result must name one run.

    Two identifiers for the same execution would make a trace impossible to
    follow across the boundary -- the artifact says one thing, the result
    another.
    """
    pipeline = tmp_path / "p.yaml"
    pipeline.write_text(_pipeline("{{ execution.id }}"))

    result = _cli("run", pipeline, tmp_path)
    assert result.returncode == 0, f"{result.stdout[-800:]}{result.stderr[-800:]}"

    rendered = (tmp_path / "out_0.txt").read_text().strip()
    reported = json.loads(result.stdout[result.stdout.index("{"):])["execution_id"]
    assert rendered == reported, (
        f"the template saw run {rendered!r}, the result reports {reported!r}"
    )
