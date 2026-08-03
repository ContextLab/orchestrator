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
    BARE_RUNTIME_NAMES,
    CONTEXT_KEY,
    EXECUTION_FIELDS,
    RUNTIME_NAMESPACE,
    RuntimeContext,
    execution_namespace_for,
    new_execution_id,
    runtime_context_for,
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


# ---------------------------------------------------------------------------
# The same run, under its bare names
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.parametrize("name", sorted(BARE_RUNTIME_NAMES))
def test_a_bare_runtime_name_both_validates_and_renders(name, tmp_path):
    """`{{ pipeline_id }}` is registered by the runtime and rendered correctly.

    All three were reported as undefined variables -- the same false positive
    `execution.timestamp` had before it was declared, just without the prefix.
    """
    pipeline = tmp_path / "p.yaml"
    pipeline.write_text(_pipeline(f"{{{{ {name} }}}}"))

    validated = _cli("validate", pipeline, tmp_path)
    ran = _cli("run", pipeline, tmp_path)

    assert ran.returncode == 0, f"{name} did not run: {ran.stdout[-400:]}"
    assert (tmp_path / "out_0.txt").read_text().strip(), f"{name} rendered nothing"
    assert validated.returncode == 0, (
        f"{name} renders correctly but validation rejects it: "
        f"{validated.stdout[-400:]}"
    )


@pytest.mark.e2e
def test_the_bare_timestamp_is_the_run_s_own(tmp_path):
    """`{{ timestamp }}` and `{{ execution.timestamp }}` are one instant.

    They were two readings of two different clocks: `TemplateManager` seeds a
    base context from `datetime.now()` when the *manager* is constructed --
    neither the run's start nor the same value, and in local time without a
    zone while the run context is UTC:

        bare: 2026-08-03T08:07:56.022350
        exec: 2026-08-03T12:07:56.022379+00:00

    Four hours and 29 microseconds apart, in one run.
    """
    pipeline = tmp_path / "p.yaml"
    pipeline.write_text(_pipeline("{{ timestamp }}|{{ execution.timestamp }}"))

    result = _cli("run", pipeline, tmp_path)
    assert result.returncode == 0, f"{result.stdout[-400:]}"

    bare, prefixed = (tmp_path / "out_0.txt").read_text().split("|")
    assert bare == prefixed, (
        f"one run reported two start times: bare={bare!r} execution={prefixed!r}"
    )


@pytest.mark.e2e
@pytest.mark.parametrize("name", ["current_timestamp", "current_date"])
def test_a_name_the_runtime_does_not_provide_is_still_refused(name, tmp_path):
    """Declaring three names must not wave through every bare name.

    These two appear in the catalogue as often as the real ones and are
    populated by nothing.
    """
    pipeline = tmp_path / "p.yaml"
    pipeline.write_text(_pipeline(f"{{{{ {name} }}}}"))

    assert _cli("run", pipeline, tmp_path).returncode != 0, (
        f"{name} runs after all; it should be declared rather than refused"
    )
    assert _cli("validate", pipeline, tmp_path).returncode != 0


# ---------------------------------------------------------------------------
# The run owns its identity; dictionaries only receive projections of it
# ---------------------------------------------------------------------------

def _state(pipeline_id="p"):
    from orchestrator.runtime.execution_state import PipelineExecutionState

    return PipelineExecutionState(pipeline_id=pipeline_id)


def test_reading_a_run_s_context_twice_does_not_start_a_second_run():
    """The defect this section exists for.

    `get_available_context` built a fresh dict per call and let the run's
    identity be established *inside it*, then returned the dict and dropped
    the cache with it. So every read invented a new run::

        first:  2026-08-03T23:15:17.116698+00:00
        second: 2026-08-03T23:15:17.117053+00:00

    A context dict cannot tell whether it is the run or a copy of part of it,
    so it is the wrong owner. The state is the run and holds the object.
    """
    state = _state()
    first, second = state.get_available_context(), state.get_available_context()

    assert first["timestamp"] == second["timestamp"], (
        f"one run reported two start times: {first['timestamp']!r} then "
        f"{second['timestamp']!r}"
    )
    assert first["execution_id"] == second["execution_id"], (
        f"one run reported two identities: {first['execution_id']!r} then "
        f"{second['execution_id']!r}"
    )
    assert first["execution"] == second["execution"]


def test_a_run_s_context_stays_json_serializable():
    """Checkpointing writes it out; an object in there fails the whole run."""
    json.dumps(_state().get_available_context())


def test_elapsed_time_is_measurable_against_the_run_s_start():
    """`start_time` became timezone-aware when it became the run's own.

    Subtracting an aware datetime from a naive one raises `TypeError`, so
    this is the arithmetic that would break silently at the two call sites
    that measure duration.
    """
    state = _state()
    assert state.get_available_context()["execution_time"] >= 0
    assert state.get_execution_summary()["duration_seconds"] >= 0


# ---------------------------------------------------------------------------
# A resumed run is the same run
# ---------------------------------------------------------------------------

def test_a_checkpoint_round_trip_preserves_the_run(tmp_path):
    """Export, restore, and it is still the same run.

    The identity was not exported at all, so a resumed run got a new id and a
    new start time -- the artifacts written after the checkpoint stamped
    differently from the ones written before it, in the one situation where
    you most want to line them up.
    """
    original = _state()
    before = original.get_available_context()

    exported = json.loads(json.dumps(original.export_state()))  # as it travels
    restored = _state(pipeline_id="something-else")
    restored.import_state(exported)
    after = restored.get_available_context()

    assert after["execution_id"] == before["execution_id"]
    assert after["timestamp"] == before["timestamp"]
    assert after["execution"] == before["execution"]


def test_a_checkpoint_written_before_start_times_had_a_zone_still_loads():
    """Older checkpoints hold a naive local `start_time`.

    Reading one must not leave the state unable to subtract its own start
    time from the clock.
    """
    from datetime import datetime as _datetime

    state = _state()
    exported = state.export_state()
    exported["start_time"] = _datetime.now().isoformat()  # naive, as before
    exported.pop("execution_id")

    state.import_state(exported)
    assert state.get_execution_summary()["duration_seconds"] >= 0
    assert state.get_available_context()["execution_id"]


# ---------------------------------------------------------------------------
# Two runs are two runs
# ---------------------------------------------------------------------------

def test_ids_generated_in_one_second_are_distinct():
    """`f"{pipeline.id}_{int(time.time())}"` is unique only until a pipeline
    starts twice inside one second -- routine under a test suite, a scheduler
    or a retry loop. The id names checkpoints, so a collision means the second
    run resumes from the first one's state.
    """
    ids = {new_execution_id("same-pipeline") for _ in range(5000)}
    assert len(ids) == 5000, f"{5000 - len(ids)} of 5000 ids collided"


def test_an_id_still_names_its_pipeline():
    """Uniqueness must not cost traceability: the id is read in logs."""
    assert new_execution_id("my-pipeline").startswith("my-pipeline_")


def test_two_states_are_two_runs():
    assert (
        _state().get_available_context()["execution_id"]
        != _state().get_available_context()["execution_id"]
    )


# ---------------------------------------------------------------------------
# One language, whichever engine runs it
# ---------------------------------------------------------------------------

def test_the_builder_emits_exactly_the_declared_language():
    """`public_names` is the contract every engine is held to.

    If it drifts from what the validators accept, an engine offers a name no
    pipeline may use, or a declared name no engine populates -- both of which
    this namespace has done.
    """
    names = RuntimeContext.create(execution_id="x", pipeline_id="p").public_names()

    assert set(names) == BARE_RUNTIME_NAMES | {RUNTIME_NAMESPACE}
    assert set(names[RUNTIME_NAMESPACE]) == EXECUTION_FIELDS


def test_projection_is_idempotent():
    """Engines project defensively, sometimes more than once per run."""
    runtime = RuntimeContext.create(execution_id="x", pipeline_id="p")
    context = {}
    runtime.project_into(context)
    first = dict(context)
    runtime.project_into(context)
    assert context == first


def test_a_context_that_already_knows_its_run_does_not_start_another():
    context = {"execution_id": "abc", "pipeline_id": "p"}
    execution_namespace_for(context)
    recovered = runtime_context_for(context)

    assert recovered.id == "abc"
    assert recovered.as_template_namespace() == context[CONTEXT_KEY]


def test_the_declarative_engine_offers_the_same_names_as_everyone_else():
    """It offered `start_time` and no `timestamp`, so `{{ execution.timestamp }}`
    rendered under the main orchestrator and nowhere else."""
    from orchestrator.engine.declarative_engine import DeclarativePipelineEngine
    from orchestrator.engine.pipeline_spec import PipelineSpec

    spec = PipelineSpec(
        name="p", steps=[{"id": "s", "action": "generate", "inputs": {}}]
    )
    context = DeclarativePipelineEngine()._initialize_context(spec, {"topic": "x"})

    assert BARE_RUNTIME_NAMES <= set(context), (
        f"the declarative engine does not populate "
        f"{sorted(BARE_RUNTIME_NAMES - set(context))}"
    )
    assert set(context[RUNTIME_NAMESPACE]) == EXECUTION_FIELDS
    assert context["timestamp"] == context[RUNTIME_NAMESPACE]["started_at"]


def test_no_engine_offers_a_namespace_validation_refuses():
    """`pipeline` was populated by two engines and refused by the validators.

    A pipeline written against it ran on those engines and could not be
    validated anywhere -- the contract described one engine, not the product.
    `context` and `env` are refused on the same grounds.
    """
    import re

    refused = ("pipeline", "context", "env")
    pattern = re.compile(
        r"""context\[\s*['"](""" + "|".join(refused) + r""")['"]\s*\]\s*="""
    )

    offenders = [
        f"{path.relative_to(REPO)}:{i}: {line.strip()}"
        for path in (REPO / "src" / "orchestrator").rglob("*.py")
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offenders, (
        "these populate a namespace the validators refuse, so a pipeline "
        f"using it cannot be validated: {offenders}"
    )
