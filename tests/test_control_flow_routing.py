"""Control-flow routing: `on_false`, `on_success`, `on_failure` (#333).

A step may say where execution goes next. Routing jumps forward, marking the
steps in between as skipped -- the same machinery `goto` already used, rather
than a second one beside it.

The awkward part is `on_failure`, which already meant a *failure policy*
(`fail` / `continue` / `skip` / `retry`) and now also names a step to jump to.
Both are supported, disambiguated by value, and the ambiguous case -- a step
whose id *is* a policy word -- is refused at compile time rather than guessed.
"""

import asyncio
import os
from pathlib import Path

import pytest

from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.core.exceptions import YAMLCompilerError
from orchestrator.core.routing import FAILURE_POLICIES, is_failure_policy, routing_targets
from orchestrator.models.model_registry import ModelRegistry
from tests.test_infrastructure import MockTestModel, create_test_orchestrator

pytestmark = [pytest.mark.contract, pytest.mark.e2e]


def _compiler():
    registry = ModelRegistry()
    registry.register_model(MockTestModel())
    return YAMLCompiler(model_registry=registry)


def _run(yaml_content, cwd):
    previous = Path.cwd()
    os.chdir(cwd)
    try:
        return asyncio.run(
            create_test_orchestrator().execute_yaml(yaml_content=yaml_content, context={})
        )
    finally:
        os.chdir(previous)


def _write_step(step_id, filename, *, depends_on=None, **extra):
    lines = [
        f"  - id: {step_id}",
        "    tool: filesystem",
        "    action: write",
        "    parameters:",
        f'      path: "out/{filename}"',
        f'      content: "{step_id}-ran"',
    ]
    if depends_on:
        lines.append(f"    dependencies: [{depends_on}]")
    for key, value in extra.items():
        lines.append(f"    {key}: {value}")
    return "\n".join(lines)


def _conditional_pipeline(condition):
    """check -> (middle) -> recover, where `check` routes on_false to recover."""
    return f"""
id: routing_pipeline
name: Routing Pipeline
steps:
  - id: check
    action: evaluate_condition
    condition: "{condition}"
    parameters:
      condition: "1 == 1"
    on_false: recover
{_write_step("middle", "middle.txt", depends_on="check")}
{_write_step("recover", "recover.txt", depends_on="middle")}
"""


# ---------------------------------------------------------------------------
# on_false
# ---------------------------------------------------------------------------

def test_on_false_jumps_and_skips_the_steps_in_between(tmp_path):
    result = _run(_conditional_pipeline("false"), tmp_path)

    assert result.steps["check"].status == "skipped"
    assert result.steps["middle"].status == "skipped", (
        "the step routed over must be skipped, not run"
    )
    assert result.steps["recover"].success is True

    assert not (tmp_path / "out" / "middle.txt").exists(), (
        "a skipped step must not have had its side effect"
    )
    assert (tmp_path / "out" / "recover.txt").read_text() == "recover-ran"


def test_a_routed_pipeline_still_reports_success(tmp_path):
    """Skipping is not failing. A pipeline that routed around a step worked."""
    result = _run(_conditional_pipeline("false"), tmp_path)

    assert result.success is True
    assert result.status == "completed"
    assert [s.id for s in result.failed_steps] == []
    assert sorted(s.id for s in result.skipped_steps) == ["check", "middle"]


def test_on_false_does_not_fire_when_the_condition_holds(tmp_path):
    """The branch not taken must actually not be taken."""
    result = _run(_conditional_pipeline("true"), tmp_path)

    assert result.steps["check"].status == "completed"
    assert result.steps["middle"].success is True, "the normal path must run"
    assert (tmp_path / "out" / "middle.txt").exists()


# ---------------------------------------------------------------------------
# on_success
# ---------------------------------------------------------------------------

def test_on_success_jumps_past_the_intervening_step(tmp_path):
    pipeline = f"""
id: on_success_pipeline
name: On Success Pipeline
steps:
{_write_step("first", "first.txt", on_success="last")}
{_write_step("skipped_middle", "middle.txt", depends_on="first")}
{_write_step("last", "last.txt", depends_on="skipped_middle")}
"""
    result = _run(pipeline, tmp_path)

    assert result.steps["first"].success is True
    assert result.steps["skipped_middle"].status == "skipped"
    assert result.steps["last"].success is True
    assert not (tmp_path / "out" / "middle.txt").exists()


# ---------------------------------------------------------------------------
# on_failure: policy or step id
# ---------------------------------------------------------------------------

def test_on_failure_naming_a_step_routes_instead_of_aborting(tmp_path):
    pipeline = f"""
id: on_failure_routing
name: On Failure Routing
steps:
  - id: boom
    tool: filesystem
    action: read
    on_failure: handler
    parameters:
      path: "/nonexistent/definitely/not/here.txt"
{_write_step("never", "never.txt", depends_on="boom")}
{_write_step("handler", "handler.txt", depends_on="never")}
"""
    result = _run(pipeline, tmp_path)

    assert result.steps["boom"].success is False
    assert result.steps["handler"].success is True, (
        "routing on failure must reach the handler"
    )
    assert not (tmp_path / "out" / "never.txt").exists()


@pytest.mark.parametrize("policy", sorted(FAILURE_POLICIES))
def test_reserved_policy_words_keep_their_policy_meaning(policy):
    """`on_failure: continue` is a policy, not a jump to a step called continue."""
    assert is_failure_policy(policy)
    assert routing_targets({"id": "s", "on_failure": policy}) == {}


def test_a_non_policy_on_failure_is_a_routing_target():
    assert routing_targets({"id": "s", "on_failure": "handler"}) == {
        "on_failure": "handler"
    }


# ---------------------------------------------------------------------------
# compile-time validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["on_false", "on_success", "on_failure"])
def test_routing_to_a_step_that_does_not_exist_fails_to_compile(key):
    """Exit 2 naming the bad target, not a run that jumps into nothing."""
    pipeline = f"""
id: bad_routing
name: Bad Routing
steps:
{_write_step("only", "only.txt", **{key: "nowhere"})}
"""
    with pytest.raises(YAMLCompilerError) as excinfo:
        asyncio.run(_compiler().compile(pipeline, {}, resolve_ambiguities=False))

    assert "nowhere" in str(excinfo.value)


def test_routing_to_itself_fails_to_compile():
    pipeline = f"""
id: self_routing
name: Self Routing
steps:
{_write_step("only", "only.txt", on_success="only")}
"""
    with pytest.raises(YAMLCompilerError) as excinfo:
        asyncio.run(_compiler().compile(pipeline, {}, resolve_ambiguities=False))

    assert "itself" in str(excinfo.value)


def test_a_step_named_after_a_failure_policy_is_refused():
    """The one genuinely ambiguous case, refused rather than guessed at.

    With a step called `retry`, `on_failure: retry` could mean either the
    policy or a jump to that step, and no rule can tell them apart.
    """
    pipeline = f"""
id: colliding_ids
name: Colliding Ids
steps:
{_write_step("first", "first.txt")}
{_write_step("retry", "retry.txt", depends_on="first")}
"""
    with pytest.raises(YAMLCompilerError) as excinfo:
        asyncio.run(_compiler().compile(pipeline, {}, resolve_ambiguities=False))

    message = str(excinfo.value)
    assert "retry" in message and "collides" in message


def test_valid_routing_compiles_and_is_carried_on_the_task():
    pipeline = _conditional_pipeline("false")
    compiled = asyncio.run(_compiler().compile(pipeline, {}, resolve_ambiguities=False))

    assert compiled.tasks["check"].metadata["on_false"] == "recover"
