"""Failure-policy semantics: timeout, retry, and what happens after.

These were the last unpinned part of the execution contract. Everything here
was already implemented; none of it was asserted anywhere, so none of it was
safe to describe in the documentation.

Measured, not assumed: a timeout is retried like any other failure, so a step's
worst-case wall time is roughly `timeout x (max_retries + 1)`.

There is deliberately no test for automatic model fallback. It does not exist
on the canonical path -- `ModelRegistry.select_model` raises
`NoEligibleModelsError` rather than substituting a different model -- and the
last test in this file pins that fail-closed behaviour, because silently
selecting an unrequested model is how a cost-control policy gets bypassed.
"""

import asyncio
import os
import time
from pathlib import Path

import pytest

from orchestrator.core.exceptions import NoEligibleModelsError
from orchestrator.models.model_registry import ModelRegistry
from tests.test_infrastructure import create_test_orchestrator

pytestmark = [pytest.mark.contract, pytest.mark.e2e]


def _run(yaml_content, cwd):
    previous = Path.cwd()
    os.chdir(cwd)
    try:
        return asyncio.run(
            create_test_orchestrator().execute_yaml(yaml_content=yaml_content, context={})
        )
    finally:
        os.chdir(previous)


def _slow_step_pipeline(timeout, max_retries, sleep_for=3):
    """A step that cannot finish inside its timeout, using a real subprocess."""
    return f"""
id: timeout_pipeline
name: Timeout Pipeline
steps:
  - id: slow
    tool: terminal
    action: execute
    timeout: {timeout}
    max_retries: {max_retries}
    on_failure: continue
    parameters:
      command: "sleep {sleep_for}"
"""


# ---------------------------------------------------------------------------
# timeout
# ---------------------------------------------------------------------------

def test_a_step_that_exceeds_its_timeout_fails(tmp_path):
    result = _run(_slow_step_pipeline(timeout=1, max_retries=0), tmp_path)

    step = result.steps["slow"]
    assert step.success is False
    assert step.error, "a timed-out step must say why"
    assert "timeout" in step.error.lower()


def test_a_timeout_is_distinguishable_from_an_ordinary_failure(tmp_path):
    """"Raise the timeout" and "fix the step" are different responses."""
    result = _run(_slow_step_pipeline(timeout=1, max_retries=0), tmp_path)

    step = result.steps["slow"]
    assert step.error_type == "TimeoutError"
    assert step.timed_out is True
    assert step.to_dict()["timed_out"] is True


def test_a_step_that_finishes_in_time_is_not_marked_timed_out(tmp_path):
    quick = """
id: quick_pipeline
name: Quick Pipeline
steps:
  - id: quick
    tool: terminal
    action: execute
    timeout: 30
    parameters:
      command: "echo done"
"""
    result = _run(quick, tmp_path)

    assert result.steps["quick"].timed_out is False


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------

def test_retries_are_bounded_by_max_retries(tmp_path):
    """`Task.reset()` deliberately does not clear `retry_count`.

    If it did, every retry would restore the budget and a permanently failing
    step would retry for ever. This pins that it does not.
    """
    result = _run(_slow_step_pipeline(timeout=1, max_retries=2), tmp_path)

    # Measured: `max_retries` bounds total ATTEMPTS, not retries beyond the
    # first, so 2 means two attempts and therefore one retry. The name and the
    # behaviour disagree; the behaviour is pinned here so changing it has to be
    # a deliberate decision rather than a silent one.
    assert result.steps["slow"].retries == 1


def test_max_retries_zero_means_a_single_attempt(tmp_path):
    result = _run(_slow_step_pipeline(timeout=1, max_retries=0), tmp_path)

    assert result.steps["slow"].retries == 0


def test_a_timeout_is_retried_so_wall_time_multiplies(tmp_path):
    """The consequence worth knowing before setting these numbers.

    A timeout is an exception like any other, so it goes through the same
    retry path. `timeout: 1` with `max_retries: 2` is three attempts of about
    a second each, not one second total.
    """
    started = time.monotonic()
    result = _run(_slow_step_pipeline(timeout=1, max_retries=2), tmp_path)
    elapsed = time.monotonic() - started

    assert result.steps["slow"].retries == 1
    assert elapsed >= 2.0, (
        f"three one-second attempts cannot take {elapsed:.1f}s -- if this "
        f"fails, the timeout is no longer being retried"
    )
    # Generous ceiling: this asserts the bound exists, not the exact timing.
    assert elapsed < 15.0, f"retrying took far longer than the bound: {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# what happens after a failure
# ---------------------------------------------------------------------------

def test_on_failure_continue_lets_the_run_finish(tmp_path):
    pipeline = """
id: continue_pipeline
name: Continue Pipeline
steps:
  - id: boom
    tool: filesystem
    action: read
    on_failure: continue
    parameters:
      path: "/nonexistent/definitely/not/here.txt"
  - id: after
    tool: filesystem
    action: write
    dependencies: [boom]
    parameters:
      path: "out/after.txt"
      content: "ran anyway"
"""
    result = _run(pipeline, tmp_path)

    assert result.steps["boom"].success is False
    assert result.steps["after"].success is True, (
        "`continue` must let the following step run"
    )
    # The pipeline still reports failure: a step in it did not succeed.
    assert result.success is False
    assert [s.id for s in result.failed_steps] == ["boom"]


def test_a_non_raising_failure_does_not_trigger_the_failure_policy(tmp_path):
    """The measured behaviour, and a real gap in it.

    `fail` is the default policy, but it only fires for a step that *raised*.
    A tool returning {"success": False} without raising leaves its task
    COMPLETED, so the policy never sees it and the run continues -- the
    downstream step below runs on a failure the pipeline author asked to abort
    on.

    The failure is not lost: it surfaces in the result and in the exit code.
    But "fail fast" does not currently fail fast for this class of failure.

    Making the policy consult StepResult.success was tried and reverted,
    because the policy aborts by raising and the run then produced no result
    document at all -- discarding the trace exactly when it is most wanted.
    Closing this properly means the execution loop stops scheduling rather
    than throwing, and still returns a PipelineResult. Recorded in ADR 0001.
    """
    pipeline = """
id: default_policy_pipeline
name: Default Policy Pipeline
steps:
  - id: boom
    tool: filesystem
    action: read
    parameters:
      path: "/nonexistent/definitely/not/here.txt"
  - id: after
    tool: filesystem
    action: write
    dependencies: [boom]
    parameters:
      path: "out/after.txt"
      content: "ran despite the default policy"
"""
    result = _run(pipeline, tmp_path)

    assert result.steps["boom"].success is False
    assert result.success is False, "the run must still report failure"
    assert [s.id for s in result.failed_steps] == ["boom"]

    # The gap, pinned so that closing it is a visible change rather than a
    # silent one: the downstream step ran anyway.
    assert result.steps["after"].success is True
    assert (tmp_path / "out" / "after.txt").exists()


# ---------------------------------------------------------------------------
# no automatic model fallback
# ---------------------------------------------------------------------------

def test_model_selection_fails_closed_rather_than_substituting():
    """There is no automatic fallback, and that is the safe behaviour.

    Quietly selecting a model the pipeline did not ask for is how a
    cost-control policy gets bypassed -- the pipeline believes it ran on the
    free model it requested. Selection raises instead, so an unsatisfiable
    requirement is an error the caller sees.
    """
    registry = ModelRegistry()  # deliberately empty

    with pytest.raises(NoEligibleModelsError):
        asyncio.run(registry.select_model({"tasks": ["generate"]}))
