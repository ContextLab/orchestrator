"""The typed result of a pipeline run.

`Orchestrator.execute_pipeline` used to return a bare `{step_id: value}` dict --
except when the pipeline declared `outputs:`, in which case it returned
`{"steps": ..., "outputs": ...}` instead. Two different shapes from one method,
distinguishable only by inspecting the keys, and a pipeline with a step called
`outputs` collided with the second. Everything else the run knew -- how long
each step took, which model answered, what failed and why, what order things
ran in -- was computed and then dropped on the floor.

`PipelineResult` is that state, kept. It is a `Mapping`, so `result["step_id"]`
still returns exactly what it always did and existing pipelines and tests are
unaffected; the trace arrives as attributes alongside.

`to_dict()` is the stable serialisation the CLI emits, so a CLI run and a
Python run of the same pipeline can be compared as data rather than by fishing
selected values out of nested dicts.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .task import Task, TaskStatus


#: Fields whose values are a property of *when* a run happened rather than of
#: what it did. Two runs of the same pipeline differ here and nowhere else.
RUN_SPECIFIC_FIELDS: Tuple[str, ...] = ("started_at", "completed_at", "duration")


def normalize_result_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Blank the run-specific fields of a serialised result, leaving the rest.

    The CLI emits `to_dict()` as JSON and the Python API returns a
    `PipelineResult`, so comparing the two surfaces means comparing a plain
    dict against an object. This works on the dict form, which both can reach,
    and is the *only* definition of what "run-specific" means -- a second
    definition living in a test is how a comparison quietly stops comparing
    anything. Everything not blanked here, `outputs` and every step `value`
    included, is behaviour the two surfaces must agree on exactly.
    """
    data = deepcopy(data)
    data["execution_id"] = "<execution-id>"
    for key in RUN_SPECIFIC_FIELDS:
        data[key] = None
    for step in data.get("steps", {}).values():
        for key in RUN_SPECIFIC_FIELDS:
            step[key] = None
    return data


def _describe_error(error: Any) -> Tuple[Optional[str], Optional[str]]:
    """(message, type name) for whatever a task recorded as its error."""
    if error is None:
        return None, None
    if isinstance(error, BaseException):
        return str(error), type(error).__name__
    return str(error), type(error).__name__


@dataclass(frozen=True)
class StepResult:
    """What one step did.

    `value` is the step's raw result -- the same object `result["step_id"]`
    has always returned. Everything else is the trace around it.
    """

    id: str
    action: str
    status: str
    success: bool
    value: Any = None

    error: Optional[str] = None
    error_type: Optional[str] = None

    #: Which tool ran it, when a tool did.
    tool: Optional[str] = None
    #: Which model answered, and from which provider, when a model did.
    model: Optional[str] = None
    provider: Optional[str] = None

    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    #: Seconds, or None when the step never started.
    duration: Optional[float] = None

    #: Attempts *after* the first. `Task.retry_count` counts the initial
    #: failure too, so it is not the same number.
    retries: int = 0
    dependencies: Tuple[str, ...] = ()

    @classmethod
    def from_task(cls, task: Task, value: Any = None) -> "StepResult":
        message, error_type = _describe_error(task.error)
        duration = None
        if task.started_at is not None and task.completed_at is not None:
            duration = task.completed_at - task.started_at

        # A model step records the model it used in its own envelope; there is
        # nowhere else the identity survives.
        model = provider = None
        reported_failure = False
        if isinstance(value, Mapping):
            model = value.get("model_used") or value.get("model")
            provider = value.get("provider")
            # A step can fail *without raising*: a tool returns
            # {"success": False, "error": ...} and the run continues. The task
            # is then COMPLETED -- it did finish -- but the step did not
            # succeed, and reading only the task status reports a failing
            # pipeline as successful.
            reported_failure = value.get("success") is False
            if reported_failure and message is None:
                message, error_type = _describe_error(value.get("error"))

        succeeded = task.status is TaskStatus.COMPLETED and not reported_failure

        return cls(
            id=task.id,
            action=str(task.action),
            status=task.status.value,
            success=succeeded,
            value=value,
            error=message,
            error_type=error_type,
            tool=task.metadata.get("tool"),
            model=model,
            provider=provider,
            started_at=task.started_at,
            completed_at=task.completed_at,
            duration=duration,
            retries=max(0, task.retry_count - 1) if task.error else task.retry_count,
            dependencies=tuple(task.dependencies),
        )

    @property
    def timed_out(self) -> bool:
        """Whether this step ran out of time rather than failing on its merits.

        A timeout is retried like any other failure, so a step with
        `timeout: 2` and `max_retries: 3` can occupy roughly eight seconds
        before giving up. Distinguishing it from an ordinary failure is the
        difference between "make the timeout bigger" and "fix the step".
        """
        return self.error_type == "TimeoutError"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action,
            "status": self.status,
            "success": self.success,
            "value": self.value,
            "error": self.error,
            "error_type": self.error_type,
            "tool": self.tool,
            "model": self.model,
            "provider": self.provider,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.duration,
            "retries": self.retries,
            "timed_out": self.timed_out,
            "dependencies": list(self.dependencies),
        }


@dataclass(frozen=True)
class PipelineResult(Mapping):
    """The typed result of one pipeline run.

    Behaves as a mapping of step id -> that step's raw value, which is what
    `execute_pipeline` has always returned, so existing callers keep working.
    """

    pipeline_id: str
    execution_id: str
    status: str
    success: bool

    steps: Dict[str, StepResult] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)

    execution_order: Tuple[str, ...] = ()
    #: Steps grouped by dependency level; members of a level may run together.
    execution_levels: Tuple[Tuple[str, ...], ...] = ()

    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration: Optional[float] = None

    #: Pipeline-level failure, when the run itself aborted.
    error: Optional[str] = None
    error_type: Optional[str] = None

    # -- Mapping: backwards compatible access ------------------------------

    def __getitem__(self, key: str) -> Any:
        return self.steps[key].value

    def __iter__(self) -> Iterator[str]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    # -- the trace ---------------------------------------------------------

    @property
    def failed_steps(self) -> List[StepResult]:
        """Steps that did not succeed, whether they raised or reported it.

        Selecting on `success` rather than on status is deliberate: a tool that
        returns `{"success": False}` leaves its task COMPLETED, and a pipeline
        containing one is not a successful pipeline.
        """
        return [
            s
            for s in self.steps.values()
            if not s.success and s.status != TaskStatus.SKIPPED.value
        ]

    @property
    def skipped_steps(self) -> List[StepResult]:
        return [s for s in self.steps.values() if s.status == TaskStatus.SKIPPED.value]

    @property
    def retried_steps(self) -> List[StepResult]:
        return [s for s in self.steps.values() if s.retries > 0]

    def to_dict(self) -> Dict[str, Any]:
        """Stable serialisation. The CLI emits exactly this."""
        return {
            "pipeline_id": self.pipeline_id,
            "execution_id": self.execution_id,
            "status": self.status,
            "success": self.success,
            "steps": {sid: step.to_dict() for sid, step in self.steps.items()},
            "outputs": self.outputs,
            "execution_order": list(self.execution_order),
            "execution_levels": [list(level) for level in self.execution_levels],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.duration,
            "error": self.error,
            "error_type": self.error_type,
        }

    def normalized(self) -> Dict[str, Any]:
        """`to_dict()` with everything run-specific removed.

        Wall-clock times and the execution id differ between any two runs, so
        comparing raw results would always fail. This is the form the CLI and
        the Python API must agree on byte for byte.
        """
        return normalize_result_payload(self.to_dict())
