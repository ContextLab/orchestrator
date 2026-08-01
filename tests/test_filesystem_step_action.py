"""A pipeline step must keep the tool operation it asked for.

A step carries two different `action` values that are easy to confuse:

- the **step** action (`action: file`) selects which tool runs;
- the **parameter** action (`parameters.action: read`) selects what that tool
  does.

The control system used to copy the first over the second, so `FileSystemTool`
received `action="file"` -- not one of its operations -- and every such step
failed with "Unknown filesystem action: file". Reading a file from a pipeline
was impossible.

Real files, real tool, no mocks.
"""

import json

import pytest

from orchestrator.control_systems.hybrid_control_system import HybridControlSystem
from orchestrator.core.task import Task
from orchestrator.models.model_registry import ModelRegistry

pytestmark = pytest.mark.unit


@pytest.fixture
def control_system():
    """A control system with an empty registry -- no model is needed here."""
    return HybridControlSystem(model_registry=ModelRegistry())


def _task(step_action, **parameters):
    """Build a task. `step_action` is the routing action, kwargs the params.

    Named `step_action` rather than `action` precisely because the two
    collide -- which is the confusion this whole module exists to pin down.
    """
    return Task(id="step", name="step", action=step_action, parameters=parameters)


@pytest.mark.asyncio
async def test_read_operation_survives_a_file_step_action(control_system, tmp_path):
    """The regression: `action: file` must not overwrite `action: read`."""
    target = tmp_path / "data.json"
    target.write_text(json.dumps({"items": [1, 2, 3]}))

    result = await control_system._handle_file_operation(
        _task("file", action="read", path=str(target)), {}
    )

    assert result["success"] is True, f"read failed: {result['error']}"
    assert result["result"]["action"] == "read", (
        "the step's routing action overwrote the requested operation"
    )
    assert json.loads(result["result"]["content"]) == {"items": [1, 2, 3]}


@pytest.mark.asyncio
async def test_write_operation_survives_a_file_step_action(control_system, tmp_path):
    target = tmp_path / "out.txt"

    result = await control_system._handle_file_operation(
        _task("file", action="write", path=str(target), content="written"), {}
    )

    assert result["success"] is True, f"write failed: {result['error']}"
    assert target.read_text() == "written"


@pytest.mark.asyncio
async def test_list_operation_survives_a_file_step_action(control_system, tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")

    result = await control_system._handle_file_operation(
        _task("file", action="list", path=str(tmp_path)), {}
    )

    assert result["success"] is True, f"list failed: {result['error']}"
    assert {item["name"] for item in result["result"]["items"]} == {"a.txt", "b.txt"}


@pytest.mark.asyncio
async def test_step_action_still_supplies_the_operation_when_absent(
    control_system, tmp_path
):
    """The fix must not break the case it was originally written for.

    With no `parameters.action`, the step action IS the operation, and that
    behaviour has to survive.
    """
    target = tmp_path / "data.txt"
    target.write_text("hello")

    result = await control_system._handle_file_operation(
        _task("read", path=str(target)), {}
    )

    assert result["success"] is True, f"read failed: {result['error']}"
    assert result["result"]["content"] == "hello"
