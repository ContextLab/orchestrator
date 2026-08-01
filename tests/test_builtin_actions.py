"""The built-in action registry, and the two consumers that must agree on it.

`validate` and `run` used to hold separate notions of what an action is. The
validator treated a step's `action:` as a tool name whenever the step had no
`tool:` key, so `action: generate` was rejected as a missing *tool* while the
executor ran the same step correctly (#241).

`core.actions` is now the single source of truth. These tests pin both
directions of that: the executor dispatches every name in the registry to a
handler that exists, and the validator accepts every name in the registry.
"""

import asyncio

import pytest

from orchestrator.core.actions import (
    BUILTIN_ACTION_HANDLERS,
    BUILTIN_ACTIONS,
    STRUCTURED_ACTIONS,
    is_builtin_action,
)
from orchestrator.control_systems.hybrid_control_system import HybridControlSystem
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.validation.tool_validator import ToolValidator
from tests.test_infrastructure import MockTestModel

pytestmark = [pytest.mark.contract]


@pytest.fixture
def control_system():
    registry = ModelRegistry()
    registry.register_model(MockTestModel())
    return HybridControlSystem(model_registry=registry)


@pytest.fixture
def validator():
    """A strict validator: unknown tools are errors, not warnings."""
    return ToolValidator(allow_unknown_tools=False)


# ---------------------------------------------------------------------------
# the registry cannot promise what the executor cannot deliver
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("action", sorted(BUILTIN_ACTION_HANDLERS))
def test_every_registered_action_has_a_real_handler(action, control_system):
    """Dispatch is driven off this mapping, so a bad entry is an AttributeError.

    Catching it here names the offending action instead of surfacing as a
    mystery failure inside whichever pipeline happened to use it.
    """
    handler_name = BUILTIN_ACTION_HANDLERS[action]
    handler = getattr(control_system, handler_name, None)

    assert handler is not None, (
        f"action {action!r} is registered to {handler_name!r}, "
        f"which does not exist on HybridControlSystem"
    )
    assert callable(handler), f"{handler_name!r} is not callable"


def test_structured_actions_are_builtin_but_handled_one_level_down():
    """They have no handler entry: ModelBasedControlSystem runs them.

    They must still be in BUILTIN_ACTIONS or the validator would reject them
    as unknown tools -- the original #241 failure, in a different action.
    """
    assert STRUCTURED_ACTIONS <= BUILTIN_ACTIONS
    assert not (STRUCTURED_ACTIONS & set(BUILTIN_ACTION_HANDLERS))


def test_is_builtin_action_matches_the_executors_normalisation():
    """The executor lowercases and strips before dispatching; so must this."""
    assert is_builtin_action("generate")
    assert is_builtin_action("  GENERATE  ")
    assert is_builtin_action("Generate_Text")
    assert not is_builtin_action("summarise the report in three sentences")
    assert not is_builtin_action("")


# ---------------------------------------------------------------------------
# the validator accepts exactly what the executor runs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("action", sorted(BUILTIN_ACTIONS))
def test_validator_accepts_every_builtin_action(action, validator):
    """No built-in action may be reported as a missing tool."""
    result = validator.validate_pipeline_tools(
        {"steps": [{"id": "step_one", "action": action, "parameters": {}}]}
    )

    unknown = [e for e in result.errors if e.error_type == "unknown_tool"]
    assert not unknown, (
        f"built-in action {action!r} was rejected as a tool: "
        f"{[e.message for e in unknown]}"
    )


def test_validator_still_rejects_an_action_that_is_neither_tool_nor_builtin(validator):
    """The fix must not turn the validator into a rubber stamp."""
    result = validator.validate_pipeline_tools(
        {"steps": [{"id": "step_one", "action": "definitely_not_a_real_thing"}]}
    )

    assert [e for e in result.errors if e.error_type == "unknown_tool"], (
        "an unknown action still has to be reported"
    )


def test_validator_still_validates_the_legacy_single_field_form(validator):
    """`action: filesystem` names the tool itself and keeps full validation.

    `filesystem` is both a registered tool and a built-in action name. The tool
    lookup has to win, or this form would silently lose its parameter checks.
    """
    result = validator.validate_pipeline_tools(
        {"steps": [{"id": "step_one", "action": "filesystem", "parameters": {}}]}
    )

    assert "filesystem" in result.tool_availability, (
        "the legacy form must still be resolved as a tool, not skipped as an action"
    )
    assert result.tool_availability["filesystem"] is True


def test_a_tool_step_is_unaffected(validator):
    """The two-field form still resolves `tool:`, never `action:`."""
    result = validator.validate_pipeline_tools(
        {
            "steps": [
                {
                    "id": "step_one",
                    "tool": "filesystem",
                    "action": "read",
                    "parameters": {"path": "x.txt"},
                }
            ]
        }
    )

    assert "filesystem" in result.tool_availability
    assert "read" not in result.tool_availability, (
        "`read` is an operation on the tool, not a tool to look up"
    )


# ---------------------------------------------------------------------------
# end to end: the two agree on the same document
# ---------------------------------------------------------------------------

def test_builtin_action_dispatches_away_from_the_model(control_system):
    """A built-in action must reach its handler, not the prompt fallback.

    The executor's last resort turns an unrecognised action into natural
    language for the model. That is how `generate_structured` returned a
    sentence instead of an object for as long as it did, so the boundary is
    pinned: `evaluate_condition` is a marker action with no model involvement,
    and reaching the model would produce prose instead of a verdict.
    """
    from orchestrator.core.task import Task

    task = Task(
        id="check",
        name="check",
        action="evaluate_condition",
        parameters={"condition": "1 == 1"},
    )
    result = asyncio.run(control_system._execute_task_impl(task, {}))

    assert isinstance(result, dict), f"expected the handler's verdict, got {type(result)}"
    assert "result" in result or "condition" in result, (
        f"this does not look like _handle_evaluate_condition output: {result}"
    )
