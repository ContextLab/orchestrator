"""The actions the runtime executes itself, without a `tool:`.

A step names either a tool and an operation on it::

    tool: filesystem
    action: read

or an action the runtime handles directly::

    action: generate

Both the executor and the validator need to know which names fall in the
second group, and they used to know separately. `ToolValidator` treated a
step's `action:` as a tool name whenever the step had no `tool:` key, so a
model step was reported as ``Tool 'generate' not found in registry`` while the
executor ran the very same step correctly -- #241, `validate` and `run`
disagreeing about the same document.

This module is the single source of truth. `HybridControlSystem` *dispatches*
off `BUILTIN_ACTION_HANDLERS`, so the mapping cannot claim an action the
executor does not actually run, and the validator accepts exactly the names in
it. Adding a runtime action means adding one entry here; both sides follow.

Not covered here are the natural-language action families -- ``"echo ..."``,
``"write the following content to report.md"`` -- which are matched by pattern
rather than by name. Those are `HybridControlSystem._is_echo_operation` and
`_is_file_operation`.
"""

from __future__ import annotations

from typing import Dict, FrozenSet

# Action name -> the `HybridControlSystem` method that runs it.
#
# Values are method *names* rather than functions because the handlers are
# bound methods on the control system; the dispatcher resolves them per call.
BUILTIN_ACTION_HANDLERS: Dict[str, str] = {
    "control_flow": "_handle_control_flow",
    # `file` and `filesystem` are the two spellings the executor has always
    # accepted for the filesystem handler.
    "file": "_handle_file_operation",
    "filesystem": "_handle_file_operation",
    "process": "_handle_data_processing",
    "validate": "_handle_validation",
    "loop_complete": "_handle_loop_complete",
    "capture_result": "_handle_capture_result",
    "evaluate_condition": "_handle_evaluate_condition",
    "create_parallel_queue": "_handle_create_parallel_queue",
    "action_loop": "_handle_action_loop",
    "analyze": "_handle_analyze_text",
    "analyze_text": "_handle_analyze_text",
    "generate": "_handle_generate_text",
    "generate_text": "_handle_generate_text",
}

# Structured generation is handled one level down, by
# `ModelBasedControlSystem`, so it has no entry above -- but it is still an
# action the runtime executes without a tool, and the validator must accept it.
#
# Every other action here uses underscores. `generate-structured` was the lone
# hyphenated one, and for a long time the only spelling that dispatched: the
# underscore fell through to the natural-language branch, which turns an
# unrecognised action into a *prompt*, so the step returned a sentence instead
# of an object and still reported success. Both spellings are supported.
STRUCTURED_ACTIONS: FrozenSet[str] = frozenset(
    {"generate-structured", "generate_structured"}
)

#: Every action name the runtime executes without a `tool:`.
BUILTIN_ACTIONS: FrozenSet[str] = frozenset(BUILTIN_ACTION_HANDLERS) | STRUCTURED_ACTIONS


def is_builtin_action(action: str) -> bool:
    """Whether `action` names something the runtime executes without a tool.

    Comparison is case-insensitive and ignores surrounding whitespace, matching
    the executor, which lowercases a task's action before dispatching on it.
    """
    return action.strip().lower() in BUILTIN_ACTIONS
