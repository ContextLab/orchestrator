"""The action vocabulary: one authoritative registry, several generated views.

A step names either a tool and an operation on it::

    tool: filesystem
    action: read

or an action the runtime executes itself::

    action: generate
    parameters:
      prompt: "..."

This module owns the second group. Everything downstream is *derived* from
`ACTION_SPECS` rather than restated beside it: the executor's dispatch table,
the validator's notion of a recognised action, the control systems' advertised
`supported_actions`, alias normalisation in the compiler, the documented
vocabulary, and the parametrised contract tests. A name is supported exactly
when there is an `ActionSpec` for it.

The vocabulary had drifted into three disagreeing definitions -- #241 closed
the first pair, this module closes the rest:

1. `HybridControlSystem`'s dispatch chain -- what actually ran.
2. `ModelBasedControlSystem.supported_actions` -- which advertised ten names
   (`transform`, `search`, `extract`, `filter`, `synthesize`, `create`,
   `optimize`, `review`, `write`, `compile`) that had no handler at all.
3. An implicit fallback turning *any* unrecognised action into a prompt, so
   `action: gernate` did not fail -- it silently became a model request and
   reported success.

Number 3 is the dangerous one, and it is gone: an action with no spec is
refused, at compile time and again at dispatch. A census of 130 pipeline
documents found 430 steps dispatched by action name, of which 247 already used
a registered action and 167 relied on the prompt fallback -- including
`generate-text`, a hyphen slip of `generate_text` that had never dispatched.

Two families remain matched by pattern rather than by name, because prose has
no name to register: ``"echo ..."`` and ``"write the following content to
report.md"``. They are declared as `ActionFamily` entries so they belong to
this vocabulary rather than being another undocumented special case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Pattern, Tuple

# --- what a step's result must look like -----------------------------------
#
# Handlers return a mapping carrying the outcome. These are the envelopes the
# contract tests assert against, so a handler cannot quietly change shape.

_TEXT_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["success", "result"],
    "properties": {
        "success": {"type": "boolean"},
        "result": {"type": "string"},
        "action": {"type": "string"},
        "model_used": {"type": "string"},
        "error": {"type": ["string", "null"]},
    },
}

_OBJECT_RESULT_SCHEMA: Dict[str, Any] = {"type": "object"}


@dataclass(frozen=True)
class ActionSpec:
    """One action the runtime executes without a `tool:`."""

    #: The single spelling that appears in the task graph and the trace.
    name: str

    #: One-line description, used to generate the documented vocabulary.
    summary: str

    #: Method on `HybridControlSystem` that runs it. `None` means the action is
    #: handled further down, by `ModelBasedControlSystem`.
    handler: Optional[str] = None

    #: Accepted spellings that normalise to `name`. Kept for compatibility;
    #: the compiler rewrites them and warns.
    aliases: FrozenSet[str] = frozenset()

    #: "model", "tool", or "none" -- what the action needs to do its work.
    requires: str = "none"

    #: Parameters without which the action cannot run. Checked by the
    #: validator, so a missing one fails at compile time rather than at 3am.
    required_parameters: FrozenSet[str] = frozenset()

    #: JSON Schema the step's result must satisfy, where the shape is fixed.
    result_schema: Optional[Dict[str, Any]] = None

    #: Whether this is part of the supported v1 contract. Everything shipped
    #: today is; the flag exists so adding an experimental action later does
    #: not require inventing a second registry for it.
    v1: bool = True

    def __post_init__(self) -> None:
        if self.requires not in {"model", "tool", "none"}:
            raise ValueError(
                f"action {self.name!r}: requires must be 'model', 'tool' or "
                f"'none', got {self.requires!r}"
            )
        if self.name in self.aliases:
            raise ValueError(f"action {self.name!r} lists its own name as an alias")


@dataclass(frozen=True)
class ActionFamily:
    """A family of actions matched by pattern, because they are prose.

    ``"echo Starting run"`` and ``"write the following content to report.md"``
    are instructions, not identifiers. They still dispatch deterministically --
    they simply cannot be looked up by name.
    """

    name: str
    summary: str
    #: Method on `HybridControlSystem`, or `None` for "hand it to the model".
    handler: Optional[str]
    pattern: Pattern[str]
    v1: bool = True


ACTION_SPECS: Tuple[ActionSpec, ...] = (
    # -- model actions ------------------------------------------------------
    ActionSpec(
        name="generate",
        summary="Generate text from a prompt using the selected model.",
        handler="_handle_generate_text",
        aliases=frozenset({"generate_text", "generate-text"}),
        requires="model",
        required_parameters=frozenset({"prompt"}),
        result_schema=_TEXT_RESULT_SCHEMA,
    ),
    ActionSpec(
        name="generate_structured",
        summary="Generate an object conforming to a declared JSON Schema.",
        # No handler entry: ModelBasedControlSystem runs this one.
        aliases=frozenset({"generate-structured"}),
        requires="model",
        required_parameters=frozenset({"prompt", "schema"}),
        result_schema=_OBJECT_RESULT_SCHEMA,
    ),
    ActionSpec(
        name="analyze_text",
        summary="Analyse supplied text with the selected model.",
        handler="_handle_analyze_text",
        aliases=frozenset({"analyze"}),
        requires="model",
    ),
    # -- deterministic runtime actions --------------------------------------
    ActionSpec(
        name="filesystem",
        summary="Filesystem operation named by the step's `action` parameter.",
        handler="_handle_file_operation",
        aliases=frozenset({"file"}),
        requires="tool",
    ),
    ActionSpec(
        name="process",
        summary="Transform data with the deterministic data-processing tool.",
        handler="_handle_data_processing",
        requires="tool",
    ),
    ActionSpec(
        name="validate",
        summary="Validate data against a schema with the validation tool.",
        handler="_handle_validation",
        requires="tool",
    ),
    # -- control flow -------------------------------------------------------
    ActionSpec(
        name="control_flow",
        summary="Control-flow marker step.",
        handler="_handle_control_flow",
    ),
    ActionSpec(
        name="evaluate_condition",
        summary="Evaluate a condition expression and record the verdict.",
        handler="_handle_evaluate_condition",
        required_parameters=frozenset({"condition"}),
    ),
    ActionSpec(
        name="loop_complete",
        summary="Marker emitted when a loop finishes.",
        handler="_handle_loop_complete",
    ),
    ActionSpec(
        name="capture_result",
        summary="Capture a prior step's result under a new name.",
        handler="_handle_capture_result",
    ),
    ActionSpec(
        name="create_parallel_queue",
        summary="Fan work out across a parallel queue.",
        handler="_handle_create_parallel_queue",
    ),
    ActionSpec(
        name="action_loop",
        summary="Run a sequence of actions repeatedly until a condition holds.",
        handler="_handle_action_loop",
    ),
)


ACTION_FAMILIES: Tuple[ActionFamily, ...] = (
    ActionFamily(
        name="echo",
        summary='Print a message, e.g. `action: echo "starting"`.',
        handler="_handle_echo_operation",
        pattern=re.compile(r"^(?:echo|print|display|show|output)\s+", re.IGNORECASE),
    ),
    ActionFamily(
        name="file_prose",
        summary=(
            "Filesystem instruction written as prose, e.g. "
            "`action: write the following content to report.md`."
        ),
        handler="_handle_file_operation",
        pattern=re.compile(
            r"write.*to\s+(?:a\s+)?(?:file|path)"
            r"|save.*to\s+(?:a\s+)?(?:file|path)"
            r"|write.*following.*content.*to"
            r"|save.*following.*content.*to"
            r"|create.*file\s+at"
            r"|export.*to\s+file"
            r"|store.*in\s+file"
            r"|write.*to\s+[^\s]+\.(?:txt|md|json|yaml|yml|csv|html)"
            r"|save.*to\s+[^\s]+\.(?:txt|md|json|yaml|yml|csv|html)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    ActionFamily(
        name="auto",
        summary=(
            "An explicit request for the model to interpret the instruction, "
            "e.g. `action: <AUTO>summarise the findings</AUTO>`."
        ),
        # No handler: this is the one case that is *meant* to reach the model.
        handler=None,
        pattern=re.compile(r"<AUTO>.*?</AUTO>", re.IGNORECASE | re.DOTALL),
    ),
)


def _build_index() -> Dict[str, ActionSpec]:
    index: Dict[str, ActionSpec] = {}
    for spec in ACTION_SPECS:
        for spelling in (spec.name, *spec.aliases):
            key = spelling.lower()
            if key in index:
                raise ValueError(
                    f"action spelling {spelling!r} is claimed by both "
                    f"{index[key].name!r} and {spec.name!r}"
                )
            index[key] = spec
    return index


_BY_SPELLING: Dict[str, ActionSpec] = _build_index()


# --- generated views -------------------------------------------------------
#
# Every one of these is derived. None is maintained by hand, so none can drift.

#: Action spelling -> the `HybridControlSystem` method that runs it. Values are
#: method *names* because the handlers are bound methods on the control system;
#: the dispatcher resolves them per call.
BUILTIN_ACTION_HANDLERS: Dict[str, str] = {
    spelling: spec.handler
    for spelling, spec in _BY_SPELLING.items()
    if spec.handler is not None
}

#: Every spelling the runtime executes without a `tool:`, aliases included.
BUILTIN_ACTIONS: FrozenSet[str] = frozenset(_BY_SPELLING)

#: The spellings of the structured-generation action.
STRUCTURED_ACTIONS: FrozenSet[str] = frozenset(
    spelling
    for spelling, spec in _BY_SPELLING.items()
    if spec.name == "generate_structured"
)

#: What the control systems advertise. Generated, so it can no longer promise
#: an action nobody implements.
SUPPORTED_ACTIONS: Tuple[str, ...] = tuple(
    sorted(spelling for spelling, spec in _BY_SPELLING.items() if spec.v1)
)


def resolve_action(action: str) -> Optional[ActionSpec]:
    """The spec for `action`, by canonical name or alias, else `None`.

    Matching is case-insensitive and ignores surrounding whitespace, because
    the executor lowercases a task's action before dispatching on it.
    """
    if not isinstance(action, str):
        return None
    return _BY_SPELLING.get(action.strip().lower())


def canonical_action(action: str) -> Optional[str]:
    """The one spelling that should appear in the task graph and the trace."""
    spec = resolve_action(action)
    return spec.name if spec else None


def is_builtin_action(action: str) -> bool:
    """Whether `action` names something the runtime executes without a tool."""
    return resolve_action(action) is not None


def match_action_family(action: str) -> Optional[ActionFamily]:
    """The prose family `action` belongs to, if any."""
    if not isinstance(action, str):
        return None
    for family in ACTION_FAMILIES:
        if family.pattern.search(action):
            return family
    return None


def is_known_action(action: str) -> bool:
    """Whether anything at all can execute `action`.

    The question both the validator and the dispatcher ask. Anything that
    answers `False` is refused rather than turned into a model prompt.
    """
    return is_builtin_action(action) or match_action_family(action) is not None
