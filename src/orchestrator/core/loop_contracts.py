"""What each loop construct binds, and where those bindings reach.

Validation treated "inside a loop" as one boolean and one union of names. Two
things follow from that, both wrong.

**The source expression was validated in loop scope.** `_validate_object_templates`
computed `is_loop` from the step dictionary and passed it down to every child,
including the iterable itself::

    - id: process
      for_each: "{{ item.children }}"   # accepted -- but no item exists yet
      parameters:
        text: "{{ item.name }}"         # correct

The iterable must resolve before there is an item to bind. A reference to a
loop variable in it can never work, and validation said nothing.

**Every construct got every name.** `while` does not bind `item`; only
`create_parallel_queue` binds `queue`. Accepting the union meant
``{{ is_last }}`` inside a `while` loop passed validation and then failed to
render -- the false negative #470 introduced while removing a false positive.

The tables below were read from the runtime rather than from
`docs/loop_variables.md`, which claims names the runtime does not bind in
every construct:

`for_each` / `foreach`
    `ControlSystem._render_task_templates` registers `item`, `index`,
    `is_first` and `is_last` from `metadata["loop_context"]`;
    `ContextManager.loop_context` supplies `item`, `index` and `loop_id`.
`while`
    `WhileLoopHandler` builds `iteration`, `index`, `is_first`, `position`,
    `loop_state` and `loop_id`.
`create_parallel_queue`
    `ParallelQueueTask.get_template_variables` adds `queue`, `queue_size`,
    `parallel_queue_id` and `parent_task` alongside the per-item names.
`action_loop`
    `ActionLoopContext` exposes iteration state and previous-result metadata,
    but no item: an action loop is not iterating a collection.

Each construct's `source_fields` are evaluated before the loop exists, so they
are validated in the enclosing scope. Everything else in the step is body, and
sees that construct's bindings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Tuple

#: Jinja's own loop object, bound by `{% for %}` inside a template rather than
#: by a pipeline construct. Jinja's undeclared-name analysis already accounts
#: for it; it is listed so that a step-level loop does not report it either.
JINJA_LOOP: FrozenSet[str] = frozenset({"loop"})


@dataclass(frozen=True)
class LoopContract:
    """One loop construct: what it binds, and which fields see the bindings."""

    key: str
    #: Fields evaluated before the loop exists. Validated in the outer scope.
    source_fields: Tuple[str, ...]
    #: Names available to the body, in the bare spelling. The runtime also
    #: registers a `$`-prefixed alias for each.
    bindings: FrozenSet[str]

    def dollar_bindings(self) -> FrozenSet[str]:
        return frozenset(f"${name}" for name in self.bindings)

    def all_bindings(self) -> FrozenSet[str]:
        return self.bindings | self.dollar_bindings() | JINJA_LOOP


FOR_EACH = LoopContract(
    key="for_each",
    source_fields=("for_each", "foreach"),
    bindings=frozenset({"item", "index", "is_first", "is_last"}),
)

#: `foreach` is the same construct under the spelling the declarative engine
#: accepts. One contract, two keys -- the compiler treats them identically, so
#: validation must too.
FOREACH = LoopContract(
    key="foreach",
    source_fields=FOR_EACH.source_fields,
    bindings=FOR_EACH.bindings,
)

WHILE = LoopContract(
    key="while",
    source_fields=("while", "until"),
    bindings=frozenset({
        "iteration", "index", "is_first", "position", "loop_state", "loop_id",
    }),
)

CREATE_PARALLEL_QUEUE = LoopContract(
    key="create_parallel_queue",
    source_fields=("create_parallel_queue",),
    bindings=frozenset({
        "item", "index", "queue", "queue_size", "is_first", "is_last",
        "parallel_queue_id", "parent_task",
    }),
)

ACTION_LOOP = LoopContract(
    key="action_loop",
    source_fields=("action_loop", "until"),
    bindings=frozenset({
        "loop_id", "iteration", "is_first", "has_previous", "total_duration",
        "termination_reason",
    }),
)

LOOP_CONTRACTS: Dict[str, LoopContract] = {
    contract.key: contract
    for contract in (FOR_EACH, FOREACH, WHILE, CREATE_PARALLEL_QUEUE, ACTION_LOOP)
}

#: Every name any construct binds. Used where a construct is not known -- never
#: as a substitute for the per-construct set, which is what made
#: `{{ is_last }}` acceptable inside a `while` loop.
ALL_BINDINGS: FrozenSet[str] = frozenset().union(
    *(contract.all_bindings() for contract in LOOP_CONTRACTS.values())
)


def contract_for(step: Any) -> Optional[LoopContract]:
    """The loop contract a step declares, or None if it is not a loop.

    Checked in declaration order of `LOOP_CONTRACTS` so a step carrying more
    than one loop key resolves deterministically.
    """
    if not isinstance(step, dict):
        return None
    for key, contract in LOOP_CONTRACTS.items():
        if key in step:
            return contract
    return None


def is_source_field(contract: Optional[LoopContract], field: str) -> bool:
    """Whether `field` is evaluated before the loop's bindings exist."""
    return contract is not None and field in contract.source_fields
