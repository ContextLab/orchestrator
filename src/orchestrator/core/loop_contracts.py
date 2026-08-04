"""What each loop construct binds, and where those bindings reach.

Every table here was read off a *running pipeline*, not off the source and not
off `docs/loop_variables.md`. `tests/test_loop_runtime_parity.py` re-derives it
the same way on every run: each declared binding must render in a real
execution, and a name this module withholds must fail to render. A table that
cannot be reproduced by running the thing is a claim, not a contract -- which
is how the previous version of this file came to declare `foreach` a supported
alias of `for_each` when no engine expands it (#475).

Two axes, both of which the first version got wrong.

**Scope is per field, not per step.** `_validate_object_templates` computed one
boolean for the whole step, so the iterable was validated in the scope it
introduces::

    - id: process
      for_each: "{{ item.children }}"   # no item exists yet
      parameters:
        text: "{{ item.name }}"         # correct

#473 fixed that with a flat `source_fields` tuple, which then made the opposite
error: it declared `action_loop` and `until` sources, so the *body* of an
action loop -- the one place iteration state certainly exists -- was validated
outside the loop. `scopes` below is keyed by field path for that reason:
`create_parallel_queue.on` is evaluated before there is a queue while the rest
of that same object is per-item.

**Bindings are per construct.** `while` binds no `item`; only
`create_parallel_queue` binds `queue`.

Where the runtime leaks a name that has no meaning for the construct -- a
`for_each:` iterable can read `{{ index }}` and get `0` from a context that is
not its own -- this file withholds it and #477 tracks the leak. Rendering a
stale zero is not a binding; blessing it here would turn a runtime bug into a
documented feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

#: Jinja's own loop object, bound by `{% for %}` *inside* a template rather
#: than by a pipeline construct. It belongs to no contract -- putting it in one
#: would make that construct's empty scopes non-empty, and a scope that is
#: never empty reads downstream as "some loop is in scope". It is in the union
#: below because the data-flow validator consults that to know `{{ loop.index }}`
#: names Jinja's counter and not a missing task.
JINJA_LOOP: FrozenSet[str] = frozenset({"loop"})

#: `{{ $name }}` is not Jinja -- `$` is a syntax error. It works only because
#: `UnifiedTemplateResolver._preprocess_dollar_variables` rewrites it first,
#: and that rewrite does not reach every render path: `{{ $item }}` resolves
#: while `{{ $position }}` raises `unexpected char '$'`. So the `$` spelling is
#: an observed list, not `"$" + every binding` (#474).
PRESTRIPPED_DOLLAR_NAMES: FrozenSet[str] = frozenset({
    "$item", "$index", "$is_first", "$is_last",
    "$iteration", "$loop_id", "$loop_name", "$loop_state",
})


@dataclass(frozen=True)
class LoopContract:
    """One loop construct: what it binds, and which fields see the bindings."""

    key: str
    #: Bare names the body can render, confirmed by execution.
    bindings: FrozenSet[str]
    #: Field paths relative to the loop step whose scope is *not* the body's.
    #: Matched by longest prefix, so an entry for `create_parallel_queue.on`
    #: narrows that one field while its siblings keep the body scope.
    scopes: Mapping[str, FrozenSet[str]] = field(default_factory=dict)
    #: Names available only in the `$` spelling. `$loop_name` renders inside a
    #: `for_each` body where bare `loop_name` does not; the asymmetry is real
    #: and is part of #474.
    dollar_only: FrozenSet[str] = frozenset()

    def _spell(self, bare: FrozenSet[str]) -> FrozenSet[str]:
        """Both spellings of `bare`. An empty scope stays empty: a set that is
        never empty reads as "some loop is in scope" downstream, which turns
        every out-of-scope reference into the wrong diagnostic."""
        if not bare:
            return frozenset()
        dollar = frozenset(f"${name}" for name in bare | self.dollar_only)
        return bare | (dollar & PRESTRIPPED_DOLLAR_NAMES)

    def all_bindings(self) -> FrozenSet[str]:
        """Everything the body may name, in both spellings."""
        return self._spell(self.bindings)

    def bindings_for(self, path: str) -> FrozenSet[str]:
        """What `path` -- relative to the loop step -- may name.

        Unlisted paths are body. Longest prefix wins so that a narrowed field
        does not narrow the object around it.
        """
        best: Optional[str] = None
        for candidate in self.scopes:
            if path == candidate or path.startswith(f"{candidate}."):
                if best is None or len(candidate) > len(best):
                    best = candidate
        if best is None:
            return self.all_bindings()
        return self._spell(self.scopes[best])


#: `for_each` body, observed: `item`, `index`, `is_first`, `is_last`,
#: `position`, `length`, `remaining`, `has_next`, `has_prev`, `loop_id` all
#: render real values, and `$loop_name` renders where bare `loop_name` errors.
#: `iteration` renders `None` -- a while-loop name leaking into a construct
#: that has no iteration count (#477) -- so it is not a binding.
FOR_EACH = LoopContract(
    key="for_each",
    bindings=frozenset({
        "item", "index", "is_first", "is_last", "position",
        "length", "remaining", "has_next", "has_prev", "loop_id",
    }),
    dollar_only=frozenset({"loop_name"}),
    #: The iterable resolves before there is an item to bind. `{{ item }}`
    #: there is an error at run time; the other names render only because a
    #: zeroed context is in scope, which is the leak in #477.
    scopes={"for_each": frozenset()},
)

#: `while` body, observed: `iteration`, `index`, `is_first`, `position`,
#: `loop_id`, `loop_name`, `loop_state`. `item` renders `None` and `length`,
#: `remaining`, `has_next`, `has_prev` render collection values for a construct
#: with no collection (#477), so none of them are bindings.
WHILE = LoopContract(
    key="while",
    bindings=frozenset({
        "iteration", "index", "is_first", "position",
        "loop_id", "loop_name", "loop_state",
    }),
    #: Unlike a `for_each` iterable, a `while` condition is re-evaluated every
    #: iteration, against the context `WhileLoopHandler.should_continue`
    #: assembles at `control_flow/loops.py:428` -- which holds exactly
    #: `iteration` and `loop_state`. `until` is evaluated from that same
    #: context a few lines later, so both conditions share the scope.
    scopes={
        "while": frozenset({"iteration", "loop_state"}),
        "until": frozenset({"iteration", "loop_state"}),
    },
)

#: `create_parallel_queue` body, observed: `item`, `index`, `is_first`,
#: `is_last`, `queue`, `queue_size`, `parallel_queue_id`, `parent_task`.
CREATE_PARALLEL_QUEUE = LoopContract(
    key="create_parallel_queue",
    bindings=frozenset({
        "item", "index", "is_first", "is_last",
        "queue", "queue_size", "parallel_queue_id", "parent_task",
    }),
    #: `on` generates the queue, so nothing per-item exists while it resolves.
    #: The whole-object `source_fields` entry #473 used put the nested action
    #: list outside the loop as well, which is the opposite error.
    scopes={"create_parallel_queue.on": frozenset()},
)

#: `action_loop` body, observed: `iteration`, `is_first`, `loop_id`,
#: `has_previous`, `total_duration`, `termination_reason`. An action loop
#: repeats actions rather than walking a collection, so `item`, `index` and
#: `position` are not bound -- they pass through unrendered.
#:
#: The `action_loop` key *is* the body: `_build_iteration_context`
#: (`control_flow/action_loop_handler.py:448`) builds iteration state before
#: the action list executes. Its `until` shares that context at line 404 --
#: though nothing evaluates it today, which is #476.
ACTION_LOOP = LoopContract(
    key="action_loop",
    bindings=frozenset({
        "iteration", "is_first", "loop_id",
        "has_previous", "total_duration", "termination_reason",
    }),
)

LOOP_CONTRACTS: Dict[str, LoopContract] = {
    contract.key: contract
    for contract in (FOR_EACH, WHILE, CREATE_PARALLEL_QUEUE, ACTION_LOOP)
}

#: Every name any construct binds. Used where a construct is not known -- never
#: as a substitute for the per-construct set, which is what made
#: `{{ is_last }}` acceptable inside a `while` loop.
ALL_BINDINGS: FrozenSet[str] = JINJA_LOOP.union(
    *(contract.all_bindings() for contract in LOOP_CONTRACTS.values())
)


def contracts_for(step: Any) -> Tuple[LoopContract, ...]:
    """Every loop contract a step declares.

    More than one means the step is ambiguous. Returning them all instead of
    silently taking the first lets the caller say so: which construct wins is
    otherwise decided by this module's declaration order, which no engine
    agrees to.
    """
    if not isinstance(step, dict):
        return ()
    return tuple(
        contract for key, contract in LOOP_CONTRACTS.items() if key in step
    )
