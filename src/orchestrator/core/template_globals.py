"""The pipeline language's global functions, and how they may be called.

Filters transform a value the author already holds. Globals do not: each of
these answers a question about the state of a *run* -- what time it started,
whether a file exists yet, which loop iteration this is. That is why only the
runtime holds their implementations, and why the compiler and the validators
must know them by name without being able to call them. `template_sandbox`
explains the mechanics; this module says what the names *are*.

#450 taught both validators these names, which stopped `{{ now() }}` -- a
pipeline that runs correctly -- from being reported as an undefined variable.
It stopped there, at the name, and so accepted every way of naming a global
that is not actually a call::

    {{ nowx() }}            rejected   (not a global)
    {{ now.foo }}           accepted   -> fails at run time
    {{ file_exists.bad }}   accepted   -> fails at run time
    {{ now(1, 2, 3) }}      accepted   -> fails at run time
    {{ file_exists() }}     accepted   -> fails at run time
    {{ now }}               accepted   -> *runs*, and writes
                                       "<function ...now at 0x1084...>"

The last one is the worst, because nothing fails: the repr of a live function
object is written into the artifact as though it were data. (It is not a
sandbox escape -- `now.__globals__` and `now.__class__` are both refused by
`SandboxedEnvironment`, which #447 put in place. What leaks is the repr, not
the object graph.)

So the contract has to cover the call, not just the name, and it is declared
here rather than inferred from the callables. Inference was the right move in
#450, when the only question was which names exist and drift was the risk. It
cannot express what this needs -- an argument contract, a summary, later a
deprecation -- and it makes the public language a shadow of a private
implementation detail. `test_template_globals.py` asserts every spec matches
the callable the runtime actually registers, which keeps the drift protection
that derivation gave for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, List, Optional, Tuple


@dataclass(frozen=True)
class GlobalSpec:
    """One global function: its name, how many arguments it takes, what it does."""

    name: str
    min_args: int
    max_args: Optional[int]  # None means unbounded
    summary: str
    #: What to write instead, if this global should no longer be used. The
    #: call keeps working -- pipelines in the wild use it -- but validation
    #: says so, once, with the replacement named.
    deprecated_for: Optional[str] = None

    def accepts(self, positional: int) -> bool:
        if positional < self.min_args:
            return False
        return self.max_args is None or positional <= self.max_args

    @property
    def arity(self) -> str:
        """How the argument count reads in an error message or a doc table."""
        if self.max_args is None:
            return f"{self.min_args} or more"
        if self.min_args == self.max_args:
            return str(self.min_args)
        return f"{self.min_args} to {self.max_args}"


#: Every global the pipeline language offers. The runtime registers exactly
#: these; the compiler and validators recognise exactly these.
GLOBAL_SPECS: Tuple[GlobalSpec, ...] = (
    GlobalSpec(
        "now", 0, 0,
        "The current time, read afresh at every use, so two steps of one run "
        "disagree. Deprecated: use `execution.timestamp`, which is the same "
        "for every step of a run.",
        deprecated_for="execution.timestamp",
    ),
    GlobalSpec(
        "file_exists", 1, 2,
        "Whether a path exists, answered when the step runs, so a file an "
        "earlier step wrote counts.",
    ),
    GlobalSpec(
        "include_file", 1, 2,
        "The contents of a file, read when the step runs.",
    ),
    GlobalSpec(
        "loop_var", 2, 2,
        "A named loop's variable by name: `loop_var('outer', 'item')`.",
    ),
    GlobalSpec(
        "loop_item_at", 2, 2,
        "An item at a fixed index of a named loop: `loop_item_at('outer', 0)`.",
    ),
    GlobalSpec("current_loop_name", 0, 0, "The innermost active loop's name."),
    GlobalSpec("active_loops", 0, 0, "The names of the loops currently running."),
    GlobalSpec(
        "historical_loops", 0, 0,
        "The names of finished loops whose values are still reachable.",
    ),
)

GLOBAL_NAMES: FrozenSet[str] = frozenset(spec.name for spec in GLOBAL_SPECS)

_BY_NAME = {spec.name: spec for spec in GLOBAL_SPECS}


def global_spec(name: str) -> Optional[GlobalSpec]:
    """The spec for `name`, or None if it is not a pipeline global."""
    return _BY_NAME.get(name)


#: Stable identifiers for the two ways a global can be misused. Callers match
#: on these rather than on message text.
NOT_CALLED = "global_not_called"
WRONG_ARITY = "global_wrong_arity"
DEPRECATED = "global_deprecated"


@dataclass(frozen=True)
class GlobalMisuse:
    """A global named in a template in a way that cannot work at run time."""

    name: str
    code: str
    message: str
    suggestion: str
    #: "error" refuses the pipeline; "warning" lets it run and says so.
    severity: str = "error"


def find_global_misuse(ast: Any) -> List[GlobalMisuse]:
    """Every misuse of a pipeline global in a parsed template.

    Works on the parsed AST rather than on the text because the text does not
    distinguish the cases: `now` appears identically in `{{ now() }}`,
    `{{ now.foo }}` and `{{ now }}`, and only the first is a call.
    """
    from jinja2 import nodes

    from .template_scope import shadowed_name_nodes

    # A Name node is a legitimate use only when it is the thing being called.
    # Identity matters here, not the name: `{{ now() and now.foo }}` has two
    # Name nodes spelled the same, one valid and one not.
    calls_by_callee = {
        id(call.node): call
        for call in ast.find_all(nodes.Call)
        if isinstance(call.node, nodes.Name) and call.node.name in GLOBAL_NAMES
    }

    # `{% for now in items %}{{ now }}{% endfor %}` rebinds the name: the
    # target is a `store`, but the use inside the body is an ordinary `load`
    # and looks exactly like ours. Which uses the binding actually reaches is
    # a question of scope -- a template-wide set of bound names would silence
    # `{{ now }}` before and after that loop as well, where it really is our
    # global and really is a misuse.
    shadowed = shadowed_name_nodes(ast)

    misuse: List[GlobalMisuse] = []
    seen = set()
    for name_node in ast.find_all(nodes.Name):
        spec = global_spec(name_node.name)
        if spec is None or id(name_node) in shadowed:
            continue
        if getattr(name_node, "ctx", "load") != "load":
            continue

        call = calls_by_callee.get(id(name_node))
        if call is None:
            key = (spec.name, NOT_CALLED)
            if key in seen:
                continue
            seen.add(key)
            call_form = f"{spec.name}()" if spec.min_args == 0 else f"{spec.name}(...)"
            misuse.append(GlobalMisuse(
                name=spec.name,
                code=NOT_CALLED,
                message=(
                    f"'{spec.name}' is a function and must be called: write "
                    f"'{call_form}'. Naming it without calling it yields the "
                    f"function itself, which renders as '<function ...>'."
                ),
                suggestion=call_form,
            ))
            continue

        # `f(*args)` cannot be counted before it runs, so it is not checked.
        if call.dyn_args is not None:
            continue

        positional = len(call.args)
        if not spec.accepts(positional):
            key = (spec.name, WRONG_ARITY, positional)
            if key in seen:
                continue
            seen.add(key)
            misuse.append(GlobalMisuse(
                name=spec.name,
                code=WRONG_ARITY,
                message=(
                    f"'{spec.name}' takes {spec.arity} argument(s), not "
                    f"{positional}"
                ),
                suggestion=f"{spec.name} expects {spec.arity} argument(s)",
            ))
            continue

        # A correct call to something that should no longer be written. Not an
        # error: pipelines in the wild use it and must keep running.
        if spec.deprecated_for is not None:
            key = (spec.name, DEPRECATED)
            if key in seen:
                continue
            seen.add(key)
            misuse.append(GlobalMisuse(
                name=spec.name,
                code=DEPRECATED,
                message=(
                    f"'{spec.name}()' is deprecated: it is read afresh at every "
                    f"use, so two steps of one run disagree. Use "
                    f"'{spec.deprecated_for}', which is the same for every step."
                ),
                suggestion=spec.deprecated_for,
                severity="warning",
            ))

    return misuse
