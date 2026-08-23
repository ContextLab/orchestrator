"""Typed plan IR, its validator, and the budget ledger.

#485 asks for "loops, conditionals, and goto ... a complete (albeit simple)
language". This IR is deliberately structured: sequence + branch + bounded loop
+ call is already Turing-complete, and every one of those forms can be
statically bounded, resumed, diffed and drawn. Unstructured `goto` buys nothing
and costs all four. The repo's five open control-flow bugs (#474-#478) are the
empirical argument: they are all in a hand-written YAML dialect with a
hand-written interpreter.

The validator is where the design gets its teeth. It rejects a plan that:
  - loops without a static iteration bound,
  - calls a capability that is not registered, or not promoted,
  - requests authority its parent does not hold (authority narrows downward,
    never widens),
  - exceeds the fan-out cap (#485's "on the order of 10 or fewer" steps),
  - declares no output contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Literal

Kind = Literal[
    "capability", "plan", "decompose", "branch", "loop", "parallel", "return", "fail"
]

MAX_STEPS_PER_PLAN = 12
MAX_LOOP_ITERATIONS = 64


class PlanInvalid(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class Authority:
    """What a node is allowed to do. Inherited downward, never broadened."""

    net: frozenset[str] = frozenset()
    fs_write: frozenset[str] = frozenset()
    subprocess: bool = False
    spend_usd: float = 0.0

    def covers(self, other: "Authority") -> bool:
        return (
            other.net <= self.net
            and other.fs_write <= self.fs_write
            and (self.subprocess or not other.subprocess)
            and other.spend_usd <= self.spend_usd + 1e-9
        )

    def narrow(self, other: "Authority") -> "Authority":
        return Authority(
            net=self.net & other.net,
            fs_write=self.fs_write & other.fs_write,
            subprocess=self.subprocess and other.subprocess,
            spend_usd=min(self.spend_usd, other.spend_usd),
        )


@dataclass(frozen=True)
class Step:
    id: str
    kind: Kind
    # capability / plan
    ref: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    # decompose
    problem: str | None = None
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    # branch / loop / parallel
    guard: str | None = None
    then: tuple["Step", ...] = ()
    otherwise: tuple["Step", ...] = ()
    branches: tuple[tuple["Step", ...], ...] = ()
    max_iterations: int | None = None
    # everything
    output_schema: str = "any"
    authority: Authority = Authority()

    def children(self) -> list["Step"]:
        out = list(self.then) + list(self.otherwise)
        for b in self.branches:
            out += list(b)
        return out


@dataclass(frozen=True)
class Plan:
    id: str
    problem: str
    steps: tuple[Step, ...]
    output_schema: str = "any"
    signature: str = "any->any"

    def hash(self) -> str:
        from .store import content_hash

        return content_hash(json.dumps(as_jsonable(self), sort_keys=True))

    def walk(self) -> list[Step]:
        out: list[Step] = []
        stack = list(self.steps)
        while stack:
            s = stack.pop(0)
            out.append(s)
            stack = s.children() + stack
        return out


def as_jsonable(obj: Any) -> Any:
    if isinstance(obj, (Plan, Step, Authority)):
        d = {}
        for k, v in obj.__dict__.items():
            d[k] = as_jsonable(v)
        return d
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(as_jsonable(v) for v in obj)
    if isinstance(obj, dict):
        return {k: as_jsonable(v) for k, v in obj.items()}
    return obj


def validate(
    plan: Plan,
    known_capabilities: dict[str, str],
    parent_authority: Authority,
    max_steps: int = MAX_STEPS_PER_PLAN,
) -> list[str]:
    """Return a list of errors. Empty list == valid."""
    errors: list[str] = []
    seen_ids: set[str] = set()

    if not plan.steps:
        errors.append("plan has no steps")
    if len(plan.steps) > max_steps:
        errors.append(
            f"plan has {len(plan.steps)} top-level steps (cap {max_steps})"
        )

    def check(step: Step, depth: int) -> None:
        if step.id in seen_ids:
            errors.append(f"duplicate step id {step.id!r}")
        seen_ids.add(step.id)
        if not step.output_schema:
            errors.append(f"{step.id}: no output contract")
        if not parent_authority.covers(step.authority):
            errors.append(
                f"{step.id}: requests authority beyond its parent "
                f"(net={sorted(step.authority.net)}, subprocess={step.authority.subprocess})"
            )
        if step.kind == "capability":
            if not step.ref:
                errors.append(f"{step.id}: capability step with no ref")
            elif step.ref not in known_capabilities:
                errors.append(f"{step.id}: unknown capability {step.ref!r}")
            elif known_capabilities[step.ref] not in ("trusted", "candidate"):
                errors.append(
                    f"{step.id}: capability {step.ref!r} is "
                    f"{known_capabilities[step.ref]}, not promoted"
                )
        elif step.kind == "decompose":
            if not step.problem:
                errors.append(f"{step.id}: decompose step with no problem statement")
            if not step.outputs:
                errors.append(f"{step.id}: decompose step declares no outputs")
        elif step.kind == "loop":
            if step.max_iterations is None:
                errors.append(f"{step.id}: loop has no static iteration bound")
            elif not (1 <= step.max_iterations <= MAX_LOOP_ITERATIONS):
                errors.append(
                    f"{step.id}: loop bound {step.max_iterations} outside "
                    f"1..{MAX_LOOP_ITERATIONS}"
                )
            if not step.guard:
                errors.append(f"{step.id}: loop has no guard")
            if not step.then:
                errors.append(f"{step.id}: loop has an empty body")
        elif step.kind == "branch":
            if not step.guard:
                errors.append(f"{step.id}: branch has no guard")
            if not step.then and not step.otherwise:
                errors.append(f"{step.id}: branch has no arms")
        elif step.kind == "parallel":
            if len(step.branches) < 2:
                errors.append(f"{step.id}: parallel with fewer than 2 branches")
        elif step.kind == "plan":
            if not step.ref:
                errors.append(f"{step.id}: plan call with no ref")
        for c in step.children():
            check(c, depth + 1)

    for s in plan.steps:
        check(s, 0)
    if not any(s.kind in ("capability", "decompose", "plan") for s in plan.walk()):
        errors.append("plan does no work: no capability, plan or decompose step")
    return errors


# ------------------------------------------------------------------ budgets


class BudgetExhausted(Exception):
    pass


@dataclass
class Budget:
    """A credit line, not a depth cap.

    A depth cap silently truncates and returns a plausible-looking answer. A
    budget that is *spent* forces the node to say so, and hands the decision to
    re-allocate, re-plan or escalate to somebody with a wider view.
    """

    tokens: int
    usd: float
    nodes: int
    seconds: float
    spent_tokens: int = 0
    spent_usd: float = 0.0
    spent_nodes: int = 0
    spent_seconds: float = 0.0
    reserve_frac: float = 0.30

    def remaining(self) -> "Budget":
        return Budget(
            max(0, self.tokens - self.spent_tokens),
            max(0.0, self.usd - self.spent_usd),
            max(0, self.nodes - self.spent_nodes),
            max(0.0, self.seconds - self.spent_seconds),
            reserve_frac=self.reserve_frac,
        )

    def exhausted(self) -> bool:
        r = self.remaining()
        return r.tokens <= 0 or r.usd <= 0 or r.nodes <= 0 or r.seconds <= 0

    def spend(self, tokens: int = 0, usd: float = 0.0, nodes: int = 0,
              seconds: float = 0.0) -> None:
        self.spent_tokens += tokens
        self.spent_usd += usd
        self.spent_nodes += nodes
        self.spent_seconds += seconds

    def child_share(self, n_children: int) -> "Budget":
        """Split, holding a reserve back for reallocation and escalation."""
        n = max(1, n_children)
        r = self.remaining()
        keep = 1.0 - self.reserve_frac
        return Budget(
            int(r.tokens * keep / n),
            r.usd * keep / n,
            max(1, int(r.nodes * keep / n)),
            r.seconds * keep / n,
            reserve_frac=self.reserve_frac,
        )

    def absorb(self, child: "Budget") -> None:
        self.spend(child.spent_tokens, child.spent_usd, child.spent_nodes,
                   child.spent_seconds)

    def top_up(self, other: "Budget", tokens: int) -> int:
        """Move credit from a parent's reserve into this budget. Recorded."""
        avail = other.remaining().tokens
        moved = min(tokens, avail)
        other.spend(tokens=moved)
        self.tokens += moved
        return moved

    def as_dict(self) -> dict[str, Any]:
        r = self.remaining()
        return {
            "tokens": self.tokens, "spent_tokens": self.spent_tokens,
            "remaining_tokens": r.tokens, "nodes": self.nodes,
            "spent_nodes": self.spent_nodes, "usd": round(self.usd, 4),
            "spent_usd": round(self.spent_usd, 4),
        }
