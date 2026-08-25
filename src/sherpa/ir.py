"""Typed plan IR for sherpa (issue #492, MVP scope item 1).

The Pydantic models in this module are the semantic contract for problem
specifications and plans. YAML is an optional serialization surface (see
`sherpa.cli`), never the source of truth.

Control flow is structural: there is no ``goto``. Validation
(:func:`validate_plan`) rejects plans that exceed delegated authority budgets,
declared fan-out caps, or depth estimates, fail-closed.
"""

from __future__ import annotations

from typing import Annotated, Any, Iterator, Literal, NamedTuple

from pydantic import BaseModel, Field

TERMINAL_STATES: tuple[str, ...] = (
    "completed",
    "failed",
    "blocked",
    "escalated",
    "cancelled",
    "budget_exhausted",
)

NODE_STATES: tuple[str, ...] = (
    "pending",
    "leased",
    "running",
    "completed",
    "failed",
    "blocked",
    "escalated",
    "cancelled",
    "skipped",
    "budget_exhausted",
)


class Budgets(BaseModel):
    """Hard resource ceilings for a plan/run (issue #492 'every loop has hard ... budgets')."""

    max_nodes: int = Field(default=200, ge=1)
    max_attempts_per_node: int = Field(default=2, ge=1)
    max_depth: int = Field(default=6, ge=1)
    max_fanout: int = Field(default=4, ge=1)
    max_tokens: int = Field(default=200_000, ge=0)
    #: 0.0 means "no paid spend permitted", NOT "unlimited". The check in
    #: `kernel._check_budgets` used to skip this ceiling unless it was > 0,
    #: which made the default fail-open.
    max_cost_usd: float = Field(default=0.0, ge=0.0)
    max_wall_seconds: float = Field(default=900.0, gt=0.0)


class Authority(BaseModel):
    """Delegated powers. Child plans may only narrow a parent's grants.

    The containment rules live in :mod:`sherpa.authority`, which is the single
    implementation used by delegation, admission, and capability execution
    alike. Do not add a second matcher here: three divergent ones is what made
    the pre-#493 model unenforceable.
    """

    fs_read: tuple[str, ...] = ()
    fs_write: tuple[str, ...] = ()
    net_domains: tuple[str, ...] = ()
    subprocess_allow: tuple[str, ...] = ()

    def allows(self, child: "Authority") -> bool:
        """True when *child* requests nothing this authority does not hold."""
        from sherpa.authority import authority_covers

        return authority_covers(self, child)

    def narrower(self, other: "Authority") -> bool:
        """Readability alias: True when *self* fits inside *other*."""
        return other.allows(self)


class AcceptanceCheck(BaseModel):
    """An externally verifiable check the final output must satisfy."""

    id: str
    kind: Literal["pytest", "jsonschema", "predicate"]
    spec: dict[str, Any]


class ProblemSpec(BaseModel):
    """Identity 1: the immutable problem specification (issue #492 core model)."""

    id: str
    goal: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    acceptance: list[AcceptanceCheck] = Field(default_factory=list)
    budgets: Budgets = Field(default_factory=Budgets)
    authority: Authority = Field(default_factory=Authority)
    attended: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeBase(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str | None = None


class InvokeCapability(NodeBase):
    kind: Literal["invoke_capability"]
    capability: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    atomic_claim: bool = True


class InvokePlan(NodeBase):
    kind: Literal["invoke_plan"]
    plan_id: str
    plan_version: int | None = None
    input_map: dict[str, Any] = Field(default_factory=dict)


class Decompose(NodeBase):
    kind: Literal["decompose"]
    subgoal: str
    hints: dict[str, Any] = Field(default_factory=dict)
    budget_fraction: float = Field(default=0.5, gt=0.0, le=1.0)
    fanout_cap: int | None = Field(default=None, ge=1)


class BranchCase(BaseModel):
    when: str | None = None
    body: list["Node"] = Field(min_length=1)


class Branch(NodeBase):
    kind: Literal["branch"]
    cases: list[BranchCase] = Field(min_length=1)


class While(NodeBase):
    kind: Literal["while"]
    guard: str
    max_iterations: int = Field(ge=1)
    body: list["Node"] = Field(min_length=1)


class Parallel(NodeBase):
    kind: Literal["parallel"]
    branches: list[list["Node"]] = Field(min_length=1)


class AskUser(NodeBase):
    kind: Literal["ask_user"]
    question: str
    options: list[str] = Field(default_factory=list)


class Return(NodeBase):
    kind: Literal["return"]
    outputs: dict[str, Any] = Field(default_factory=dict)


class Fail(NodeBase):
    kind: Literal["fail"]
    reason: str


Node = Annotated[
    InvokeCapability
    | InvokePlan
    | Decompose
    | Branch
    | While
    | Parallel
    | AskUser
    | Return
    | Fail,
    Field(discriminator="kind"),
]

BranchCase.model_rebuild()
Branch.model_rebuild()
While.model_rebuild()
Parallel.model_rebuild()


class Plan(BaseModel):
    """Identity 2: a typed control-flow graph, versioned append-only."""

    id: str
    version: int = 1
    problem_id: str | None = None
    authority: Authority = Field(default_factory=Authority)
    budgets: Budgets = Field(default_factory=Budgets)
    root: list[Node] = Field(min_length=1)
    notes: dict[str, Any] = Field(default_factory=dict)


class PlanError(NamedTuple):
    path: str
    code: str
    message: str


def iter_nodes(nodes: list[Node]) -> Iterator[Any]:
    """Depth-first iteration over every node in *nodes*."""
    stack = list(reversed(nodes))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, Branch):
            stack.extend(case_node for case in reversed(node.cases) for case_node in reversed(case.body))
        elif isinstance(node, (While,)):
            stack.extend(reversed(node.body))
        elif isinstance(node, Parallel):
            for branch in reversed(node.branches):
                stack.extend(reversed(branch))


def estimate_depth(plan: Plan) -> int:
    """Longest root-to-leaf nesting depth of structured nodes."""
    def depth_of(nodes: list[Any]) -> int:
        best = 0
        for node in nodes:
            if isinstance(node, Branch):
                inner = max((depth_of(c.body) for c in node.cases), default=0)
            elif isinstance(node, While):
                inner = depth_of(node.body)
            elif isinstance(node, Parallel):
                inner = max((depth_of(b) for b in node.branches), default=0)
            else:
                inner = 0
            best = max(best, inner + 1)
        return best

    return depth_of(list(plan.root))


def validate_plan(plan: Plan, *, registry_names: set[str] | None = None) -> list[PlanError]:
    """Return every structural violation; empty list means the plan is valid."""
    errors: list[PlanError] = []
    seen: set[str] = set()

    def walk(nodes: list[Any], path: str) -> int:
        deepest = 0
        for i, node in enumerate(nodes):
            npath = f"{path}[{i}]{node.id}"
            if node.id in seen:
                errors.append(PlanError(npath, "duplicate_id", f"duplicate node id {node.id!r}"))
            seen.add(node.id)
            deepest = max(deepest, 1)
            if isinstance(node, Branch):
                if node.cases[-1].when is not None and len(node.cases) > 1:
                    # allowed: all-when cases are fine; only flag an else-case not last
                    if any(c.when is None for c in node.cases[:-1]):
                        errors.append(
                            PlanError(npath, "else_not_last", "branch case without `when` must be last")
                        )
                for j, case in enumerate(node.cases):
                    if not case.body:
                        errors.append(PlanError(f"{npath}.cases[{j}]", "empty_body", "empty branch body"))
                    deepest = max(deepest, 1 + walk(case.body, f"{npath}.cases[{j}]"))
            elif isinstance(node, While):
                if not node.guard:
                    errors.append(PlanError(npath, "missing_guard", "while requires a guard expression"))
                deepest = max(deepest, 1 + walk(node.body, f"{npath}.body"))
            elif isinstance(node, Parallel):
                if len(node.branches) > plan.budgets.max_fanout:
                    errors.append(
                        PlanError(
                            npath,
                            "fanout_exceeded",
                            f"parallel fan-out {len(node.branches)} > cap {plan.budgets.max_fanout}",
                        )
                    )
                for j, branch in enumerate(node.branches):
                    if not branch:
                        errors.append(PlanError(f"{npath}.branches[{j}]", "empty_body", "empty parallel branch"))
                    deepest = max(deepest, 1 + walk(branch, f"{npath}.branches[{j}]"))
            elif isinstance(node, Decompose):
                if node.fanout_cap is not None and node.fanout_cap > plan.budgets.max_fanout:
                    errors.append(
                        PlanError(npath, "fanout_exceeded", "decompose fanout_cap exceeds plan budget")
                    )
            elif isinstance(node, InvokeCapability) and registry_names is not None:
                if node.capability not in registry_names:
                    errors.append(
                        PlanError(npath, "unknown_capability", f"capability {node.capability!r} not registered")
                    )
        return deepest

    depth = walk(list(plan.root), "root")
    if depth > plan.budgets.max_depth:
        errors.append(
            PlanError("root", "depth_exceeded", f"estimated depth {depth} > max_depth {plan.budgets.max_depth}")
        )
    return errors
