"""One graph of what must run before what.

`{{ make.path }}` in a step's parameters is a data dependency. It says so
plainly: the value cannot exist until `make` has run. Three separate parts of
the compiler agreed about that and none of them told the scheduler:

* `DependencyValidator` inferred edges from `for_each`, `condition` and
  `while` -- but never looked inside `parameters`, so the commonest reference
  of all was invisible to the cycle check.
* `DataFlowValidator` built a `data_flow_graph` that *did* include parameter
  references, then logged it and threw it away.
* `YAMLCompiler._analyze_template` found the same references with a third
  regex and stored them on `Task.template_metadata`, where nothing scheduled
  anything from them.

Meanwhile `Task.dependencies` was read from the explicit `dependencies:` key
alone. So this validated and then failed at run time (#465)::

    - id: make
      parameters: {path: "./a.txt", content: "A"}
    - id: use
      parameters: {content: "{{ make.path }}"}     # no `dependencies:`

`use` was scheduled beside `make`, `make`'s result was not in context, and the
render failed. Validation had promised the pipeline was fine.

This module is the single answer. Explicit dependencies, template-derived
dependencies and control-flow dependencies go into one graph; that graph is
what gets validated for cycles, and the same graph is what `Task.dependencies`
is built from. A fourth mechanism would have recreated the original problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Set, Tuple

from .template_scope import template_references

#: Where an edge came from. Kept on the edge so a diagnostic can say *why*
#: two steps are ordered -- "you wrote it" reads very differently from "your
#: template implies it".
DECLARED = "declared"
TEMPLATE = "template"
CONTROL_FLOW = "control_flow"

#: Step keys whose value is a condition or iterable rather than a parameter.
#: A reference in any of them orders the step just as a parameter does.
CONTROL_FLOW_KEYS: Tuple[str, ...] = (
    "for_each", "foreach", "condition", "if", "while", "until",
)


@dataclass(frozen=True)
class DependencyEdge:
    """`task` cannot start until `depends_on` has finished."""

    task: str
    depends_on: str
    origin: str
    location: str

    @property
    def is_inferred(self) -> bool:
        return self.origin != DECLARED


@dataclass(frozen=True)
class DependencyGraph:
    """Every ordering constraint in a pipeline, and where each came from."""

    steps: Tuple[str, ...]
    edges: Tuple[DependencyEdge, ...]
    #: References a step makes to itself. Never edges -- a self-edge cannot be
    #: scheduled -- but recorded so validation can report them.
    self_references: Tuple[DependencyEdge, ...] = ()

    def dependencies_for(self, task_id: str) -> List[str]:
        """What must run before `task_id`, deduplicated, in declaration order.

        Declaration order rather than discovery order so that a pipeline
        compiles to the same task graph however its templates are written.
        """
        needed = {edge.depends_on for edge in self.edges if edge.task == task_id}
        return [step for step in self.steps if step in needed]

    def adjacency(self) -> Dict[str, List[str]]:
        return {step: self.dependencies_for(step) for step in self.steps}

    def origins_for(self, task_id: str, depends_on: str) -> Set[str]:
        return {
            edge.origin for edge in self.edges
            if edge.task == task_id and edge.depends_on == depends_on
        }

    def inferred_only(self) -> List[DependencyEdge]:
        """Edges no `dependencies:` entry also states.

        The basis for the `implicit_dependency` lint: an author who wants the
        ordering written down can be told exactly which lines to add, without
        the omission being an error.
        """
        declared = {
            (edge.task, edge.depends_on) for edge in self.edges
            if edge.origin == DECLARED
        }
        seen: Set[Tuple[str, str]] = set()
        result = []
        for edge in self.edges:
            key = (edge.task, edge.depends_on)
            if edge.origin == DECLARED or key in declared or key in seen:
                continue
            seen.add(key)
            result.append(edge)
        return result

    def unknown_references(self) -> List[DependencyEdge]:
        """Declared dependencies naming a step that does not exist.

        Only `declared` edges can be unknown: a template reference to a name
        that is not a step is not a dependency at all -- it may be a pipeline
        input, a loop variable or a typo, and deciding which is the data-flow
        validator's job.
        """
        known = set(self.steps)
        return [edge for edge in self.edges if edge.depends_on not in known]

    def cycles(self) -> List[List[str]]:
        """Every dependency cycle, each as a list of step ids.

        Reported before execution because a cycle is a deadlock: no step in it
        can ever become ready.
        """
        adjacency = self.adjacency()
        found: List[List[str]] = []
        seen_signatures: Set[Tuple[str, ...]] = set()
        visiting: List[str] = []
        state: Dict[str, int] = {}

        def visit(node: str) -> None:
            state[node] = 1
            visiting.append(node)
            for dependency in adjacency.get(node, []):
                if dependency not in state:
                    visit(dependency)
                elif state[dependency] == 1:
                    cycle = visiting[visiting.index(dependency):] + [dependency]
                    signature = tuple(sorted(set(cycle)))
                    if signature not in seen_signatures:
                        seen_signatures.add(signature)
                        found.append(cycle)
            visiting.pop()
            state[node] = 2

        for step in self.steps:
            if step not in state:
                visit(step)
        return found

    def levels(self) -> List[List[str]]:
        """Steps grouped so that everything in a level may run concurrently.

        Raises `ValueError` on a cycle: producing a partial schedule for an
        unschedulable pipeline is how a deadlock becomes a silent hang.
        """
        adjacency = self.adjacency()
        remaining = {step: set(deps) & set(self.steps) for step, deps in adjacency.items()}
        done: Set[str] = set()
        result: List[List[str]] = []

        while remaining:
            ready = [step for step, deps in remaining.items() if deps <= done]
            if not ready:
                raise ValueError(
                    f"dependency cycle among {sorted(remaining)}: no step is ready"
                )
            ready = [step for step in self.steps if step in ready]
            result.append(ready)
            done.update(ready)
            for step in ready:
                del remaining[step]
        return result


def _strings_in(value: Any, path: str = "") -> Iterator[Tuple[str, str]]:
    """Every string inside a nested structure, with the path that reached it.

    Templates hide in nested dictionaries and lists as readily as in a plain
    parameter, and a reference is a dependency wherever it appears.
    """
    if isinstance(value, str):
        yield value, path
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings_in(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _strings_in(item, f"{path}[{index}]")


def _declared_dependencies(step: Dict[str, Any]) -> List[str]:
    """The `dependencies:` / `depends_on:` value, however it is written."""
    raw = step.get("dependencies", step.get("depends_on", []))
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def build_dependency_graph(
    pipeline_def: Dict[str, Any], env: Any = None
) -> DependencyGraph:
    """The one graph, from a raw pipeline definition.

    Only names that are actually step ids become edges. A template reference
    to anything else -- a pipeline input, a loop variable, a typo -- is not an
    ordering constraint, and judging which of those it is belongs to the
    data-flow validator.
    """
    steps_def = pipeline_def.get("steps", pipeline_def.get("tasks", [])) or []
    step_ids: List[str] = [
        step["id"] for step in steps_def
        if isinstance(step, dict) and step.get("id")
    ]
    known = set(step_ids)

    edges: List[DependencyEdge] = []
    self_references: List[DependencyEdge] = []

    def add(task: str, depends_on: str, origin: str, location: str) -> None:
        edge = DependencyEdge(task, depends_on, origin, location)
        if depends_on == task:
            # A self-edge can never be satisfied. Recorded, never scheduled.
            self_references.append(edge)
            return
        edges.append(edge)

    for step in steps_def:
        if not isinstance(step, dict) or not step.get("id"):
            continue
        task_id = step["id"]

        for dependency in _declared_dependencies(step):
            add(task_id, dependency, DECLARED, "dependencies")

        for key in CONTROL_FLOW_KEYS:
            if key not in step:
                continue
            for text, path in _strings_in(step[key], key):
                for reference in template_references(text, env):
                    base = reference.split(".", 1)[0]
                    if base in known:
                        add(task_id, base, CONTROL_FLOW, path)

        # Everything else the step carries that may hold a template. `id`,
        # `dependencies` and the control-flow keys above are excluded: the
        # first two are not templates, and the third is already covered with
        # a more precise origin.
        for key, value in step.items():
            if key in ("id", "dependencies", "depends_on") or key in CONTROL_FLOW_KEYS:
                continue
            for text, path in _strings_in(value, key):
                for reference in template_references(text, env):
                    base = reference.split(".", 1)[0]
                    if base in known:
                        add(task_id, base, TEMPLATE, path)

    return DependencyGraph(
        steps=tuple(step_ids),
        edges=tuple(edges),
        self_references=tuple(self_references),
    )
