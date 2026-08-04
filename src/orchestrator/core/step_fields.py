"""Which parts of a step are rendered at run time, and which are copied verbatim.

A template that never renders cannot create a dependency. `name:` is copied
onto the task and shown in logs; nothing substitutes into it. So this pipeline
has no ordering constraint at all::

    - id: a
      name: "{{ b.result }}"
    - id: b
      name: "{{ a.result }}"

The canonical dependency graph nevertheless read `{{ b.result }}` as "a needs
b", `{{ a.result }}` as "b needs a", and rejected the pipeline with
`Dependency cycle detected: a -> b -> a`. Two inert strings produced a hard
compile error.

The cause was a blocklist. `build_dependency_graph` scanned every key it did
not specifically exclude, which meant it scanned `name`, `description`,
`metadata` and anything else a step happened to carry. Since that same graph
now drives scheduling and execution levels as well as cycle detection, a
spurious edge does not merely warn -- it serialises independent work, invents
cycles, and misreports which dependencies were inferred.

The list below is an allowlist, and each entry is here because the runtime was
read rather than guessed:

`parameters`
    `ControlSystem._render_task_templates` deep-renders it.
`action`
    Same function renders it when it is a string.
`location`
    Kept as `location_template` on the task's output metadata and resolved
    when the output is recorded, which is why real examples write
    ``location: "./reports/{{ inputs.topic | slugify }}.md"``.

Control-flow keys (`for_each`, `condition`, `while`, ...) are also rendered,
but they are inferred separately in `dependency_graph` so their edges carry
the more precise `control_flow` origin.

Everything absent from this list is inert: a reference inside it orders
nothing. Adding a field here without first confirming that the runtime renders
it re-opens the false-cycle bug; omitting one the runtime *does* render
re-opens #465, where a step ran before the value it needed existed. The
catalogue and `examples/supported/` are the check on both directions --
the former for false rejections, the latter because those pipelines actually
run.
"""

from __future__ import annotations

from typing import FrozenSet, Tuple

#: Step fields whose templates are rendered at run time. A reference in one of
#: these is a real ordering constraint.
RENDERABLE_STEP_FIELDS: Tuple[str, ...] = (
    "parameters",
    "action",
    "location",
)

#: Fields holding child steps. Their own renderable fields count, attributed to
#: the enclosing step -- a child is not scheduled independently of its parent.
NESTED_STEP_FIELDS: Tuple[str, ...] = (
    "steps",
)

#: Prose. Nothing renders it and nothing acts on it, so a template here is the
#: author's mistake to know about but not a defect in the pipeline: the braces
#: appear in a log line and everything still runs.
INERT_PROSE_STEP_FIELDS: FrozenSet[str] = frozenset({
    "name",          # human-readable label, copied verbatim
    "description",   # prose
})

#: Also never rendered -- and that makes a template here an error, not a note.
#: These fields *name* things: a step, a tool in the registry, a dependency to
#: schedule against. A literal `{{ x }}` names nothing, so the pipeline is
#: already broken; warning about the wording and calling it valid says the
#: opposite.
NON_RENDERED_STRUCTURAL_STEP_FIELDS: FrozenSet[str] = frozenset({
    "id",            # the step's own name
    "tool",          # a registry key
    "dependencies",  # read with the `declared` origin
    "depends_on",
})

#: `metadata` is arbitrary author data *except* for the keys the runtime
#: reads. Those are not prose: an unrendered `{{ ... }}` is handed to control
#: code as a literal string, so `goto: "{{ nosuch }}"` sends execution to a
#: step named `{{ nosuch }}` and `timeout: "{{ t }}"` is a timeout of that
#: text. Each key here has a runtime read behind it:
#:
#:     goto                  orchestrator.py:1074, control_flow/dynamic_flow.py:53
#:     priority              orchestrator.py:1345
#:     requires_model        orchestrator.py:2599
#:     dynamic_dependencies  orchestrator.py:1166
#:     on_failure            orchestrator.py:1111
#:     timeout               orchestrator.py:1888
#:     max_iterations        orchestrator.py:1001
#:     condition             runtime/orchestrator_integration.py:123
#:     output_schema         adapters/enhanced_langgraph_adapter.py:222
#:     produces              auto_resolution/requirements_analyzer.py:219
#:     required_capabilities core/control_system.py:130
#:     validation            compiler/schema_resolver.py:149
#:
#: Keys the compiler *writes* to metadata (`step_type`, `retry_count`, the
#: loop bookkeeping) are absent: an author does not supply them, so a template
#: in one is not a case that arises.
OPERATIONAL_METADATA_KEYS: FrozenSet[str] = frozenset({
    "goto", "priority", "requires_model", "dynamic_dependencies",
    "on_failure", "timeout", "max_iterations", "condition",
    "output_schema", "produces", "required_capabilities", "validation",
})

#: Everything a step carries that the runtime does not render, whatever the
#: consequence of writing a template in it. Dependency inference only needs
#: "not scanned"; the distinctions above are for diagnostics.
INERT_STEP_FIELDS: FrozenSet[str] = (
    INERT_PROSE_STEP_FIELDS
    | NON_RENDERED_STRUCTURAL_STEP_FIELDS
    | frozenset({"metadata"})
)

#: The same distinction one level up. A pipeline's own `name` and
#: `description` are prose about the pipeline; nothing renders them either, so
#: `name: "{{ nosuch }}"` at the top of a document was failing validation for a
#: pipeline that runs.
#:
#: Deliberately short. `outputs` *is* rendered; `parameters` declares names
#: rather than using them; and `id` and `version` are schema-constrained --
#: `version` must match `\d+\.\d+\.\d+`, so a template there is a real error
#: and calling the field inert would describe it wrongly. Only fields a real
#: run tolerates an unresolvable reference in are listed, which is what
#: `test_a_pipeline_with_templates_in_prose_still_runs` checks.
INERT_PIPELINE_FIELDS: FrozenSet[str] = frozenset({
    "name",
    "description",
    "metadata",
})
