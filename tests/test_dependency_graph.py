"""A template reference is a dependency, and the schedule agrees.

`{{ make.path }}` says plainly that the value cannot exist until `make` has
run. Three parts of the compiler agreed and none told the scheduler:
`DependencyValidator` inferred edges from `for_each`/`condition`/`while` but
never looked inside `parameters`; `DataFlowValidator` built a graph including
parameters and logged it; `YAMLCompiler._analyze_template` found the same
references with a third regex and stored them where nothing scheduled from
them. `Task.dependencies` came from the explicit key alone, so this validated
and then failed at run time (#465)::

    - id: make
      parameters: {path: "./a.txt", content: "A"}
    - id: use
      parameters: {content: "{{ make.path }}"}     # no `dependencies:`

These tests pin the replacement: one graph, and the same graph is what cycle
validation inspects and what every `Task.dependencies` is built from. The
danger of fixing this by appending edges inside `_build_task` is a schedule
nobody validated, so several of these ask the compiled pipeline rather than
the graph object.
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.core.dependency_graph import (
    CONTROL_FLOW,
    DECLARED,
    TEMPLATE,
    build_dependency_graph,
)

pytestmark = [pytest.mark.contract]

REPO = Path(__file__).resolve().parent.parent


def _graph(*steps):
    return build_dependency_graph({"id": "p", "steps": list(steps)})


def _step(step_id, content=None, **extra):
    step = {"id": step_id, "tool": "filesystem", "action": "write",
            "parameters": {"path": f"./{step_id}.txt", "content": content or step_id}}
    step.update(extra)
    return step


def _compile(*steps):
    """Compile the way `orchestrator validate` does."""
    import yaml as _yaml

    source = _yaml.safe_dump({"id": "p", "name": "P", "steps": list(steps)})
    return asyncio.run(YAMLCompiler().compile(source, {}))


def _compile_error(*steps):
    try:
        _compile(*steps)
        return None
    except Exception as exc:  # noqa: BLE001 - the message is the subject
        return str(exc)


def _cli(command, pipeline, cwd):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["ORCHESTRATOR_AUTO_INSTALL"] = "0"
    env.pop("ANTHROPIC_API_KEY", None)
    return subprocess.run(
        [sys.executable, "-m", "orchestrator.cli", command, str(pipeline)],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=300,
    )


# ---------------------------------------------------------------------------
# The reference orders the steps
# ---------------------------------------------------------------------------

def test_a_referenced_step_precedes_its_consumer():
    graph = _graph(_step("make"), _step("use", "{{ make.path }}"))
    assert graph.dependencies_for("use") == ["make"]
    assert graph.dependencies_for("make") == []


def test_the_compiled_task_carries_the_inferred_edge():
    """The graph being right is not enough; `Task.dependencies` is what runs."""
    pipeline = _compile(_step("make"), _step("use", "{{ make.path }}"))
    assert pipeline.tasks["use"].dependencies == ["make"]


def test_execution_levels_put_the_producer_first():
    pipeline = _compile(_step("make"), _step("use", "{{ make.path }}"))
    levels = pipeline.get_execution_levels()
    assert levels.index(["make"]) < levels.index(["use"]), levels


def test_the_edge_records_where_it_came_from():
    """A diagnostic that says *why* two steps are ordered reads very
    differently from one that only says they are."""
    graph = _graph(_step("make"), _step("use", "{{ make.path }}"))
    edge = next(e for e in graph.edges if e.task == "use")
    assert edge.origin == TEMPLATE
    assert edge.location == "parameters.content"


# ---------------------------------------------------------------------------
# The issue #465 reproduction
# ---------------------------------------------------------------------------

def _repro_pipeline(reference):
    return (
        "id: uo\nname: UO\nsteps:\n"
        "  - id: make\n    tool: filesystem\n    action: write\n"
        "    parameters:\n      path: \"./out_a.txt\"\n      content: \"A ran\"\n"
        "  - id: use\n    tool: filesystem\n    action: write\n"
        "    parameters:\n      path: \"./out_b.txt\"\n"
        f"      content: \"made={reference}\"\n"
    )


@pytest.mark.e2e
def test_the_issue_465_reproduction_runs(tmp_path):
    """The pipeline that validated and then could not run.

    `use` references `make` and declares no `dependencies:`. Before the graph
    was shared, the edge was computed and discarded, so both steps went into
    the same level, `make`'s result was not in context, and the render failed.

    The reference is `make.result.path`. The issue was filed with
    `make.path`, which is not a field the filesystem tool returns -- so that
    reproduction conflated the missing edge with a genuinely wrong field name,
    and would fail here for the second reason even with the first fixed. The
    companion test below covers that case separately.
    """
    pipeline = tmp_path / "p.yaml"
    pipeline.write_text(_repro_pipeline("{{ make.result.path }}"))

    assert _cli("validate", pipeline, tmp_path).returncode == 0
    result = _cli("run", pipeline, tmp_path)
    assert result.returncode == 0, f"{result.stdout[-700:]}{result.stderr[-700:]}"

    written = (tmp_path / "out_b.txt").read_text()
    assert "{{" not in written, f"the reference never resolved: {written!r}"
    assert "out_a" in written, f"the producer's output did not reach it: {written!r}"


@pytest.mark.e2e
def test_a_field_the_producer_does_not_return_still_fails_loudly(tmp_path):
    """Ordering a step correctly does not invent a field it never produced.

    `make.path` does not exist -- the filesystem tool returns
    `{'result': {'path': ...}, 'success': ...}`. The step is stopped rather
    than writing an empty value, which is the right outcome. What is wrong is
    that `validate` says nothing: the data-flow validator warns, because
    `make` declares no outputs, and that warning never reaches stdout.
    """
    pipeline = tmp_path / "p.yaml"
    pipeline.write_text(_repro_pipeline("{{ make.path }}"))

    assert _cli("validate", pipeline, tmp_path).returncode == 0
    result = _cli("run", pipeline, tmp_path)
    assert result.returncode != 0, "a reference to a field that does not exist ran"
    assert not (tmp_path / "out_b.txt").exists(), (
        "the step wrote a file despite an unresolved reference"
    )


# ---------------------------------------------------------------------------
# Explicit and inferred together
# ---------------------------------------------------------------------------

def test_an_explicit_and_inferred_edge_is_one_dependency():
    graph = _graph(_step("make"),
                   _step("use", "{{ make.path }}", dependencies=["make"]))
    assert graph.dependencies_for("use") == ["make"]
    assert graph.origins_for("use", "make") == {DECLARED, TEMPLATE}


def test_an_explicit_dependency_with_no_reference_survives():
    """Ordering that is only stated, never implied, must not be dropped."""
    graph = _graph(_step("first"), _step("second", dependencies=["first"]))
    assert graph.dependencies_for("second") == ["first"]


def test_dependencies_are_reported_in_declaration_order():
    """So a pipeline compiles to the same graph however its templates are
    written -- discovery order would make the schedule depend on text layout."""
    graph = _graph(_step("a"), _step("b"),
                   _step("c", "{{ b.path }} {{ a.path }}"))
    assert graph.dependencies_for("c") == ["a", "b"]


def test_a_forward_reference_orders_correctly():
    """A step may reference one written later in the file."""
    graph = _graph(_step("first", "{{ later.path }}"), _step("later"))
    assert graph.dependencies_for("first") == ["later"]
    assert graph.levels() == [["later"], ["first"]]


# ---------------------------------------------------------------------------
# What must still be refused
# ---------------------------------------------------------------------------

def test_an_inferred_cycle_is_rejected_before_execution():
    """A cycle is a deadlock: no step in it can ever become ready.

    It must be refused at compile time whether it was written as an explicit
    `dependencies:` entry or implied by a template.
    """
    error = _compile_error(_step("a", "{{ b.path }}"), _step("b", "{{ a.path }}"))
    assert error is not None, "a cyclic pipeline compiled"
    assert "cycle" in error.lower(), error


def test_a_longer_inferred_cycle_is_rejected():
    error = _compile_error(_step("a", "{{ c.path }}"),
                           _step("b", "{{ a.path }}"),
                           _step("c", "{{ b.path }}"))
    assert error is not None and "cycle" in error.lower(), error


def test_a_self_reference_is_never_an_edge():
    """A self-edge cannot be satisfied, so it is recorded and not scheduled."""
    graph = _graph(_step("s", "{{ s.path }}"))
    assert graph.edges == ()
    assert [(e.task, e.depends_on) for e in graph.self_references] == [("s", "s")]


def test_a_self_reference_is_rejected_by_validation():
    error = _compile_error(_step("s", "{{ s.path }}"))
    assert error is not None, "a step referencing itself compiled"


def test_a_reference_to_a_nonexistent_step_is_still_an_error():
    """Inference must not turn a typo into silence: `{{ ghost.path }}` names
    nothing, so it is not an edge, and the data-flow validator still reports
    it."""
    graph = _graph(_step("real"), _step("use", "{{ ghost.path }}"))
    assert graph.dependencies_for("use") == []

    error = _compile_error(_step("real"), _step("use", "{{ ghost.path }}"))
    assert error is not None and "ghost" in error, error


# ---------------------------------------------------------------------------
# Where references hide
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "content",
    [
        "{{ src.path }}",                                   # plain
        "{{ src.path | upper }}",                           # through a filter
        "{{ src['path'] }}",                                # subscript
        "{{ src.items[0] }}",                               # index
        "{{ 'yes' if src.ok else 'no' }}",                  # conditional
        "{% if src.ok %}x{% endif %}",                      # statement
        "{{ [src.a, src.b] | join(',') }}",                 # inside a list literal
    ],
)
def test_a_reference_is_found_however_it_is_written(content):
    graph = _graph(_step("src"), _step("use", content))
    assert graph.dependencies_for("use") == ["src"], content


def test_a_reference_nested_in_dictionaries_and_lists_is_found():
    """Templates hide in nested structures as readily as in a plain parameter."""
    step = {"id": "deep", "action": "generate",
            "parameters": {"cfg": {"items": ["{{ src.a }}", {"k": "{{ src.b }}"}]}}}
    graph = _graph(_step("src"), step)
    assert graph.dependencies_for("deep") == ["src"]


@pytest.mark.parametrize("key", ["condition", "for_each", "while", "until"])
def test_a_control_flow_reference_orders_the_step(key):
    graph = _graph(_step("src"), _step("use", **{key: "{{ src.items }}"}))
    assert graph.dependencies_for("use") == ["src"]
    assert graph.origins_for("use", "src") == {CONTROL_FLOW}


def test_a_loop_local_name_is_not_a_step_dependency():
    """`{% for src in ... %}` rebinds the name; the body is talking about the
    loop variable, not the step that happens to share its id."""
    graph = _graph(
        _step("src"),
        _step("use", "{% for src in [1, 2] %}{{ src }}{% endfor %}"),
    )
    assert graph.dependencies_for("use") == []


def test_a_reference_outside_the_loop_that_shadows_it_still_counts():
    graph = _graph(
        _step("src"),
        _step("use", "{{ src.path }}{% for src in [1] %}{{ src }}{% endfor %}"),
    )
    assert graph.dependencies_for("use") == ["src"]


def test_a_pipeline_input_is_not_a_step_dependency():
    graph = _graph(_step("src"), _step("use", "{{ inputs.topic }}"))
    assert graph.dependencies_for("use") == []


# ---------------------------------------------------------------------------
# The unknown-output contract is unchanged
# ---------------------------------------------------------------------------

def test_an_unknown_field_of_a_declared_output_is_still_an_error():
    """Inferring the edge must not soften the check on the field name."""
    producer = {"id": "make", "action": "generate",
                "outputs": {"content": {"type": "string"}}, "parameters": {}}
    error = _compile_error(producer, _step("use", "{{ make.contnet }}"))
    assert error is not None, "a misspelled declared output compiled"


def test_an_unknown_field_of_an_undeclared_output_is_still_only_a_warning():
    """When a task does not declare its outputs the validator has no basis to
    reject a field name -- the false-positive class removed in #448/#450/#461.
    """
    assert _compile_error(_step("make"), _step("use", "{{ make.anything }}")) is None


def test_the_undeclared_case_still_orders_the_steps():
    """The warning is about the *field*; the *step* reference is certain."""
    graph = _graph(_step("make"), _step("use", "{{ make.anything }}"))
    assert graph.dependencies_for("use") == ["make"]


# ---------------------------------------------------------------------------
# The lint for authors who want it written down
# ---------------------------------------------------------------------------

def test_an_inferred_edge_is_reported_as_implicit():
    graph = _graph(_step("make"), _step("use", "{{ make.path }}"))
    implicit = graph.inferred_only()
    assert [(e.task, e.depends_on) for e in implicit] == [("use", "make")]


def test_an_edge_that_is_also_declared_is_not_implicit():
    graph = _graph(_step("make"),
                   _step("use", "{{ make.path }}", dependencies=["make"]))
    assert graph.inferred_only() == []


def _implicit_issues(*steps):
    """Compile and return the `implicit_dependency` findings."""
    import yaml as _yaml

    compiler = YAMLCompiler()
    source = _yaml.safe_dump({"id": "p", "name": "P", "steps": list(steps)})
    asyncio.run(compiler.compile(source, {}))
    return [
        issue for issue in compiler.validation_report.issues
        if issue.code == "implicit_dependency"
    ]


def test_the_lint_reaches_the_validation_report():
    """The graph knowing an edge is implicit is no use unless something says so."""
    issues = _implicit_issues(_step("make"), _step("use", "{{ make.result.path }}"))
    assert len(issues) == 1, [i.message for i in issues]
    assert issues[0].metadata["referenced_step"] == "make"
    assert issues[0].metadata["parameter_path"] == "parameters.content"


def test_the_lint_is_informational_and_does_not_fail_validation():
    """Inference is a supported way to write a pipeline, not a mistake.

    Making this an error would break every pipeline that relies on it, which
    is the option this issue deliberately did not take.
    """
    issues = _implicit_issues(_step("make"), _step("use", "{{ make.result.path }}"))
    assert issues[0].severity.value == "info"
    assert not issues[0].is_error and not issues[0].is_warning


def test_the_lint_names_the_line_to_add():
    """A lint that reports a problem without the fix is a nag."""
    issues = _implicit_issues(_step("make"), _step("use", "{{ make.result.path }}"))
    assert any("make" in suggestion for suggestion in issues[0].suggestions)


def test_a_declared_dependency_produces_no_lint():
    assert _implicit_issues(
        _step("make"),
        _step("use", "{{ make.result.path }}", dependencies=["make"]),
    ) == []
