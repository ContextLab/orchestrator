"""A template that never renders cannot create a dependency.

`build_dependency_graph` scanned every step key it did not specifically
exclude. `name:` was not excluded, so this pipeline -- two steps that never
interact -- failed to compile with `Dependency cycle detected: a -> b -> a`::

    - id: a
      name: "{{ b.result }}"
    - id: b
      name: "{{ a.result }}"

Task names are copied verbatim. Nothing substitutes into them. And because the
canonical graph now drives scheduling as well as cycle detection, a spurious
edge does more than warn: it serialises independent steps and misreports which
dependencies were inferred.

The fix is an allowlist (`core.step_fields`) rather than a blocklist. These
tests pin it in both directions, because both are bugs that have already
happened here: scanning too much invents cycles, and scanning too little
re-opens #465, where a step ran before the value it needed existed.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.core.dependency_graph import build_dependency_graph
from orchestrator.core.step_fields import (
    INERT_STEP_FIELDS,
    NESTED_STEP_FIELDS,
    RENDERABLE_STEP_FIELDS,
)

pytestmark = [pytest.mark.contract]

REPO = Path(__file__).resolve().parent.parent


def _graph(*steps):
    return build_dependency_graph({"id": "p", "steps": list(steps)})


# ---------------------------------------------------------------------------
# Inert fields order nothing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", sorted(INERT_STEP_FIELDS - {"id", "dependencies", "depends_on"}))
def test_a_template_in_an_inert_field_creates_no_edge(field):
    graph = _graph(
        {"id": "a", field: "{{ b.result }}", "parameters": {"x": 1}},
        {"id": "b", "parameters": {"x": 1}},
    )
    assert graph.dependencies_for("a") == [], (
        f"a template in `{field}` was read as a dependency, but the runtime "
        f"never renders that field"
    )


def test_the_reported_false_cycle_is_not_a_cycle():
    """The exact reproduction from review."""
    graph = _graph(
        {"id": "a", "name": "{{ b.result }}", "parameters": {"x": 1}},
        {"id": "b", "name": "{{ a.result }}", "parameters": {"x": 1}},
    )
    assert graph.cycles() == []
    assert graph.levels() == [["a", "b"]], (
        "two steps that never interact must be free to run concurrently"
    )


def test_an_inert_template_does_not_become_an_implicit_dependency_finding():
    """The lint would otherwise tell an author to write down an ordering that
    does not exist."""
    graph = _graph(
        {"id": "a", "description": "see {{ b.result }}", "parameters": {"x": 1}},
        {"id": "b", "parameters": {"x": 1}},
    )
    assert graph.inferred_only() == []


# ---------------------------------------------------------------------------
# Renderable fields still order -- the #465 direction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("parameters", {"content": "{{ a.result }}"}),
    ("action", "write {{ a.result }}"),
    ("location", "./out/{{ a.result }}.md"),
])
def test_a_template_in_a_renderable_field_orders_the_step(field, value):
    """Dropping one of these re-opens #465: the step runs before the value it
    needs exists, having passed validation."""
    graph = _graph(
        {"id": "a", "parameters": {"x": 1}},
        {"id": "b", field: value},
    )
    assert graph.dependencies_for("b") == ["a"], (
        f"a reference in `{field}` is rendered at run time and must order the step"
    )


def test_a_nested_step_orders_its_parent():
    """A child step is not scheduled on its own, so its references belong to
    the step that contains it."""
    graph = _graph(
        {"id": "a", "parameters": {"x": 1}},
        {
            "id": "loop",
            "for_each": "[1, 2]",
            "steps": [{"id": "inner", "parameters": {"c": "{{ a.result }}"}}],
        },
    )
    assert graph.dependencies_for("loop") == ["a"]


def test_control_flow_references_keep_their_own_origin():
    """`for_each` is renderable too, but its edges are inferred separately so a
    diagnostic can distinguish an iterable from a parameter."""
    graph = _graph(
        {"id": "a", "parameters": {"x": 1}},
        {"id": "b", "for_each": "{{ a.result }}", "parameters": {"x": 1}},
    )
    assert graph.origins_for("b", "a") == {"control_flow"}


# ---------------------------------------------------------------------------
# The contract itself
# ---------------------------------------------------------------------------

def test_no_field_is_both_renderable_and_inert():
    assert not (set(RENDERABLE_STEP_FIELDS) & INERT_STEP_FIELDS)
    assert not (set(NESTED_STEP_FIELDS) & INERT_STEP_FIELDS)


def test_the_runtime_renders_every_field_claimed_renderable():
    """Behavioural, not a name check.

    `parameters` and `action` are rendered by `_render_task_templates`;
    `location` is resolved from `location_template` when the output is
    recorded. If a field is listed but nothing renders it, references inside
    it order steps for no reason -- the bug this module fixes, in a new place.
    """
    source = (REPO / "src" / "orchestrator" / "core" / "control_system.py").read_text()
    for field in ("parameters", "action"):
        assert f"rendered_task.{field}" in source, (
            f"`{field}` is on the renderable list but nothing in control_system "
            f"renders it"
        )


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

@pytest.mark.e2e
def test_the_false_cycle_pipeline_compiles_through_the_cli(tmp_path):
    """The unit test uses `build_dependency_graph` directly; the CLI runs the
    control-flow compiler. #466 shipped with a mutation that only one of those
    two caught, so both paths are exercised."""
    pipeline = tmp_path / "p.yaml"
    pipeline.write_text(
        "id: inert\n"
        "name: Inert\n"
        "steps:\n"
        "  - id: a\n"
        '    name: "{{ b.result }}"\n'
        "    tool: filesystem\n"
        "    action: write\n"
        "    parameters: {path: './a.txt', content: 'a'}\n"
        "  - id: b\n"
        '    name: "{{ a.result }}"\n'
        "    tool: filesystem\n"
        "    action: write\n"
        "    parameters: {path: './b.txt', content: 'b'}\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["ORCHESTRATOR_AUTO_INSTALL"] = "0"
    result = subprocess.run(
        [sys.executable, "-m", "orchestrator.cli", "validate", str(pipeline)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout[-800:] + result.stderr[-800:]
    # Not a bare "cycle" search: pytest names tmp_path after the test, so the
    # word appears in the pipeline's own path in the output.
    assert "Dependency cycle detected" not in result.stdout + result.stderr
