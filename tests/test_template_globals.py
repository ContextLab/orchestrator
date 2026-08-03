"""A global must be called, and called correctly.

#450 taught the validators the eight global *names*, which stopped `{{ now() }}`
-- a pipeline that runs correctly -- from being reported as an undefined
variable. It stopped at the name. Every other way of writing the name was
accepted and then failed at run time, except the one that did not fail at all:

    {{ now }}   ->   validates, runs, and writes
                     "<function TemplateManager._setup_custom_filters.<locals>.now
                      at 0x1084...>" into the artifact

Nothing errored. The repr of a live function object was written where the
author expected a timestamp. (It is not a sandbox escape: `now.__globals__` and
`now.__class__` are refused by the sandboxed environment #447 introduced. What
leaks is the repr, not the object graph -- and these tests pin that too.)

These tests are the negative half of the contract: what the language refuses.
"""

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.core.template_globals import (
    DEPRECATED,
    GLOBAL_NAMES,
    GLOBAL_SPECS,
    NOT_CALLED,
    WRONG_ARITY,
    find_global_misuse,
)
from orchestrator.core.template_manager import TemplateManager
from orchestrator.core.template_sandbox import pipeline_global_names

pytestmark = [pytest.mark.contract]

ROOT = Path(__file__).resolve().parent.parent


def _misuse(expression):
    """Codes that would *refuse* the template. Warnings are not refusals."""
    ast = TemplateManager().env.parse(expression)
    return {m.code for m in find_global_misuse(ast) if m.severity == "error"}


def _reported(expression):
    """Every code, whatever its severity."""
    ast = TemplateManager().env.parse(expression)
    return {m.code for m in find_global_misuse(ast)}


# ---------------------------------------------------------------------------
# The registry describes the functions the runtime actually registers
# ---------------------------------------------------------------------------

def test_the_registry_matches_what_the_runtime_registers():
    """The drift check that derivation used to give for free.

    #450 derived the names from `TemplateManager`, so they could not disagree.
    Declaring them buys an argument contract and costs that guarantee, unless
    something asserts it -- this.
    """
    registered = set(TemplateManager().env.globals) - set(
        __import__("jinja2").sandbox.SandboxedEnvironment().globals
    )
    assert GLOBAL_NAMES == registered, (
        f"the declared language and the runtime disagree: "
        f"declared_only={sorted(GLOBAL_NAMES - registered)}, "
        f"registered_only={sorted(registered - GLOBAL_NAMES)}"
    )


@pytest.mark.parametrize("spec", GLOBAL_SPECS, ids=lambda s: s.name)
def test_each_declared_arity_matches_the_real_signature(spec):
    """A declared contract that does not match the callable is worse than none.

    It would reject calls that work, or accept calls that cannot.
    """
    func = TemplateManager().env.globals[spec.name]
    signature = inspect.signature(func)

    positional = [
        p for p in signature.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    required = sum(1 for p in positional if p.default is p.empty)
    unbounded = any(p.kind is p.VAR_POSITIONAL for p in signature.parameters.values())

    assert spec.min_args == required, (
        f"{spec.name} declares {spec.min_args} required argument(s) but its "
        f"signature {signature} requires {required}"
    )
    if unbounded:
        assert spec.max_args is None, f"{spec.name} takes *args but declares a maximum"
    else:
        assert spec.max_args == len(positional), (
            f"{spec.name} declares a maximum of {spec.max_args} but its "
            f"signature {signature} accepts {len(positional)}"
        )


def test_pipeline_global_names_comes_from_the_registry():
    assert pipeline_global_names() == GLOBAL_NAMES


# ---------------------------------------------------------------------------
# What the language refuses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "expression,expected",
    [
        # Named but never called: renders the function object itself.
        ("{{ now }}", NOT_CALLED),
        ("{{ current_loop_name }}", NOT_CALLED),
        # Attribute access on the function object.
        ("{{ now.foo }}", NOT_CALLED),
        ("{{ file_exists.bad }}", NOT_CALLED),
        ("{{ active_loops.x.y }}", NOT_CALLED),
        # Indexing it.
        ("{{ now[0] }}", NOT_CALLED),
        ("{{ active_loops['a'] }}", NOT_CALLED),
        # Passed somewhere as a value.
        ("{{ [1, 2] | map(now) | list }}", NOT_CALLED),
        ("{% if now %}x{% endif %}", NOT_CALLED),
        # Called with the wrong number of arguments.
        ("{{ now(1) }}", WRONG_ARITY),
        ("{{ now(1, 2, 3) }}", WRONG_ARITY),
        ("{{ file_exists() }}", WRONG_ARITY),
        ("{{ file_exists('a', 'b', 'c') }}", WRONG_ARITY),
        ("{{ loop_var('only_one') }}", WRONG_ARITY),
        ("{{ current_loop_name('unexpected') }}", WRONG_ARITY),
    ],
)
def test_misuse_is_detected(expression, expected):
    assert expected in _misuse(expression), (
        f"{expression} is not a usable call and was accepted"
    )


@pytest.mark.parametrize(
    "expression",
    [
        "{{ now() }}",
        "{{ current_loop_name() }}",
        "{{ active_loops() }}",
        "{{ file_exists('a.txt') }}",
        "{{ file_exists('a.txt', '/base') }}",
        "{{ include_file('a.txt') }}",
        "{{ loop_var('outer', 'item') }}",
        "{{ loop_item_at('outer', 0) }}",
        # Keyword arguments are the callable's business, not the arity check's.
        "{{ file_exists('a.txt', base_dir='/base') }}",
        # Nested in a larger expression.
        "{{ 'yes' if file_exists('a.txt') else 'no' }}",
        "{{ now() | string | upper }}",
        # A name that merely contains a global's name is not that global.
        "{{ nowhere }}",
        "{{ my_now }}",
    ],
)
def test_valid_use_is_left_alone(expression):
    assert _misuse(expression) == set(), f"{expression} is valid and was rejected"


def test_a_deprecated_global_is_reported_but_not_refused():
    """`now()` still runs -- pipelines in the wild use it -- and says so.

    Refusing it would break working pipelines to make a style point. Staying
    silent would leave authors on a function that gives a different answer to
    every step of one run.
    """
    assert DEPRECATED in _reported("{{ now() }}")
    assert _misuse("{{ now() }}") == set(), "a deprecated global must still validate"


def test_a_global_with_no_replacement_is_not_reported():
    assert _reported("{{ file_exists('a.txt') }}") == set()


def test_deprecation_is_not_reported_for_a_call_that_cannot_work():
    """One problem at a time: fix the call, then hear about the replacement."""
    assert _reported("{{ now(1, 2) }}") == {WRONG_ARITY}


@pytest.mark.parametrize(
    "expression",
    [
        "{% for now in items %}{{ now }}{% endfor %}",
        "{% set now = 'x' %}{{ now }}",
        "{% macro f(now) %}{{ now }}{% endmacro %}",
    ],
)
def test_a_rebound_name_is_not_our_global(expression):
    """A template that binds the name is talking about its own variable.

    The binding site is a `store`, but the use is an ordinary `load` and looks
    exactly like ours. Rejecting it would be a false positive of the kind the
    last three changes here existed to remove.
    """
    assert _misuse(expression) == set()


def test_one_expression_can_hold_a_valid_and_an_invalid_use():
    """Identity, not spelling: two `now` nodes, only one of them a call."""
    assert NOT_CALLED in _misuse("{{ now() }}{{ now.foo }}")


# ---------------------------------------------------------------------------
# The validator actually asks
# ---------------------------------------------------------------------------

def _compile(expression):
    """Compile the way `orchestrator validate` does. Returns None or the error."""
    import asyncio

    from orchestrator.compiler.yaml_compiler import YAMLCompiler

    pipeline = f"""
id: probe
name: Probe
steps:
  - id: write_it
    tool: filesystem
    action: write
    parameters:
      path: "./out.txt"
      content: "{expression}"
"""
    try:
        asyncio.run(YAMLCompiler().compile(pipeline, {}))
        return None
    except Exception as exc:  # noqa: BLE001 - the message is the subject
        return str(exc)


@pytest.mark.parametrize(
    "expression",
    ["{{ now }}", "{{ now.foo }}", "{{ file_exists() }}", "{{ now(1, 2, 3) }}"],
)
def test_the_validator_refuses_a_misused_global(expression):
    """The checks above test the function; this tests that anything calls it.

    Without this, deleting the call site in the template validator leaves every
    test in this file passing.
    """
    assert _compile(expression) is not None, (
        f"{expression} compiles, so the misuse check is not wired into validation"
    )


@pytest.mark.parametrize(
    "expression", ["{{ now() }}", "{{ file_exists('a.txt') }}"]
)
def test_the_validator_accepts_a_proper_call(expression):
    assert _compile(expression) is None, f"{expression} is valid and was refused"


# ---------------------------------------------------------------------------
# The function object itself must stay out of reach
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "expression",
    [
        "{{ now.__globals__ }}",
        "{{ now.__class__ }}",
        "{{ now.__call__() }}",
        "{{ now.__globals__['__builtins__'] }}",
    ],
)
def test_a_global_is_not_a_way_out_of_the_sandbox(expression):
    """`{{ now }}` puts a live function in reach of the template.

    The sandbox refuses to traverse it, and that is what keeps this a
    correctness bug rather than the #447 escape all over again. Rendering, not
    validating, is the subject here: this must hold even for a template that
    never went through the validator.
    """
    from jinja2.exceptions import SecurityError, UndefinedError

    env = TemplateManager().env
    with pytest.raises((SecurityError, UndefinedError)):
        env.from_string(expression).render()


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

def test_the_generated_globals_docs_are_committed_and_current():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_template_globals_docs.py"),
         "--check"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"docs/template_globals.md is out of date. Run:\n"
        f"    python scripts/generate_template_globals_docs.py\n\n"
        f"{result.stdout}\n{result.stderr}"
    )


def test_every_global_appears_in_the_documentation():
    text = (ROOT / "docs" / "template_globals.md").read_text()
    for spec in GLOBAL_SPECS:
        assert f"`{spec.name}`" in text, f"{spec.name} is undocumented"


def test_no_example_still_calls_a_deprecated_global():
    """The catalogue was migrated; this keeps it migrated.

    `{{ now() }}` gave each step of a run a different answer, so the six
    examples using it stamped their reports inconsistently. They now use
    `execution.timestamp`.
    """
    import re

    deprecated = {s.name for s in GLOBAL_SPECS if s.deprecated_for}
    if not deprecated:
        # An empty alternation matches every call, so guard rather than report
        # the whole catalogue as offending.
        return
    pattern = re.compile(r"\{\{[^}]*\b(" + "|".join(sorted(deprecated)) + r")\s*\(")

    offenders = sorted(
        f"{path.relative_to(ROOT)}: {pattern.search(path.read_text()).group(0)}"
        for path in (ROOT / "examples").rglob("*.yaml")
        if pattern.search(path.read_text())
    )
    assert not offenders, (
        "these examples call a deprecated global; the replacement is in "
        f"docs/template_globals.md: {offenders}"
    )
