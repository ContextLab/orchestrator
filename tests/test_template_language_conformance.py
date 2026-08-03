"""One pipeline language, rendered identically wherever it is rendered.

A pipeline's `{{ }}` expressions pass through several Jinja environments: the
template validator checks them, the YAML compiler renders the ones it can
resolve at compile time, and the runtime renders the rest. If those
environments disagree, an expression means different things depending on which
one reaches it -- and the disagreement shows up as a pipeline that validates
and then fails, or fails to compile and then runs correctly.

#448 made all three environments share one filter registry, and asserted it by
comparing *filter names*. Matching names are not matching behaviour. The
compiler re-registered eleven of those filters immediately afterwards with its
own implementations, so the name sets agreed while the semantics did not:

    {{ 0 | default('X') }}          runtime '0'   compiler 'X'
    {{ '' | default('X') }}         runtime ''    compiler 'X'
    {{ missing | default('X') }}    runtime 'X'   compiler UndefinedError

The last one is the case `default` exists for, and the compiler was the only
environment that could not do it.

These tests compare *results*, not registries: the same expression and input
through every environment, asserting the same value or the same failure.

Globals are the other half of the language, and they do not want the same
treatment. `now()`, `file_exists()` and the loop helpers each answer a question
about the state of a *run*, so only the runtime can answer them; copying them
into the compiler the way filters are copied makes it answer at compile time,
before any step has run, and write a confident wrong answer to a file. The rule
for them is therefore split: every environment knows the *names* -- otherwise
`{{ now() }}` is reported undefined on a pipeline that runs correctly -- and
only the runtime holds the implementations.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.core.template_manager import TemplateManager
from orchestrator.core.template_sandbox import pipeline_global_names
from orchestrator.validation.template_validator import TemplateValidator

pytestmark = [pytest.mark.contract]

REPO = Path(__file__).parent.parent


def _cli(command, pipeline, cwd):
    """Run `orchestrator {command} {pipeline}` the way a user would."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("ANTHROPIC_API_KEY", None)
    env["ORCHESTRATOR_AUTO_INSTALL"] = "0"
    return subprocess.run(
        [sys.executable, "-m", "orchestrator.cli", command, str(pipeline)],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _environments():
    """Every environment a pipeline template can be rendered by."""
    return {
        "runtime": TemplateManager().env,
        "compiler": YAMLCompiler().template_engine,
        "validator": TemplateValidator().env,
    }


def _outcome(env, expression, context):
    """('value', rendered) or ('raised', ExceptionName) -- comparable either way."""
    try:
        return ("value", env.from_string(expression).render(**context))
    except Exception as exc:  # noqa: BLE001 - the class is the observation
        return ("raised", type(exc).__name__)


#: (id, expression, context). Chosen for the places filter implementations
#: usually diverge: the difference between undefined and falsy, None handling,
#: non-ASCII text, and malformed input.
CASES = [
    # `default` must distinguish "not defined" from "defined and falsy". This
    # is the whole point of the filter and where the compiler's copy was wrong.
    ("default_undefined", "{{ missing | default('X') }}", {}),
    ("default_zero", "{{ v | default('X') }}", {"v": 0}),
    ("default_empty_string", "{{ v | default('X') }}", {"v": ""}),
    ("default_false", "{{ v | default('X') }}", {"v": False}),
    ("default_none", "{{ v | default('X') }}", {"v": None}),
    ("default_empty_list", "{{ v | default('X') }}", {"v": []}),
    ("default_present", "{{ v | default('X') }}", {"v": "real"}),
    # `default(..., true)` is the opt-in falsy form and must stay distinct.
    ("default_boolean_form", "{{ v | default('X', true) }}", {"v": 0}),
    # Case filters on non-ASCII: str.lower() and str.upper() are not
    # interchangeable across scripts.
    ("lower_unicode", "{{ v | lower }}", {"v": "ÄÖÜ Straße ÉCOLE"}),
    ("upper_unicode", "{{ v | upper }}", {"v": "äöü straße école"}),
    ("lower_non_string", "{{ v | lower }}", {"v": 42}),
    # Malformed input: whatever the answer is, it must be the same answer.
    ("from_json_invalid", "{{ v | from_json }}", {"v": "Hello World Report"}),
    ("from_json_valid", '{{ v | from_json }}', {"v": '{"a": 1}'}),
    ("from_json_empty", "{{ v | from_json }}", {"v": ""}),
    ("to_json_unicode", "{{ v | to_json }}", {"v": {"k": "café"}}),
    # Regex: no match, special characters, and a group.
    ("regex_no_match", "{{ v | regex_search('zzz') }}", {"v": "abc"}),
    ("regex_match", "{{ v | regex_search('b.') }}", {"v": "abc"}),
    ("regex_special", "{{ v | regex_search('a.c') }}", {"v": "a.c"}),
    # Path handling.
    ("basename_plain", "{{ '/a/b/c.txt' | basename }}", {}),
    ("basename_trailing_slash", "{{ '/a/b/' | basename }}", {}),
    ("basename_empty", "{{ '' | basename }}", {}),
    # slugify on text that is not already slug-shaped.
    ("slugify_unicode", "{{ v | slugify }}", {"v": "Café Résumé 2024!"}),
    ("slugify_empty", "{{ v | slugify }}", {"v": ""}),
    ("replace_basic", "{{ v | replace('a', 'b') }}", {"v": "banana"}),
]


@pytest.mark.parametrize(
    "expression,context", [(e, c) for _, e, c in CASES], ids=[i for i, _, _ in CASES]
)
def test_every_environment_renders_the_same_result(expression, context):
    """Same expression, same input, same answer -- or the same failure."""
    outcomes = {
        name: _outcome(env, expression, context)
        for name, env in _environments().items()
    }

    distinct = set(outcomes.values())
    assert len(distinct) == 1, (
        f"{expression!r} with {context!r} means different things depending on "
        f"which environment renders it: "
        + ", ".join(f"{n}={o!r}" for n, o in outcomes.items())
    )


def test_the_environments_offer_the_same_filters():
    """Names, as well as behaviour. Both are required; neither is sufficient."""
    runtime = set(TemplateManager().env.filters)
    drifted = {
        name: sorted(runtime.symmetric_difference(set(env.filters)))
        for name, env in _environments().items()
        if set(env.filters) != runtime
    }
    assert not drifted, f"filter registries differ from the runtime: {drifted}"


#: Probe inputs for the sweep below. Deliberately includes the values that
#: separate "undefined" from "falsy", plus non-ASCII and a non-string.
PROBES = [{"v": v} for v in ("text", "", "Café Résumé", 0, False, None, 42, ["a"])]

#: Filters whose output is not a function of their input, so "the same answer
#: everywhere" is not a property they can have. Jinja's own, not ours.
NONDETERMINISTIC_FILTERS = frozenset({"random", "shuffle"})

_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


def _comparable(outcome):
    """Rendered output with object addresses masked.

    Lazy filters render as `<generator object do_items at 0x10b0e2190>`. The
    address differs between any two renders, including two of the same
    environment, so comparing it raw reports every lazy filter as divergent.
    """
    kind, payload = outcome
    return (kind, _ADDRESS.sub("0xADDR", payload) if kind == "value" else payload)


def test_no_filter_diverges_between_environments():
    """Every shared filter, not only the ones this file thought to name.

    The cases above were chosen by guessing where implementations diverge, and
    that is exactly how the original name-only check missed the problem: it
    tested what I remembered to test. This sweeps the whole registry, so a
    filter re-registered with different behaviour is caught whether or not
    anyone anticipated it.

    A filter that raises for a probe is fine -- it only has to raise the same
    way everywhere.
    """
    environments = _environments()
    runtime = environments["runtime"]

    divergent = {}
    for filter_name in sorted(runtime.filters):
        if filter_name in NONDETERMINISTIC_FILTERS:
            continue
        expression = "{{ v | " + filter_name + " }}"
        for probe in PROBES:
            outcomes = {
                name: _comparable(_outcome(env, expression, probe))
                for name, env in environments.items()
            }
            if len(set(outcomes.values())) > 1:
                divergent.setdefault(filter_name, []).append((probe["v"], outcomes))

    assert not divergent, (
        "these filters behave differently depending on which environment "
        "renders them: "
        + "; ".join(
            f"{name} on {cases[0][0]!r} -> "
            + ", ".join(f"{n}={o!r}" for n, o in cases[0][1].items())
            for name, cases in divergent.items()
        )
    )


# ---------------------------------------------------------------------------
# End to end: what a real run actually writes to a file
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.parametrize(
    "expression,expected",
    [
        ("{{ zero | default('MISSING') }}", "0"),
        ("{{ blank | default('MISSING') }}", ""),
        ("{{ absent | default('MISSING') }}", "MISSING"),
    ],
    ids=["zero_is_not_missing", "empty_is_not_missing", "absent_is_missing"],
)
def test_the_cli_writes_what_the_runtime_renders(expression, expected, tmp_path):
    """The environments agreeing is only worth something if the file agrees."""
    pipeline = tmp_path / "p.yaml"
    pipeline.write_text(
        f"""
id: conformance
name: Conformance
parameters:
  zero:
    type: integer
    default: 0
  blank:
    type: string
    default: ""
steps:
  - id: write_it
    tool: filesystem
    action: write
    parameters:
      path: "./out.txt"
      content: "{expression}"
"""
    )

    result = _cli("run", pipeline, tmp_path)

    assert result.returncode == 0, (
        f"{expression} did not run: {result.stdout[-800:]}{result.stderr[-800:]}"
    )
    assert (tmp_path / "out.txt").read_text() == expected


# ---------------------------------------------------------------------------
# Globals: shared names, run-time-only evaluation
# ---------------------------------------------------------------------------

#: One call per pipeline global. The arguments only have to be plausible --
#: these cases test that the *name* is part of the language everywhere, not
#: what the call returns.
GLOBAL_CALLS = {
    "now": "now()",
    "file_exists": "file_exists('./x.txt')",
    "include_file": "include_file('./x.txt')",
    "loop_var": "loop_var('l', 'item')",
    "loop_item_at": "loop_item_at('l', 0)",
    "current_loop_name": "current_loop_name()",
    "active_loops": "active_loops()",
    "historical_loops": "historical_loops()",
}


def test_every_pipeline_global_has_a_case_here():
    """A ninth global must arrive with a case, or the sweep below skips it.

    `pipeline_global_names` is derived from what the runtime registers, so it
    picks up a new global automatically. This table does not, and a silently
    untested global is how `now()` reached a release rejecting its own
    pipelines.
    """
    assert set(GLOBAL_CALLS) == pipeline_global_names(), (
        f"GLOBAL_CALLS does not match the language: "
        f"missing={sorted(pipeline_global_names() - set(GLOBAL_CALLS))}, "
        f"stale={sorted(set(GLOBAL_CALLS) - pipeline_global_names())}"
    )


@pytest.mark.parametrize("call", sorted(GLOBAL_CALLS.values()))
def test_a_pipeline_calling_a_global_validates(call):
    """`{{ now() }}` runs correctly, so it must not fail validation.

    It used to draw four errors from two validators: `Undefined variable:
    'now'` from the template validator, and -- because splitting on `.` leaves
    the call syntax attached -- `Undefined task reference: 'now()'` from the
    data-flow validator, which looked the name up as a task id.
    """
    # Note the braces: an earlier version interpolated the bare call, so the
    # pipeline carried the literal string `now()` and validated for the one
    # reason the test was not looking for.
    expression = "{{ " + call + " }}"
    pipeline = f"""
id: globals_probe
name: Globals Probe
steps:
  - id: write_it
    tool: filesystem
    action: write
    parameters:
      path: "./out.txt"
      content: "{expression}"
"""
    import asyncio

    try:
        asyncio.run(YAMLCompiler().compile(pipeline, {}))
    except Exception as exc:  # noqa: BLE001 - the message is the subject
        pytest.fail(f"{call} does not validate, but the runtime renders it: {exc}")


def test_globals_are_not_evaluated_outside_the_runtime():
    """The names are shared; the implementations deliberately are not.

    Filters transform a value the caller already holds, so copying them into
    every environment is right. Globals answer questions about the state of a
    run, and the obvious symmetry -- copy the globals too -- makes the compiler
    answer them at compile time, before any step has run. Leaving them
    unregistered is what makes the compiler keep the template for the runtime.
    """
    leaked = {
        name: sorted(pipeline_global_names() & set(env.globals))
        for name, env in _environments().items()
        if name != "runtime" and pipeline_global_names() & set(env.globals)
    }
    assert not leaked, (
        f"these environments can evaluate run-time globals at compile or "
        f"validation time, which answers them wrongly rather than deferring: "
        f"{leaked}"
    )


def test_the_runtime_can_evaluate_every_global():
    """The other half: unregistered everywhere would be no language at all."""
    missing = sorted(pipeline_global_names() - set(TemplateManager().env.globals))
    assert not missing, f"the runtime cannot evaluate {missing}"


def test_pipeline_global_names_excludes_jinja_s_own():
    """`range` and `dict` are Jinja's, not ours.

    They must keep working in every environment, so they must not be treated as
    run-time-only names.
    """
    assert "range" not in pipeline_global_names()
    assert "dict" not in pipeline_global_names()
    for name, env in _environments().items():
        assert env.from_string("{{ range(3) | list }}").render() == "[0, 1, 2]", (
            f"{name} lost Jinja's own globals"
        )


@pytest.mark.e2e
def test_a_global_is_answered_after_earlier_steps_have_run(tmp_path):
    """The decisive case: `file_exists` on a file an earlier step writes.

    At run time the answer is True, because `make_it` has already written it.
    Evaluated at compile time -- which is what copying the globals into the
    compiler's environment does -- the answer is False, and the pipeline
    silently writes the wrong thing. This is the test that says which.
    """
    pipeline = tmp_path / "p.yaml"
    pipeline.write_text(
        """
id: ordering
name: Ordering
steps:
  - id: make_it
    tool: filesystem
    action: write
    parameters:
      path: "artifact"
      content: "hello"

  - id: check_it
    tool: filesystem
    action: write
    parameters:
      path: "./b.txt"
      content: "exists={{ file_exists('artifact') }}"
    dependencies:
      - make_it
"""
    )

    result = _cli("run", pipeline, tmp_path)
    assert result.returncode == 0, f"{result.stdout[-800:]}{result.stderr[-800:]}"
    assert (tmp_path / "b.txt").read_text() == "exists=True", (
        "file_exists was answered before make_it ran, so it was evaluated at "
        "compile time rather than at run time"
    )


@pytest.mark.e2e
def test_validate_and_run_agree_about_globals(tmp_path):
    """A pipeline the runtime executes must not be refused by `validate`."""
    pipeline = tmp_path / "p.yaml"
    pipeline.write_text(
        """
id: now_probe
name: Now Probe
steps:
  - id: write_it
    tool: filesystem
    action: write
    parameters:
      path: "./out.txt"
      content: "generated {{ now() }}"
"""
    )

    validated = _cli("validate", pipeline, tmp_path)
    ran = _cli("run", pipeline, tmp_path)

    assert ran.returncode == 0, f"run failed: {ran.stdout[-800:]}{ran.stderr[-800:]}"
    assert (tmp_path / "out.txt").read_text().startswith("generated 20")
    assert validated.returncode == 0, (
        f"`run` executes this pipeline but `validate` rejects it: "
        f"{validated.stdout[-800:]}{validated.stderr[-800:]}"
    )
