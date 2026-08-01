"""A template reference that never resolved must not reach a tool.

Template rendering deliberately falls back to returning the original text when
a reference is undefined, because resolution runs in several passes and a
reference that cannot resolve yet may resolve later. The failure mode (#153) is
that the fallback survived all the way to the tool: the literal
`{{ step.field }}` was written to a file and the step reported `success: true`.

Measured before the fix, on the pipeline below:

    $ orchestrator run bad_template.yaml
    "success": true, "error": null
    $ cat out/bad.txt
    value = {{ nonexistent_step.result.field }}

The complementary case -- a template that *does* resolve must not be written
through literally -- is
tests/test_golden_pipelines.py::test_unresolved_template_does_not_reach_output.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.core.exceptions import UnresolvedTemplateError
from orchestrator.core.unified_template_resolver import (
    UnifiedTemplateResolver,
    _find_template_markers,
)

pytestmark = [pytest.mark.contract, pytest.mark.e2e]

# Exit codes, from docs/adr/0001-product-contract.md.
EXIT_EXECUTION_ERROR = 1

UNRESOLVED_PIPELINE = """
id: unresolved_template_pipeline
name: Unresolved Template Pipeline
steps:
  - id: write_it
    tool: filesystem
    action: write
    parameters:
      path: "out/bad.txt"
      content: "value = {{ nonexistent_step.result.field }}"
"""


def _run_cli(args, cwd):
    """Invoke the CLI as a subprocess, the way a user would."""
    env = dict(os.environ)
    src = str(Path(__file__).parent.parent / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("ANTHROPIC_API_KEY", None)
    env["ORCHESTRATOR_AUTO_INSTALL"] = "0"
    return subprocess.run(
        [sys.executable, "-m", "orchestrator.cli", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_an_unresolved_reference_fails_the_run_and_writes_nothing(tmp_path):
    """The whole point: no artifact, non-zero exit, and the reason is named."""
    pipeline = tmp_path / "bad_template.yaml"
    pipeline.write_text(UNRESOLVED_PIPELINE)

    result = _run_cli(["run", str(pipeline)], cwd=tmp_path)

    assert result.returncode == EXIT_EXECUTION_ERROR, (
        f"expected exit {EXIT_EXECUTION_ERROR}, got {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert not (tmp_path / "out" / "bad.txt").exists(), (
        "the step was stopped, so it must not have written its output file"
    )

    combined = result.stdout + result.stderr
    assert "nonexistent_step" in combined, (
        f"the error must name the reference that failed:\n{combined[-2000:]}"
    )


# ---------------------------------------------------------------------------
# the check itself
# ---------------------------------------------------------------------------

@pytest.fixture
def resolver():
    return UnifiedTemplateResolver()


def _resolve(resolver, parameters):
    context = resolver.collect_context(pipeline_id="p", task_id="t")
    return resolver.resolve_before_tool_execution("filesystem", parameters, context)


def test_a_resolvable_template_still_passes(resolver):
    """No false positives: a reference that resolves must go straight through."""
    context = resolver.collect_context(
        pipeline_id="p", task_id="t", pipeline_inputs={"name": "world"}
    )
    resolved = resolver.resolve_before_tool_execution(
        "filesystem", {"content": "hello {{ name }}"}, context
    )

    assert resolved["content"] == "hello world"


def test_an_unresolved_reference_raises_naming_the_parameter(resolver):
    with pytest.raises(UnresolvedTemplateError) as excinfo:
        _resolve(resolver, {"content": "value = {{ missing.field }}"})

    error = excinfo.value
    assert error.details["parameter"] == "content"
    assert error.details["tool"] == "filesystem"
    assert error.details["unresolved"] == ["{{ missing.field }}"]
    # The message has to be usable without opening a debugger.
    assert "content" in str(error) and "missing.field" in str(error)


def test_a_surviving_control_block_is_caught_too(resolver):
    """`{% if %}` reaching a tool is the same defect wearing different syntax."""
    with pytest.raises(UnresolvedTemplateError):
        _resolve(resolver, {"content": "{% if missing %}yes{% endif %}"})


def test_a_reference_buried_in_a_container_is_caught(resolver):
    """Parameters are routinely lists of paths or dicts of fields.

    An unresolved reference nested inside one reaches the tool exactly as
    readily as a top-level string, so scanning only top-level strings would
    leave the hole open.
    """
    with pytest.raises(UnresolvedTemplateError) as excinfo:
        _resolve(resolver, {"paths": ["ok.txt", {"nested": "{{ missing }}"}]})

    assert excinfo.value.details["parameter"] == "paths"


def test_internal_plumbing_parameters_are_not_inspected(resolver):
    """The control systems attach `_`-prefixed objects on the way to a tool.

    Those are machinery, not user-authored values, and one of them holding a
    template-looking string must not fail the step.
    """
    resolved = _resolve(
        resolver, {"content": "fine", "_internal_note": "{{ not_user_authored }}"}
    )

    assert resolved["_internal_note"] == "{{ not_user_authored }}"


def test_escaped_delimiters_are_not_a_false_positive(resolver):
    """A pipeline may legitimately emit literal Jinja, e.g. documentation.

    Jinja's own escape renders to the delimiters as text, and by then there is
    no marker left to find -- so this must pass, or writing a tutorial about
    templates would be impossible.
    """
    resolved = _resolve(resolver, {"content": "write {{ '{{' }} name {{ '}}' }} here"})

    assert resolved["content"] == "write {{ name }} here"


def test_a_resolved_value_containing_template_text_is_content_not_a_failure(resolver):
    """The test is survival, not presence.

    A step whose *result* contains something shaped like a template -- a
    scraped page, a code sample, a prompt about Jinja -- has resolved
    perfectly well. Flagging it would mean the pipeline can never carry text
    about templates through a tool.
    """
    context = resolver.collect_context(
        pipeline_id="p",
        task_id="t",
        pipeline_inputs={"snippet": "use {{ user.name }} to interpolate"},
    )
    resolved = resolver.resolve_before_tool_execution(
        "filesystem", {"content": "{{ snippet }}"}, context
    )

    assert resolved["content"] == "use {{ user.name }} to interpolate"


def test_one_bad_reference_takes_the_whole_string_down(resolver):
    """Rendering is all-or-nothing per string, and the report says so.

    `StrictUndefined` raises on the first undefined reference, and the fallback
    returns the *entire* original string -- so in `"{{ here }} but not
    {{ missing }}"` the resolvable `{{ here }}` does not survive either. Both
    markers are reported, which is the truthful account: nothing in this string
    was rendered. Reporting only `{{ missing }}` would imply `{{ here }}` had
    been substituted, and it had not.
    """
    context = resolver.collect_context(
        pipeline_id="p", task_id="t", pipeline_inputs={"here": "ok"}
    )

    with pytest.raises(UnresolvedTemplateError) as excinfo:
        resolver.resolve_before_tool_execution(
            "filesystem", {"content": "{{ here }} but not {{ missing }}"}, context
        )

    assert excinfo.value.details["unresolved"] == ["{{ here }}", "{{ missing }}"]


# ---------------------------------------------------------------------------
# the scanner
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        ("plain text", []),
        ("{{ a }}", ["{{ a }}"]),
        ("{{ a }} and {{ b }}", ["{{ a }}", "{{ b }}"]),
        ("{% for x in y %}", ["{% for x in y %}"]),
        ({"k": "{{ a }}"}, ["{{ a }}"]),
        ({"{{ key }}": "v"}, ["{{ key }}"]),
        ([["{{ deep }}"]], ["{{ deep }}"]),
        (("{{ t }}",), ["{{ t }}"]),
        (42, []),
        (None, []),
        ("", []),
    ],
)
def test_find_template_markers(value, expected):
    assert _find_template_markers(value) == expected


def test_find_template_markers_spans_newlines():
    """A multi-line block is still one marker, not two halves of nothing."""
    assert _find_template_markers("{{ a\n  + b }}") == ["{{ a\n  + b }}"]
