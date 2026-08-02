"""Template rendering is sandboxed: parameter values are data, not code.

A pipeline's `{{ }}` expressions are written by the pipeline author, but the
*values* substituted into them are not: they arrive from `-i name=value`, from
an inputs file, or from an earlier step's output. Those values are rendered
too, so a value that itself contains `{{ }}` gets evaluated.

That much is a feature -- `-i out_dir='{{ base }}/reports'` is useful. What is
not a feature is the evaluation being unrestricted. Jinja's default
`Environment` exposes Python's object graph, so a parameter value could reach
`''.__class__.__mro__[1].__subclasses__()` and walk from a string literal to
arbitrary types. That is the standard Jinja SSTI-to-RCE escalation, and it was
reachable from a plain `orchestrator run ... -i greeting=<payload>`:

    -i greeting='{{ "".__class__.__mro__[1].__subclasses__() | length }}'
    -> output/greeting.txt == "1183, world"

These tests pin the boundary. Ordinary expressions still evaluate; anything
that reaches for Python internals is refused rather than rendered.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.core.template_manager import TemplateManager

pytestmark = [pytest.mark.contract]

REPO = Path(__file__).parent.parent
EXAMPLE = REPO / "examples" / "supported" / "01_hello_filesystem.yaml"

#: Payloads that must never evaluate. Each one is the first hop of a documented
#: Jinja sandbox escape -- reaching any of them means the object graph is open.
ESCAPES = [
    '{{ "".__class__ }}',
    '{{ "".__class__.__mro__[1].__subclasses__() | length }}',
    '{{ [].__class__.__base__.__subclasses__() | length }}',
    '{{ "".__class__.__init__.__globals__ }}',
    '{{ self.__init__.__globals__ }}',
    '{{ cycler.__init__.__globals__.os.popen("id").read() }}',
]


def _run_cli(args, cwd):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
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


@pytest.mark.parametrize("payload", ESCAPES, ids=lambda p: p[:34])
def test_a_parameter_value_cannot_reach_python_internals(payload):
    """The unit-level boundary: rendering refuses, it does not compute."""
    manager = TemplateManager()

    with pytest.raises(Exception) as excinfo:
        manager.render(payload, {})

    # A SecurityError, not an incidental AttributeError that happens to abort.
    assert "SecurityError" in type(excinfo.value).__name__ or "unsafe" in str(
        excinfo.value
    ).lower(), (
        f"{payload!r} failed, but not because the sandbox refused it: "
        f"{type(excinfo.value).__name__}: {excinfo.value}"
    )


@pytest.mark.e2e
def test_an_escape_payload_never_reaches_an_artifact(tmp_path):
    """End to end, through the CLI, exactly as the vulnerability was found.

    Before the sandbox this wrote `1183, world` and exited 0.
    """
    payload = '{{ "".__class__.__mro__[1].__subclasses__() | length }}'

    result = _run_cli(["run", str(EXAMPLE), "-i", f"greeting={payload}"], cwd=tmp_path)

    artifact = tmp_path / "output" / "greeting.txt"
    if artifact.exists():
        content = artifact.read_text()
        assert "__" not in content and not content.split(",")[0].strip().isdigit(), (
            f"a template-injection payload was evaluated into an artifact: "
            f"{content!r}"
        )
    assert result.returncode != 0, (
        "an injection payload was accepted and the run reported success"
    )


def test_ordinary_expressions_still_evaluate():
    """The sandbox must not cost us the feature it protects.

    Parameter values holding templates are legitimate; only the object graph
    is off limits.
    """
    manager = TemplateManager()

    assert manager.render("{{ 7 * 7 }}", {}) == "49"
    assert manager.render("{{ name | upper }}", {"name": "ada"}) == "ADA"
    assert manager.render("{{ items | length }}", {"items": [1, 2, 3]}) == "3"


def test_step_result_references_still_resolve():
    """The shape every supported example depends on: `{{ id.result.field }}`."""
    manager = TemplateManager()
    context = {"read_header": {"result": {"content": "# Title\n"}}}

    assert manager.render("{{ read_header.result.content }}", context) == "# Title\n"
