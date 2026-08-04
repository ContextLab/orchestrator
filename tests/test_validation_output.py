"""A warning nobody can see is not a warning.

The compiler has always collected findings into a `ValidationReport`. Nothing
displayed them on a successful run, so `orchestrator validate` printed::

    ✓ p.yaml is valid

for a pipeline carrying a warning that a reference could not be checked -- and
that pipeline then failed at run time on exactly that reference (#465). The
warning existed the whole time, in the log stream, which `validate` does not
write to and a script capturing stdout never sees.

These tests pin both channels: the human one, and a structured one whose field
names are an interface rather than English prose to be parsed.
"""

import ast
import builtins
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.contract]

REPO = Path(__file__).resolve().parent.parent

#: A step whose output shape is unknown to the validator, referenced by a
#: field name it therefore cannot check. Warns; does not fail.
WARNING_PIPELINE = """
id: uo
name: UO
steps:
  - id: make
    tool: filesystem
    action: write
    parameters:
      path: "./out_a.txt"
      content: "A ran"
  - id: use
    tool: filesystem
    action: write
    parameters:
      path: "./out_b.txt"
      content: "made={{ make.path }}"
"""

CLEAN_PIPELINE = """
id: clean
name: Clean
steps:
  - id: only
    tool: filesystem
    action: write
    parameters:
      path: "./out.txt"
      content: "no references at all"
"""


def _cli(pipeline_text, tmp_path, *args):
    pipeline = tmp_path / "p.yaml"
    pipeline.write_text(pipeline_text)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["ORCHESTRATOR_AUTO_INSTALL"] = "0"
    return subprocess.run(
        [sys.executable, "-m", "orchestrator.cli", "validate", str(pipeline), *args],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300,
    )


# ---------------------------------------------------------------------------
# The human channel
# ---------------------------------------------------------------------------

@pytest.mark.e2e
def test_a_warning_appears_on_stdout(tmp_path):
    """stdout, specifically. The warning was always in the log stream, and
    `validate` does not write there."""
    result = _cli(WARNING_PIPELINE, tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "does not declare its outputs" in result.stdout, (
        f"the warning never reached stdout:\n{result.stdout}"
    )


@pytest.mark.e2e
def test_the_summary_line_counts_the_warnings(tmp_path):
    """`✓ is valid` alone is what let a warning pass unnoticed."""
    result = _cli(WARNING_PIPELINE, tmp_path)
    first = result.stdout.splitlines()[0]
    assert "warning" in first, f"the summary hides the warnings: {first!r}"


@pytest.mark.e2e
def test_a_finding_names_its_code_and_location(tmp_path):
    """A code is what makes a finding searchable and suppressible later."""
    result = _cli(WARNING_PIPELINE, tmp_path)
    assert "data_flow_undefined_output" in result.stdout


@pytest.mark.e2e
def test_a_clean_pipeline_says_nothing_extra(tmp_path):
    """Reporting must not become noise, or it stops being read."""
    result = _cli(CLEAN_PIPELINE, tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "warning" not in result.stdout.lower(), result.stdout
    assert result.stdout.splitlines()[0].endswith("is valid")


@pytest.mark.e2e
def test_warnings_do_not_fail_validation(tmp_path):
    """A warning is a warning. Making these errors would reject pipelines that
    run correctly -- the false-positive class removed in #448/#450/#461."""
    assert _cli(WARNING_PIPELINE, tmp_path).returncode == 0


# ---------------------------------------------------------------------------
# The structured channel
# ---------------------------------------------------------------------------

@pytest.mark.e2e
def test_json_output_is_only_json(tmp_path):
    """Anything appended after the document makes it unparseable -- which is
    exactly what the catalogue report did before it was caught."""
    result = _cli(WARNING_PIPELINE, tmp_path, "--json")
    payload = json.loads(result.stdout)  # must not raise
    assert payload["valid"] is True


@pytest.mark.e2e
def test_json_findings_carry_stable_fields(tmp_path):
    """The field names are the interface. A consumer must not have to read the
    English message to learn which step and which reference are at fault."""
    result = _cli(WARNING_PIPELINE, tmp_path, "--json")
    findings = json.loads(result.stdout)["findings"]

    finding = next(f for f in findings if f["code"] == "data_flow_undefined_output")
    assert finding["severity"] == "warning"
    assert finding["step"] == "use"
    assert finding["referenced_step"] == "make"
    assert finding["referenced_field"] == "path"
    assert finding["parameter_path"]


@pytest.mark.e2e
def test_json_reports_the_task_graph(tmp_path):
    result = _cli(WARNING_PIPELINE, tmp_path, "--json")
    assert set(json.loads(result.stdout)["tasks"]) == {"make", "use"}


@pytest.mark.e2e
def test_json_is_emitted_for_an_invalid_pipeline_too(tmp_path):
    """A consumer should not have to switch parsers depending on the outcome."""
    broken = WARNING_PIPELINE.replace("{{ make.path }}", "{{ ghost.path }}")
    result = _cli(broken, tmp_path, "--json")

    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is False
    assert "ghost" in payload["error"]


# ---------------------------------------------------------------------------
# The shadowed builtin that hid all of it
# ---------------------------------------------------------------------------

def test_the_cli_module_shadows_no_builtin_at_module_scope():
    """`def list()` at module scope rebinds the builtin for the whole file.

    A click Command is callable, so a later `list(...)` did not build a list --
    it *invoked the command*, printed the configured providers and exited
    before its own output. `--json` emitted nothing at all and looked like a
    click registration problem.

    Anything named after a builtin here is one call away from repeating that.
    """
    module = ast.parse((REPO / "src" / "orchestrator" / "cli.py").read_text())
    shadowed = sorted({
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and hasattr(builtins, node.name)
    })
    assert not shadowed, (
        f"these module-level names shadow builtins in cli.py: {shadowed}. "
        f"Give the function its own name and pass the CLI name to the "
        f"decorator, e.g. @keys.command(\"list\")."
    )


@pytest.mark.e2e
def test_keys_list_is_still_reachable_under_its_cli_name(tmp_path):
    """Renaming the function must not rename the command."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "orchestrator.cli", "keys", "list"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "provider" in result.stdout.lower(), result.stdout
