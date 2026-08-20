"""Validation findings reach a Python caller, not only the terminal.

`orchestrator validate --json` has emitted structured findings since #467. The
Python API had `PipelineAPI.validate_yaml`, which returns a bare `bool`: a
caller embedding the orchestrator could learn *that* a document was rejected
and nothing about why, and could not see warnings at all -- which is where
"this reference could not be checked" lives, the warning that precedes the
run-time failure in #465.

The findings existed; they were private to `cli.py`. So the risk in exposing
them was building a *second* implementation that drifts from the one the CLI
prints, which is the class of bug #466 removed for dependencies. There is one
implementation, and `test_the_cli_and_the_api_report_the_same_findings` is
what holds it to that.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.validation.pipeline_report import (
    Finding,
    validate_pipeline_file,
    validate_pipeline_text,
)

pytestmark = [pytest.mark.contract]

REPO = Path(__file__).resolve().parent.parent
SUPPORTED = REPO / "examples" / "supported" / "01_hello_filesystem.yaml"

#: A document that validates *and* carries findings. Every assertion about
#: findings needs one: the hermetic example above produces none, so a suite
#: written only against it passes just as happily when warnings are dropped
#: on the floor -- which is what the first version of this file did.
WITH_FINDINGS = REPO / "examples" / "auto_tags_demo.yaml"

#: The payload's field names are interface: a consumer must not have to parse
#: the human message to learn which step or reference is at fault. Asserted as
#: an exact set so removing or renaming one is a deliberate act.
FINDING_FIELDS = {
    "code", "severity", "category", "step", "parameter_path",
    "referenced_step", "referenced_field", "message", "suggestions",
}


def test_a_valid_pipeline_reports_its_graph():
    result = validate_pipeline_file(SUPPORTED)
    assert result.valid, result.error
    assert result.pipeline_id == "hello_filesystem"
    assert result.tasks["read_back"] == ["write_greeting"], result.tasks
    assert result.error is None


def test_a_broken_pipeline_is_a_result_not_an_exception():
    """A caller validating user input wants the findings alongside the
    failure. Raising would discard them at the moment they are most useful."""
    import asyncio

    result = asyncio.run(validate_pipeline_text("id: broken\nsteps: [[[["))
    assert not result.valid
    assert result.error, "the failure must say what happened"


def test_warnings_reach_a_python_caller():
    """The reported gap. `validate_yaml` returns a bool, so a warning saying
    a reference could not be checked -- the one that precedes the run-time
    failure in #465 -- was invisible to anything but the terminal."""
    result = validate_pipeline_file(WITH_FINDINGS)
    assert result.valid, result.error
    assert result.warnings, "this document warns; the API reported nothing"
    assert any(
        "will be resolved at runtime" in w.message for w in result.warnings
    ), [w.message for w in result.warnings]


def test_a_warning_does_not_make_a_document_invalid():
    """Warnings inform; they do not block. Conflating them would reject
    documents that run."""
    assert validate_pipeline_file(WITH_FINDINGS).valid


def test_findings_carry_stable_named_fields():
    result = validate_pipeline_file(WITH_FINDINGS)
    assert result.findings, "nothing to check the field names of"
    for finding in result.findings:
        assert set(finding.as_dict()) == FINDING_FIELDS, finding.as_dict()


def test_errors_and_warnings_are_separable():
    """A caller that wants to block on errors and log warnings should not have
    to string-match a severity out of a message."""
    findings = (
        Finding(code="a", severity="error", message="x"),
        Finding(code="b", severity="warning", message="y"),
    )
    from orchestrator.validation.pipeline_report import PipelineValidation

    result = PipelineValidation(valid=False, findings=findings)
    assert [f.code for f in result.errors] == ["a"]
    assert [f.code for f in result.warnings] == ["b"]
    assert result.errors[0].is_error and result.warnings[0].is_warning


def test_the_same_document_yields_the_same_findings_in_the_same_order():
    """Order is part of the contract: a caller diffing two runs, or a test
    pinning output, cannot use a set that reshuffles."""
    first = validate_pipeline_file(WITH_FINDINGS)
    second = validate_pipeline_file(WITH_FINDINGS)
    assert first.findings, "an empty list is trivially stable"
    assert [f.as_dict() for f in first.findings] == [
        f.as_dict() for f in second.findings
    ]


@pytest.mark.e2e
def test_findings_are_stable_across_processes():
    """The check above cannot catch this on its own.

    Findings were emitted while iterating sets, and `PYTHONHASHSEED` is fixed
    for the life of a process -- so two calls in one interpreter agreed while
    two `orchestrator validate` runs produced the same 44 findings in
    different orders, and offered the same three "did you mean" suggestions
    permuted. On a longer candidate list, truncating to three would have
    offered *different* suggestions run to run.
    """
    digests = {_cli_findings_digest() for _ in range(3)}
    assert len(digests) == 1, "validate is not reproducible run to run"


def _cli_findings_digest() -> str:
    import hashlib

    proc = subprocess.run(
        [sys.executable, "-m", "orchestrator.cli", "validate", "--json", str(WITH_FINDINGS)],
        cwd=str(REPO), env=_cli_env(), capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-800:] + proc.stderr[-800:]
    findings = json.loads(proc.stdout)["findings"]
    assert findings, "nothing to be stable about"
    return hashlib.sha256(json.dumps(findings).encode()).hexdigest()


def _cli_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["ORCHESTRATOR_AUTO_INSTALL"] = "0"
    return env


@pytest.mark.e2e
def test_the_cli_and_the_api_report_the_same_findings():
    """The guard against a second implementation.

    The CLI's findings were private helpers in `cli.py`; exposing them to
    Python by reimplementing would have produced two things to drift apart.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "orchestrator.cli", "validate", "--json", str(WITH_FINDINGS)],
        cwd=str(REPO), env=_cli_env(), capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-800:] + proc.stderr[-800:]
    from_cli = json.loads(proc.stdout)

    from_api = validate_pipeline_file(WITH_FINDINGS)
    assert from_api.findings, "comparing two empty lists proves nothing"
    assert from_cli["valid"] == from_api.valid
    assert from_cli["pipeline"] == from_api.pipeline_id
    assert from_cli["tasks"] == from_api.tasks
    assert from_cli["findings"] == [f.as_dict() for f in from_api.findings]


def test_the_package_exports_them():
    """A caller should not have to know which module they live in."""
    import orchestrator

    assert orchestrator.validate_pipeline_file is validate_pipeline_file
    assert orchestrator.Finding is Finding
