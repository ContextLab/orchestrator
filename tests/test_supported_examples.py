"""Every pipeline in examples/supported/ is executable product behaviour.

`examples/` held 111 pipelines of which 3 compiled. They were the project's
primary documentation surface and almost none of them worked, so a reader had
no way to tell which parts of the syntax were real.

`examples/supported/` is the answer: a small set that is *tested as product
behaviour*. Every file here must validate, must execute through both the CLI
and the Python API, and the two must agree as whole normalised documents.

What "tested" means here is deliberately strict, because the first version of
this file was not. It asserted that a command exited and that some files
existed, which a badly broken runtime also satisfies -- and it did: it pinned
04_conditions while that example ran *both* arms of its conditional, and it
compared the CLI against the API after deleting every step value and the whole
outputs document, so the two surfaces could have disagreed about everything
that matters and still passed.

So a case here declares, exactly:

* the exit code,
* every artifact the run may create *and its full contents* -- an unexpected
  file is a failure, not a shrug,
* the complete declared `outputs` document,
* which steps completed, which were skipped, and which failed,
* the execution levels, so "these steps run in parallel" is a checked claim,
* per-step fields where a step's *manner* of failing is the point
  (timeout classification, retry count).

A pipeline with a branch declares a case per branch, so the arm a default run
does not take is still exercised.

Adding a file to examples/supported/ without a case fails, so the set cannot
grow past its coverage.
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest

from orchestrator.core.pipeline_result import normalize_result_payload
from tests.test_infrastructure import create_test_orchestrator

pytestmark = [
    pytest.mark.contract,
    pytest.mark.e2e,
    # An abandoned subprocess or an unclosed transport does not fail anything;
    # it surfaces at teardown as an unraisable exception, which scrolls past.
    # This suite runs real pipelines with real side effects, so it is exactly
    # where that class of bug appears, and here it is an error.
    pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning"),
    pytest.mark.filterwarnings("error::ResourceWarning"),
]

SUPPORTED_DIR = Path(__file__).parent.parent / "examples" / "supported"
EXAMPLES = sorted(SUPPORTED_DIR.glob("*.yaml"))

#: Running any pipeline writes a checkpoint under ./checkpoints/. That is a
#: real side effect of a run rather than an artifact of any one example, so it
#: is named here and excluded from the artifact comparison. Naming it is the
#: point: "no unexpected artifacts" then means exactly that, instead of
#: "no artifacts beyond the ones we happened to remember".
SIDE_EFFECT_DIRS = ("checkpoints",)

#: Files the harness itself drops in the run directory.
HARNESS_FILES = ("stdout.json", "stderr.log")


@dataclass(frozen=True)
class Case:
    """One run of one example, and everything observable about it."""

    example: str
    #: Distinguishes branches of the same example in test ids.
    variant: str
    exit_code: int
    #: Relative path -> exact file contents. This is the complete set; any
    #: other file under the run directory fails the case.
    artifacts: Dict[str, str]
    #: The complete declared `outputs` document, compared exactly.
    outputs: Dict[str, Any]
    completed: Tuple[str, ...]
    skipped: Tuple[str, ...] = ()
    failed: Tuple[str, ...] = ()
    #: Pipeline parameters, passed as `-i k=v` to the CLI and as `context=` to
    #: the API -- the two surfaces must reach the same branch the same way.
    inputs: Dict[str, Any] = field(default_factory=dict)
    #: Steps grouped by dependency level. Steps sharing a level may run
    #: concurrently; this is what makes "parallel fan-out" a checked claim.
    levels: Optional[Tuple[Tuple[str, ...], ...]] = None
    #: step id -> {field: expected value} from the step's serialised form.
    step_fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.example}[{self.variant}]"


LONG_TEXT = "a reasonably long piece of text"
HEADER_MD = "# Quarterly Report\n\nBy Orchestrator\n"

CASES = (
    Case(
        example="01_hello_filesystem.yaml",
        variant="default",
        exit_code=0,
        artifacts={"output/greeting.txt": "hello, world"},
        outputs={"greeting_text": "hello, world"},
        completed=("write_greeting", "read_back"),
        levels=(("write_greeting",), ("read_back",)),
    ),
    Case(
        example="02_parallel_fanout_fanin.yaml",
        variant="default",
        exit_code=0,
        artifacts={
            "output/alpha.txt": "alpha-value",
            "output/beta.txt": "beta-value",
            "output/joined.txt": "alpha-value+beta-value",
        },
        outputs={"joined": "./output/joined.txt"},
        completed=("alpha", "beta", "read_alpha", "read_beta", "join"),
        # `alpha` and `beta` share level 0, which is the entire claim the
        # example makes. `join` is alone on the last level, strictly after
        # both reads.
        levels=(
            ("alpha", "beta"),
            ("read_alpha", "read_beta"),
            ("join",),
        ),
    ),
    # 04 declares one case per branch. The default run takes the long arm, so
    # without the second case the short arm would never execute in CI.
    Case(
        example="04_conditions.yaml",
        variant="long-branch",
        exit_code=0,
        artifacts={
            "output/long.txt": f"long: {LONG_TEXT}",
            "output/branch.txt": "long",
        },
        outputs={"taken_branch": "long"},
        completed=("check_size", "handle_long", "summarise"),
        # The short arm must not merely be absent from the artifacts: it must
        # be recorded as skipped. Those are different failures.
        skipped=("handle_short", "mark_short"),
        levels=(
            ("check_size",),
            ("handle_long",),
            ("handle_short",),
            ("mark_short",),
            ("summarise",),
        ),
    ),
    Case(
        example="04_conditions.yaml",
        variant="short-branch",
        exit_code=0,
        inputs={"content": "hi"},
        artifacts={
            "output/short.txt": "short: hi",
            "output/branch.txt": "short",
        },
        outputs={"taken_branch": "short"},
        completed=("handle_short", "mark_short", "summarise"),
        skipped=("check_size", "handle_long"),
        levels=(
            ("check_size",),
            ("handle_long",),
            ("handle_short",),
            ("mark_short",),
            ("summarise",),
        ),
    ),
    Case(
        example="06_failure_policy.yaml",
        variant="default",
        # Not a failure of the example: 06 deliberately contains a step that
        # fails, and the run is supposed to report that honestly.
        exit_code=1,
        artifacts={
            "output/before.txt": "this step succeeds",
            "output/after.txt": "the run continued",
        },
        outputs={},
        completed=("before", "after"),
        failed=("flaky",),
        levels=(("before",), ("flaky",), ("after",)),
        # *How* `flaky` failed is the example's subject. A step that failed
        # for some other reason, or that never retried, would still produce
        # the artifacts above.
        step_fields={
            "flaky": {
                "success": False,
                "error_type": "TimeoutError",
                "timed_out": True,
                # `max_retries: 2` bounds attempts, so two attempts is one
                # retry. See the note in the example.
                "retries": 1,
            }
        },
    ),
    Case(
        example="07_templates_and_outputs.yaml",
        variant="default",
        exit_code=0,
        artifacts={
            "output/header.md": HEADER_MD,
            # Both files keep the trailing newline their `content:` ends with.
            # They did not always: a parameter containing a *step-result*
            # reference was rendered by a Jinja environment left on its
            # default `keep_trailing_newline=False`, so this file -- and only
            # this file, because it is the only one referring to another step
            # -- lost its last byte. Asserting contents exactly is what
            # surfaced it; asserting that the file merely existed did not.
            "output/report.md": HEADER_MD + "\nBody of the report.\n",
        },
        outputs={
            "report_path": "./output/report.md",
            "header_text": HEADER_MD,
        },
        completed=("write_header", "read_header", "write_report"),
        levels=(("write_header",), ("read_header",), ("write_report",)),
    ),
)


def _run_cli(args, cwd):
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(Path(__file__).parent.parent / "src") + os.pathsep + env.get("PYTHONPATH", "")
    )
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


def _cli_inputs(case: Case):
    """`inputs` as repeated `-i key=value` arguments."""
    args = []
    for key, value in case.inputs.items():
        rendered = value if isinstance(value, str) else json.dumps(value)
        args += ["-i", f"{key}={rendered}"]
    return args


def _payload(stdout: str) -> Dict[str, Any]:
    """The result document the CLI prints to stdout."""
    assert stdout.lstrip().startswith("{"), (
        f"expected the CLI to print a JSON document, got: {stdout[:200]!r}"
    )
    return json.loads(stdout)


def _artifacts(run_dir: Path) -> Dict[str, str]:
    """Every file a run left behind, minus known side effects."""
    found = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir)
        if relative.parts[0] in SIDE_EFFECT_DIRS or relative.name in HARNESS_FILES:
            continue
        found[relative.as_posix()] = path.read_text()
    return found


def _assert_case_holds(case: Case, payload: Dict[str, Any], run_dir: Path):
    """Everything a case declares, checked against one run of it."""
    steps = payload["steps"]

    by_status = {}
    for step_id, step in steps.items():
        by_status.setdefault(step["status"], []).append(step_id)

    assert sorted(by_status.get("completed", [])) == sorted(case.completed), (
        f"{case.id}: completed steps were {sorted(by_status.get('completed', []))}, "
        f"expected {sorted(case.completed)}"
    )
    assert sorted(by_status.get("skipped", [])) == sorted(case.skipped), (
        f"{case.id}: skipped steps were {sorted(by_status.get('skipped', []))}, "
        f"expected {sorted(case.skipped)}"
    )
    assert sorted(by_status.get("failed", [])) == sorted(case.failed), (
        f"{case.id}: failed steps were {sorted(by_status.get('failed', []))}, "
        f"expected {sorted(case.failed)}"
    )

    # A step can fail *without raising*, leaving it COMPLETED. Checking status
    # alone would call that a success, so check the flag too.
    for step_id in case.completed:
        assert steps[step_id]["success"] is True, (
            f"{case.id}: step {step_id} completed but reported success=False: "
            f"{steps[step_id].get('error')!r}"
        )
    for step_id in case.failed:
        assert steps[step_id]["success"] is False, (
            f"{case.id}: step {step_id} was expected to fail"
        )

    assert payload["success"] is (case.exit_code == 0), (
        f"{case.id}: success={payload['success']} disagrees with the expected "
        f"exit code {case.exit_code}"
    )

    assert payload["outputs"] == case.outputs, (
        f"{case.id}: declared outputs were {payload['outputs']!r}, "
        f"expected {case.outputs!r}"
    )

    if case.levels is not None:
        actual = tuple(tuple(sorted(level)) for level in payload["execution_levels"])
        expected = tuple(tuple(sorted(level)) for level in case.levels)
        assert actual == expected, (
            f"{case.id}: execution levels were {actual}, expected {expected}"
        )

    for step_id, fields in case.step_fields.items():
        for name, value in fields.items():
            assert steps[step_id][name] == value, (
                f"{case.id}: step {step_id}.{name} was "
                f"{steps[step_id][name]!r}, expected {value!r}"
            )

    found = _artifacts(run_dir)
    assert set(found) == set(case.artifacts), (
        f"{case.id}: artifacts were {sorted(found)}, expected "
        f"{sorted(case.artifacts)}"
    )
    for relative, content in case.artifacts.items():
        assert found[relative] == content, (
            f"{case.id}: {relative} contained {found[relative]!r}, "
            f"expected {content!r}"
        )


def test_the_supported_directory_is_not_empty():
    assert EXAMPLES, f"no examples found in {SUPPORTED_DIR}"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_every_supported_example_has_a_case(example):
    """Coverage cannot lag behind the directory."""
    assert any(case.example == example.name for case in CASES), (
        f"{example.name} is in examples/supported/ but no Case declares what "
        f"it should do, so nothing asserts its behaviour"
    )


def test_every_case_names_a_real_example():
    """And the table cannot outlive the files it describes."""
    names = {example.name for example in EXAMPLES}
    for case in CASES:
        assert case.example in names, (
            f"case {case.id} names {case.example}, which is not in {SUPPORTED_DIR}"
        )


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_every_supported_example_validates(example, tmp_path):
    result = _run_cli(["validate", str(example)], cwd=tmp_path)

    assert result.returncode == 0, (
        f"{example.name} does not compile:\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_the_cli_run_matches_the_case(case, tmp_path):
    example = SUPPORTED_DIR / case.example

    result = _run_cli(["run", str(example), *_cli_inputs(case)], cwd=tmp_path)

    assert result.returncode == case.exit_code, (
        f"{case.id} exited {result.returncode}, expected {case.exit_code}\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    _assert_case_holds(case, _payload(result.stdout), tmp_path)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_the_python_api_run_matches_the_case(case, tmp_path):
    import asyncio

    example = SUPPORTED_DIR / case.example

    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        result = asyncio.run(
            create_test_orchestrator().execute_yaml(
                yaml_content=example.read_text(), context=dict(case.inputs)
            )
        )
    finally:
        os.chdir(previous)

    _assert_case_holds(case, result.to_dict(), tmp_path)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_the_cli_and_the_api_agree(case, tmp_path):
    """Whole normalised documents, including every step value and output.

    The only thing blanked is what `normalize_result_payload` blanks -- wall
    clock times and the execution id -- and it is blanked by the *same*
    function the runtime uses, so this test cannot drift into comparing less
    than it claims to.
    """
    import asyncio

    example = SUPPORTED_DIR / case.example
    cli_dir = tmp_path / "cli"
    api_dir = tmp_path / "api"
    cli_dir.mkdir()
    api_dir.mkdir()

    cli_result = _run_cli(["run", str(example), *_cli_inputs(case)], cwd=cli_dir)
    from_cli = normalize_result_payload(_payload(cli_result.stdout))

    previous = Path.cwd()
    os.chdir(api_dir)
    try:
        from_api = asyncio.run(
            create_test_orchestrator().execute_yaml(
                yaml_content=example.read_text(), context=dict(case.inputs)
            )
        ).normalized()
    finally:
        os.chdir(previous)

    assert from_cli == from_api, f"{case.id}: the CLI and the Python API disagree"

    # The two surfaces must also leave the same files behind, which the
    # document comparison alone would not catch.
    assert _artifacts(cli_dir) == _artifacts(api_dir), (
        f"{case.id}: the CLI and the Python API produced different artifacts"
    )


def test_running_a_pipeline_writes_a_checkpoint(tmp_path):
    """The side effect the artifact comparison excludes, asserted directly.

    `_artifacts` skips ./checkpoints/ so that an unexpected *output* file is a
    test failure. That exclusion is only honest if something checks the
    excluded thing exists, otherwise the runtime could stop checkpointing and
    no test would notice.
    """
    example = SUPPORTED_DIR / "01_hello_filesystem.yaml"

    result = _run_cli(["run", str(example)], cwd=tmp_path)
    assert result.returncode == 0

    checkpoints = sorted((tmp_path / "checkpoints").glob("*.json"))
    assert checkpoints, "a run left no checkpoint behind"
    assert checkpoints[0].name.startswith("hello_filesystem_"), (
        f"checkpoint {checkpoints[0].name} is not named after the pipeline"
    )
