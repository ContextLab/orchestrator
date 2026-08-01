"""The CLI must be hermetic on local-only runs, and quiet by default.

Two defects are pinned here:

1. **Credential discovery ran eagerly.** Every ``orchestrator run`` called
   ``init_models()`` before it knew whether the pipeline wanted a model, so a
   purely local, filesystem-only pipeline read ``~/.orchestrator/.env`` and
   probed every configured provider. A tool-only run must not inspect provider
   credentials at all.

2. **Default output was a debug log.** A two-step pipeline emitted well over a
   hundred lines of tracing. Tracing belongs behind ``--verbose``.

Nothing here is mocked. The credential tests point ``HOME`` at a temporary
directory holding a decoy ``.env`` and then check, from the outside, that the
decoy never reaches the process -- neither its output nor its environment.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).parent.parent / "src")
GOLDEN_DIR = Path(__file__).parent / "golden"
BASIC = GOLDEN_DIR / "basic.yaml"

DECOY_ANTHROPIC = "sk-ant-DECOY-must-never-be-read"
DECOY_OPENAI = "sk-DECOY-must-never-be-read"

PROVIDER_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_AI_API_KEY",
    "HF_TOKEN",
)

pytestmark = [pytest.mark.e2e, pytest.mark.contract]


def _decoy_home(tmp_path):
    """A throwaway HOME containing provider credentials that must not be read."""
    home = tmp_path / "home"
    (home / ".orchestrator").mkdir(parents=True)
    (home / ".orchestrator" / ".env").write_text(
        f"ANTHROPIC_API_KEY={DECOY_ANTHROPIC}\nOPENAI_API_KEY={DECOY_OPENAI}\n"
    )
    return home


def _hermetic_env(home):
    """An environment with no ambient provider credentials to fall back on."""
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["HOME"] = str(home)
    env["ORCHESTRATOR_AUTO_INSTALL"] = "0"
    for var in PROVIDER_ENV_VARS:
        env.pop(var, None)
    return env


def _run_cli(args, cwd, home):
    return subprocess.run(
        [sys.executable, "-m", "orchestrator.cli", *args],
        cwd=str(cwd),
        env=_hermetic_env(home),
        capture_output=True,
        text=True,
        timeout=300,
    )


# ---------------------------------------------------------------------------
# 0. the CLI can state its own version
# ---------------------------------------------------------------------------

def test_cli_reports_its_version(tmp_path):
    """`--version` is the first thing anyone runs against a release.

    A release rehearsal against the built wheel found this missing entirely:
    `orchestrator --version` failed with "No such option". There is no other
    way to ask an installed copy what it is.
    """
    import orchestrator

    result = _run_cli(["--version"], tmp_path, _decoy_home(tmp_path))

    assert result.returncode == 0, f"--version failed: {result.stderr}"
    assert orchestrator.__version__ in result.stdout, (
        f"version {orchestrator.__version__!r} missing from {result.stdout!r}"
    )


def test_version_does_not_read_credentials(tmp_path):
    """Asking the version must not touch the user's provider credentials.

    The version is read from distribution metadata rather than by importing
    the package, so nothing on the credential path runs.
    """
    home = _decoy_home(tmp_path)
    result = _run_cli(["--version"], tmp_path, home)

    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert DECOY_ANTHROPIC not in combined
    assert DECOY_OPENAI not in combined


# ---------------------------------------------------------------------------
# 1. hermetic boundary
# ---------------------------------------------------------------------------

def test_local_only_run_never_reads_the_credentials_file(tmp_path):
    """A filesystem-only pipeline must not touch ~/.orchestrator/.env."""
    home = _decoy_home(tmp_path)
    work = tmp_path / "work"
    work.mkdir()

    result = _run_cli(["run", str(BASIC), "-i", "greeting=hi"], cwd=work, home=home)

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert (work / "golden_out" / "greeting.txt").read_text() == "hi world"

    combined = result.stdout + result.stderr
    assert DECOY_ANTHROPIC not in combined
    assert DECOY_OPENAI not in combined
    # Not just the values: the run must not report having *found* credentials,
    # which is what the eager path did before it concluded no model was usable.
    lowered = combined.lower()
    assert "api key" not in lowered, combined
    assert ".orchestrator/.env" not in combined, combined


def test_local_only_run_leaves_provider_credentials_out_of_the_environment(tmp_path):
    """The decoy keys must not reach ``os.environ`` either.

    Population copies every discovered key into the process environment, so
    "did ANTHROPIC_API_KEY appear?" is a direct, observable test of whether
    credential discovery ran. The pipeline is executed exactly as the CLI runs
    it, through ``cli._build_orchestrator``.
    """
    home = _decoy_home(tmp_path)
    work = tmp_path / "work"
    work.mkdir()

    probe = tmp_path / "probe.py"
    probe.write_text(
        "import asyncio, json, os\n"
        "from orchestrator.cli import _build_orchestrator\n"
        "orchestrator = _build_orchestrator()\n"
        "results = asyncio.run(orchestrator.execute_yaml_file(%r, {'greeting': 'hi'}))\n"
        "asyncio.run(orchestrator.shutdown())\n"
        "print(json.dumps({\n"
        "    'step_ok': results['read_back']['success'],\n"
        "    'anthropic': os.environ.get('ANTHROPIC_API_KEY'),\n"
        "    'openai': os.environ.get('OPENAI_API_KEY'),\n"
        "    'populated': orchestrator.model_registry.populated,\n"
        "}))\n" % str(BASIC)
    )

    result = subprocess.run(
        [sys.executable, str(probe)],
        cwd=str(work),
        env=_hermetic_env(home),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["step_ok"] is True
    assert report["anthropic"] is None, "credential file was read on a tool-only run"
    assert report["openai"] is None, "credential file was read on a tool-only run"
    assert report["populated"] is False, "model registry populated without demand"


def test_registry_populates_when_a_model_is_actually_demanded(tmp_path):
    """Lazy must mean deferred, not skipped.

    Asking the registry for a model has to trigger the discovery that a
    tool-only run avoided -- otherwise the fix would simply have disabled
    models. Whether any model can be served here depends on the machine, so
    this asserts on the discovery having *run*, not on its outcome.
    """
    home = _decoy_home(tmp_path)

    probe = tmp_path / "probe.py"
    probe.write_text(
        "import asyncio, json\n"
        "from orchestrator._api import populate_model_registry\n"
        "from orchestrator.models.lazy_registry import LazyModelRegistry\n"
        "registry = LazyModelRegistry(populate_model_registry)\n"
        "before = registry.populated\n"
        "try:\n"
        "    asyncio.run(registry.select_model({'tasks': ['generate']}))\n"
        "except Exception:\n"
        "    pass\n"
        "print(json.dumps({'before': before, 'after': registry.populated}))\n"
    )

    result = subprocess.run(
        [sys.executable, str(probe)],
        cwd=str(tmp_path),
        env=_hermetic_env(home),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["before"] is False
    assert report["after"] is True, "demanding a model did not trigger discovery"


# ---------------------------------------------------------------------------
# 2. output volume
# ---------------------------------------------------------------------------

def test_default_output_is_only_the_result_document(tmp_path):
    """Default stdout is the typed result and nothing else; stderr is empty."""
    home = _decoy_home(tmp_path)
    work = tmp_path / "work"
    work.mkdir()

    result = _run_cli(["run", str(BASIC), "-i", "greeting=hi"], cwd=work, home=home)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    # Parsable with no leading noise to skip past.
    payload = json.loads(result.stdout)
    assert payload["read_back"]["result"]["content"] == "hi world"

    assert result.stderr == "", f"unexpected output on stderr:\n{result.stderr}"

    # The whole run is the result document -- a two-step pipeline previously
    # emitted well over a hundred lines around it.
    rendered = json.dumps(payload, indent=2, default=str)
    assert result.stdout.strip() == rendered
    assert len(result.stdout.splitlines()) < 30


def test_verbose_restores_the_detailed_trace(tmp_path):
    """`--verbose` must still show what the default output now suppresses."""
    home = _decoy_home(tmp_path)
    quiet_dir = tmp_path / "quiet"
    loud_dir = tmp_path / "loud"
    quiet_dir.mkdir()
    loud_dir.mkdir()

    quiet = _run_cli(["run", str(BASIC), "-i", "greeting=hi"], cwd=quiet_dir, home=home)
    loud = _run_cli(
        ["run", str(BASIC), "-i", "greeting=hi", "--verbose"], cwd=loud_dir, home=home
    )

    assert quiet.returncode == 0, quiet.stderr
    assert loud.returncode == 0, loud.stderr

    quiet_lines = len((quiet.stdout + quiet.stderr).splitlines())
    loud_lines = len((loud.stdout + loud.stderr).splitlines())
    assert loud_lines > quiet_lines * 2, (
        f"--verbose produced {loud_lines} lines vs {quiet_lines} quiet; "
        "the detailed trace is missing"
    )

    trace = loud.stdout + loud.stderr
    assert "DEBUG" in trace
    assert "Routing to tool handler" in trace

    # Verbosity is about reporting, not about reaching for credentials.
    assert DECOY_ANTHROPIC not in trace
    assert DECOY_OPENAI not in trace

    # Both surfaces still produce the same result document.
    assert json.loads(quiet.stdout) == json.loads(
        loud.stdout[loud.stdout.index("{"):]
    )


def test_tool_only_run_reports_no_provider_registration_warnings(tmp_path):
    """No "Error registering <provider> model" noise on a tool-only run."""
    home = _decoy_home(tmp_path)
    work = tmp_path / "work"
    work.mkdir()

    result = _run_cli(["run", str(BASIC), "-i", "greeting=hi"], cwd=work, home=home)
    combined = result.stdout + result.stderr
    assert "registering" not in combined.lower(), combined


def test_missing_provider_library_is_reported_once(tmp_path):
    """The cause is named once, not wrapped in a bogus installation failure.

    ``AnthropicModel(...)`` used to raise "Failed to install Anthropic library:
    Anthropic library is not installed ..." -- one cause reported twice, with
    the outer half naming an installation that was never attempted.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import orchestrator.integrations.anthropic_model as m\n"
        "if m.ANTHROPIC_AVAILABLE:\n"
        "    print('SKIP')\n"
        "else:\n"
        "    try:\n"
        "        m.AnthropicModel(model_name='claude-sonnet-4-20250514', api_key='x')\n"
        "    except ImportError as exc:\n"
        "        print('ERR', exc)\n"
    )

    env = _hermetic_env(_decoy_home(tmp_path))
    result = subprocess.run(
        [sys.executable, str(probe)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    if result.stdout.strip() == "SKIP":
        pytest.skip("anthropic library is installed; nothing to report")

    message = result.stdout.strip()
    assert message.startswith("ERR ")
    assert "is not installed" in message
    assert "Failed to install" not in message, message
    # The cause is stated once.
    assert message.count("is not installed") == 1, message
