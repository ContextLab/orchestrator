"""Repository-level invariants that only a real git tree can verify.

CI already enforces the gitignore rule, but a CI check cannot see a file that
never made it into the clone -- by then the damage is done and the package is
already broken for everyone. These run in the blocking gate so the failure
lands on the machine that still has the file, before it is pushed.

No mocks: these shell out to the real git binary in the real working tree.
"""

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    """Run git in the repository and return stdout, or skip if unavailable."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover
        pytest.skip(f"git is not usable here: {exc}")
    if result.returncode != 0:
        pytest.skip(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def test_no_source_file_is_hidden_by_gitignore():
    """The regression that shipped a broken package for ~11 months.

    An unanchored `debug_*` rule in .gitignore silently excluded
    src/orchestrator/quality/debug_artifact_detector.py -- a real 521-line
    module. Every developer's tree still had the file, so every local test
    passed; only fresh clones and installed wheels broke, with
    `ModuleNotFoundError`.

    Nothing about running the code can catch this. Only asking git what it is
    actually ignoring can.
    """
    hidden = _git(
        "ls-files", "--others", "--ignored", "--exclude-standard", "--", "src"
    ).split()
    python_sources = [path for path in hidden if path.endswith(".py")]
    assert not python_sources, (
        "these source files exist but .gitignore excludes them, so they will "
        "be missing from a fresh clone and from the built wheel:\n  "
        + "\n  ".join(python_sources)
        + "\n\nAnchor the offending .gitignore rule (e.g. `/debug_*` instead "
        "of `debug_*`) or add a negation for the source tree."
    )


def test_every_package_directory_under_src_is_tracked():
    """A package whose __init__.py is untracked breaks the same way.

    Catches the case where the module itself is committed but the package
    marker that makes it importable is not.
    """
    tracked = set(_git("ls-files", "--", "src").split())
    missing = [
        str(init.relative_to(REPO_ROOT))
        for init in (REPO_ROOT / "src").rglob("__init__.py")
        if "__pycache__" not in init.parts
        and str(init.relative_to(REPO_ROOT)) not in tracked
    ]
    assert not missing, (
        "these package markers are untracked, so the packages will not be "
        "importable from a fresh clone:\n  " + "\n  ".join(missing)
    )
