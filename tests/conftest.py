"""Shared pytest fixtures for orchestrator tests.

Two rules this file enforces (see docs/adr/0001-product-contract.md):

1. **The test suite never mutates the host.** Collection and startup must not
   install or start Docker, download models, or touch anything outside the
   working tree. A previous session-scoped ``autouse`` fixture called
   ``DockerManager.ensure_docker_ready(install_if_missing=True)``, so merely
   *collecting* tests could try to install Docker. That is gone.

2. **The package is imported under exactly one identity.** Importing both
   ``orchestrator`` and ``src.orchestrator`` creates two copies of every module:
   duplicate singleton registries, and ``isinstance`` checks that fail between
   two classes that are supposed to be the same. Everything imports
   ``orchestrator``.
"""

import os
import sys
from pathlib import Path

import pytest

# Prefer an installed package. Fall back to the source tree only when the
# package is not installed, so a plain `pytest` in a fresh clone still works.
# Note this inserts `src/` (so the import name stays `orchestrator`), never the
# repo root (which would allow the `src.orchestrator` alias back in).
try:  # pragma: no cover - environment dependent
    import orchestrator  # noqa: F401
except ImportError:  # pragma: no cover - environment dependent
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# --------------------------------------------------------------------------
# Marker-based gating
# --------------------------------------------------------------------------
# Tests that need secrets or external services are skipped unless the
# environment actually provides them. They are never silently "passed", and the
# skip reason always states what is missing.

def _docker_running() -> bool:
    """Report whether a Docker daemon is reachable. Never starts or installs."""
    try:
        from orchestrator.utils.docker_manager import DockerManager

        return bool(DockerManager.is_running())
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip opt-in tests when their prerequisites are absent."""
    have_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

    # Live tests are not all about the same provider, so they cannot share one
    # credential gate. Dartmouth Chat tests need a Dartmouth key -- and they
    # cost nothing to run, so gating them behind a paid provider's key would
    # needlessly forgo free coverage.
    def _have_dartmouth() -> bool:
        try:
            from orchestrator.models.dartmouth_credentials import (
                resolve_dartmouth_api_key,
            )

            return resolve_dartmouth_api_key(required=False) is not None
        except Exception:
            return False

    have_dartmouth = _have_dartmouth()

    def _credential_for(item) -> tuple[bool, str]:
        """Which credential a live test needs, and whether we have it."""
        if "dartmouth" in str(getattr(item, "fspath", "")).lower():
            return have_dartmouth, "DARTMOUTH_CHAT_API_KEY"
        return have_anthropic, "ANTHROPIC_API_KEY"

    # The live CI job sets ORCHESTRATOR_REQUIRE_LIVE=1. Without this guard a
    # missing key would skip every live test and the job would report success
    # having exercised no provider at all -- indistinguishable from having no
    # live coverage, which is the state this suite is meant to leave behind.
    if os.environ.get("ORCHESTRATOR_REQUIRE_LIVE") == "1" and not (
        have_anthropic or have_dartmouth
    ):
        raise pytest.UsageError(
            "ORCHESTRATOR_REQUIRE_LIVE=1 requires real live coverage, but "
            "neither ANTHROPIC_API_KEY nor a Dartmouth Chat credential is "
            "available. Provide one, or unset ORCHESTRATOR_REQUIRE_LIVE."
        )

    run_integration = os.environ.get("ORCHESTRATOR_RUN_INTEGRATION") == "1"
    docker_ok = None  # probed lazily; the check itself is cheap but not free

    for item in items:
        if "live" in item.keywords:
            available, variable = _credential_for(item)
            if not available:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"live-provider test: set {variable} to run"
                    )
                )
        if "integration" in item.keywords and not run_integration:
            item.add_marker(
                pytest.mark.skip(
                    reason="integration test: set ORCHESTRATOR_RUN_INTEGRATION=1 to run"
                )
            )
        if "docker" in item.keywords:
            if docker_ok is None:
                docker_ok = _docker_running()
            if not docker_ok:
                item.add_marker(
                    pytest.mark.skip(reason="docker test: no Docker daemon reachable")
                )


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def populated_model_registry():
    """A model registry populated from local config and available API keys.

    Session-scoped because ``init_models()`` is expensive. Tests that require
    an actual model are skipped when none could be registered, rather than
    failing with a confusing downstream error.
    """
    from orchestrator import init_models

    registry = init_models()
    if not registry.list_models():
        pytest.skip("no models available (missing API keys or local models)")
    return registry


@pytest.fixture(scope="session")
def model_registry(populated_model_registry):
    """Alias for :func:`populated_model_registry` used by older tests."""
    return populated_model_registry


@pytest.fixture
def docker_available() -> bool:
    """Whether Docker is usable. Read-only: never installs or starts anything.

    ```python
    def test_something(docker_available):
        if not docker_available:
            pytest.skip("Docker not available")
    ```
    """
    return _docker_running()
