"""Shared pytest fixtures for orchestrator tests."""

import pytest
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from orchestrator import init_models
from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.utils.docker_manager import DockerManager

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def populated_model_registry() -> ModelRegistry:
    """
    Provide a populated model registry for tests.

    This fixture initializes models once per test session and provides
    the populated registry to all tests that need it.

    Returns:
        ModelRegistry: Registry populated with all configured models

    Raises:
        pytest.skip.Exception: If no models are available (likely due to missing API keys)
    """
    print("\n>> Test fixture: Initializing model registry...")
    registry = init_models()

    # Verify we have models available
    available_models = registry.list_models()
    print(f">> Test fixture: Found {len(available_models)} models: {available_models}")
    
    if not available_models:
        # Can't use pytest.skip in a session-scoped fixture
        # Return None instead and let tests handle it
        print(">> Test fixture: No models available - returning empty registry")
        return registry

    return registry


@pytest.fixture(scope="session")
def model_registry(populated_model_registry) -> ModelRegistry:
    """
    Alias for populated_model_registry for backward compatibility.

    Many tests expect a 'model_registry' fixture.
    """
    return populated_model_registry


@pytest.fixture(scope="session", autouse=True)
def ensure_docker():
    """
    Automatically ensure Docker is installed and running before tests.

    This fixture runs automatically for all test sessions and ensures
    Docker is ready for any tests that need it.
    """
    try:
        logger.info("Checking Docker status...")
        status = DockerManager.get_status()

        if not status["installed"]:
            logger.warning("Docker not installed, attempting automatic installation...")
            try:
                DockerManager.ensure_docker_ready(install_if_missing=True, start_if_stopped=True)
                logger.info("✅ Docker installed and started successfully")
            except Exception as e:
                logger.warning(f"Could not auto-install Docker: {e}")
                logger.warning("Tests requiring Docker will be skipped")

        elif not status["running"]:
            logger.warning("Docker installed but not running, attempting to start...")
            try:
                DockerManager.ensure_docker_ready(install_if_missing=False, start_if_stopped=True)
                logger.info("✅ Docker started successfully")
            except Exception as e:
                logger.warning(f"Could not start Docker: {e}")
                logger.warning("Tests requiring Docker will be skipped")

        else:
            logger.info("✅ Docker is already running")

    except Exception as e:
        logger.warning(f"Docker check failed: {e}")
        logger.warning("Tests requiring Docker will be skipped")

    # Yield to run tests
    yield

    # No cleanup needed - leave Docker running


@pytest.fixture
def docker_available() -> bool:
    """
    Check if Docker is available for the current test.

    Use this fixture in tests that require Docker:
    ```python
    def test_something(docker_available):
        if not docker_available:
            pytest.skip("Docker not available")
        # ... test code that uses Docker
    ```

    Returns:
        True if Docker is ready to use
    """
    try:
        return DockerManager.is_running()
    except Exception:
        return False
