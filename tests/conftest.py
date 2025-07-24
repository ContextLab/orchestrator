"""Shared pytest fixtures for orchestrator tests."""

import pytest
from orchestrator import init_models
from orchestrator.models.model_registry import ModelRegistry


@pytest.fixture(scope="session")
def populated_model_registry() -> ModelRegistry:
    """
    Provide a populated model registry for tests.

    This fixture initializes models once per test session and provides
    the populated registry to all tests that need it.

    Returns:
        ModelRegistry: Registry populated with all configured models

    Raises:
        AssertionError: If no models are available (likely due to missing API keys)
    """
    registry = init_models()

    # Verify we have models available
    available_models = registry.list_models()
    if not available_models:
        raise AssertionError(
            "No models available in registry. "
            "Please configure API keys in ~/.orchestrator/.env"
        )

    return registry


@pytest.fixture(scope="session")
def model_registry(populated_model_registry) -> ModelRegistry:
    """
    Alias for populated_model_registry for backward compatibility.

    Many tests expect a 'model_registry' fixture.
    """
    return populated_model_registry
