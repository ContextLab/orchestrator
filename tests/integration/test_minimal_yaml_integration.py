"""Minimal integration test to verify basic YAML pipeline functionality."""

import pytest
from pathlib import Path

from orchestrator import Orchestrator, init_models
from orchestrator.compiler import YAMLCompiler
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem
from orchestrator.utils.api_keys import load_api_keys


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def setup_environment():
    """Setup test environment."""
    try:
        load_api_keys()
        return True
    except Exception as e:
        pytest.skip(f"API keys not configured: {e}")


@pytest.fixture(scope="module")
def orchestrator(setup_environment):
    """Create orchestrator with real models."""
    try:
        model_registry = init_models()
    except Exception as e:
        pytest.skip(f"Failed to initialize models: {e}")
    
    control_system = ModelBasedControlSystem(model_registry=model_registry)
    return Orchestrator(control_system=control_system, model_registry=model_registry)


@pytest.fixture
def yaml_compiler(orchestrator):
    """Create YAML compiler with model registry."""
    model_registry = orchestrator.control_system.model_registry if hasattr(orchestrator.control_system, 'model_registry') else None
    return YAMLCompiler(model_registry=model_registry)


@pytest.mark.timeout(60)
async def test_minimal_yaml_pipeline(orchestrator, yaml_compiler):
    """Test the simplest possible YAML pipeline."""
    yaml_content = """
name: "Minimal Test"
description: "Simplest possible pipeline"

steps:
  - id: test_step
    action: generate
    parameters:
      prompt: "Say hello"
      max_tokens: 10
"""
    
    # Compile and execute
    pipeline = await yaml_compiler.compile(yaml_content)
    result = await orchestrator.execute_pipeline(pipeline)
    
    # Verify result
    assert result is not None
    assert "test_step" in result
    assert isinstance(result["test_step"], str)
    assert len(result["test_step"]) > 0
    
    print(f"\nResult: {result['test_step']}")


@pytest.mark.timeout(90)
async def test_minimal_yaml_with_model(orchestrator, yaml_compiler):
    """Test YAML pipeline with explicit model selection."""
    yaml_content = """
name: "Minimal Model Test"
description: "Pipeline with explicit model"
model: "gpt-4o-mini"

steps:
  - id: test_with_model
    action: generate
    parameters:
      prompt: "Count to 5"
      max_tokens: 20
"""
    
    # Compile and execute
    pipeline = await yaml_compiler.compile(yaml_content)
    result = await orchestrator.execute_pipeline(pipeline)
    
    # Verify result
    assert result is not None
    assert "test_with_model" in result
    assert isinstance(result["test_with_model"], str)
    assert len(result["test_with_model"]) > 0
    
    print(f"\nResult with model: {result['test_with_model']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])