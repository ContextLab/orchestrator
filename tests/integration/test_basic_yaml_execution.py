"""Basic integration tests for YAML pipeline execution with real models.

These tests focus on verifying that YAML pipelines can be compiled and executed
with real models, making actual API calls to generate content.
"""

import pytest

from orchestrator import Orchestrator, init_models
from orchestrator.compiler import YAMLCompiler
from orchestrator.control_systems.model_based_control_system import (
    ModelBasedControlSystem,
)
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

    # Check for API models
    available_models = model_registry.list_models()
    api_models = [
        m
        for m in available_models
        if any(provider in m.lower() for provider in ["gpt", "claude", "gemini"])
    ]

    if not api_models:
        pytest.skip("No API models available")

    print(f"\nUsing {len(api_models)} API models for testing")

    control_system = ModelBasedControlSystem(model_registry=model_registry)
    return Orchestrator(control_system=control_system, model_registry=model_registry)


@pytest.fixture
def yaml_compiler(orchestrator):
    """Create YAML compiler with model registry."""
    model_registry = (
        orchestrator.control_system.model_registry
        if hasattr(orchestrator.control_system, "model_registry")
        else None
    )
    return YAMLCompiler(model_registry=model_registry)


class TestBasicYAMLExecution:
    """Test basic YAML pipeline execution."""

    @pytest.mark.timeout(60)
    async def test_simple_yaml_pipeline(self, orchestrator, yaml_compiler):
        """Test a very simple YAML pipeline with one step."""
        yaml_content = """
name: "Simple Test Pipeline"
description: "Basic test pipeline with single step"

steps:
  - id: generate_text
    action: generate
    parameters:
      prompt: "Write one sentence about the color blue."
      max_tokens: 30
"""

        # Compile and execute
        pipeline = await yaml_compiler.compile(yaml_content)
        result = await orchestrator.execute_pipeline(pipeline)

        # Verify result
        assert result is not None
        assert "generate_text" in result
        assert isinstance(result["generate_text"], str)
        assert len(result["generate_text"]) > 0

        print(f"\nGenerated text: {result['generate_text']}")

    @pytest.mark.timeout(90)
    async def test_yaml_pipeline_with_context(self, orchestrator, yaml_compiler):
        """Test YAML pipeline with context variables."""
        yaml_content = """
name: "Context Test Pipeline"
description: "Test pipeline with context variables"

inputs:
  topic:
    type: string
    description: "Topic to write about"
    required: true

  max_words:
    type: integer
    description: "Maximum number of words"
    default: 20

steps:
  - id: write_about_topic
    action: generate
    parameters:
      prompt: "Write about {{topic}} in exactly {{max_words}} words."
      max_tokens: 50
"""

        # Compile with context
        context = {"topic": "artificial intelligence", "max_words": 15}

        pipeline = await yaml_compiler.compile(yaml_content, context=context)
        result = await orchestrator.execute_pipeline(pipeline)

        # Verify result
        assert result is not None
        assert "write_about_topic" in result
        text = result["write_about_topic"]
        assert isinstance(text, str)
        assert "artificial intelligence" in text.lower() or "ai" in text.lower()

        print(f"\nGenerated text about {context['topic']}: {text}")

    @pytest.mark.timeout(120)
    async def test_yaml_pipeline_with_dependencies(self, orchestrator, yaml_compiler):
        """Test YAML pipeline with dependent steps."""
        yaml_content = """
name: "Dependency Test Pipeline"
description: "Test pipeline with step dependencies"

steps:
  - id: generate_topic
    action: generate
    parameters:
      prompt: "Generate a random topic for a blog post (one line only):"
      max_tokens: 20

  - id: write_intro
    action: generate
    parameters:
      prompt: "Write a one-sentence introduction for a blog post about: {{generate_topic}}"
      max_tokens: 50
    depends_on: [generate_topic]
"""

        # Compile and execute
        pipeline = await yaml_compiler.compile(yaml_content)
        result = await orchestrator.execute_pipeline(pipeline)

        # Verify both steps executed
        assert result is not None
        assert "generate_topic" in result
        assert "write_intro" in result

        topic = result["generate_topic"]
        intro = result["write_intro"]

        assert isinstance(topic, str)
        assert isinstance(intro, str)
        assert len(topic) > 0
        assert len(intro) > 0

        print(f"\nGenerated topic: {topic}")
        print(f"Generated intro: {intro}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
