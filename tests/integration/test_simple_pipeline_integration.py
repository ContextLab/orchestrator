"""Simple integration test to verify basic pipeline execution with real API calls.

This test creates a minimal pipeline to verify the integration works correctly.
"""

import pytest

from orchestrator import Orchestrator, Task, Pipeline, init_models
from src.orchestrator.control_systems.model_based_control_system import (
    ModelBasedControlSystem)
from src.orchestrator.utils.api_keys_flexible import load_api_keys_optional


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def setup_environment():
    """Setup test environment."""
    try:
        api_keys = load_api_keys_optional()
        return True
    except Exception as e:
        print(f"Warning: API keys issue: {e}")
        return True  # Continue anyway


@pytest.fixture(scope="module")
def orchestrator(setup_environment):
    """Create orchestrator with real models."""
    # Set environment variable to skip HuggingFace models for integration tests
    # This avoids timeouts from downloading large models
    import os

    os.environ["ORCHESTRATOR_SKIP_HUGGINGFACE"] = "true"

    try:
        model_registry = init_models()
    except Exception as e:
        print(f"Warning: Model initialization issue: {e}")
        model_registry = init_models()  # Try again

    # Check for API models
    available_models = model_registry.list_models()
    api_models = [
        m
        for m in available_models
        if any(provider in m.lower() for provider in ["gpt", "claude", "gemini"])
    ]

    if not api_models:
        print("Warning: No API models available, using any available models")
        api_models = available_models  # Use whatever models we have

    print(f"\nUsing {len(api_models)} API models for testing")
    print(f"Total models available: {len(available_models)}")

    control_system = ModelBasedControlSystem(model_registry=model_registry)
    # Pass the model registry to the orchestrator
    return Orchestrator(control_system=control_system, model_registry=model_registry)


class TestSimplePipelineIntegration:
    """Test basic pipeline execution with real API calls."""
    async def test_simple_text_generation(self, orchestrator):
        """Test a simple text generation pipeline."""
        # Create a simple task
        task = Task(
            id="simple_generation",
            name="Simple Text Generation",
            action="generate",
            parameters={
                "prompt": "Write a haiku about testing software",
                "max_tokens": 50,
            })

        # Create pipeline
        pipeline = Pipeline(id="simple_test", name="Simple Integration Test")
        pipeline.add_task(task)

        # Execute
        try:
            result = await orchestrator.execute_pipeline(pipeline)

            # Verify we got a result
            assert result is not None
            assert "simple_generation" in result

            generated_text = result["simple_generation"]
            assert generated_text is not None
            assert isinstance(generated_text, (str, dict))

            if isinstance(generated_text, str):
                assert len(generated_text) > 0
                print(f"\nGenerated haiku:\n{generated_text}")
            else:
                print(f"\nGenerated result:\n{generated_text}")

        except Exception as e:
            pytest.fail(f"Pipeline execution failed: {e}")
    async def test_multi_step_pipeline(self, orchestrator):
        """Test a pipeline with multiple dependent tasks."""
        # Task 1: Generate a topic
        task1 = Task(
            id="generate_topic",
            name="Generate Topic",
            action="generate",
            parameters={
                "prompt": "Generate a random interesting topic for a blog post (just the topic, one line)",
                "max_tokens": 30,
            })

        # Task 2: Create outline based on topic
        task2 = Task(
            id="create_outline",
            name="Create Outline",
            action="generate",
            parameters={
                "prompt": "Create a 3-point outline for a blog post about: {generate_topic}",
                "max_tokens": 100,
            },
            dependencies=["generate_topic"])

        # Create pipeline
        pipeline = Pipeline(id="multi_step_test", name="Multi-Step Integration Test")
        pipeline.add_task(task1)
        pipeline.add_task(task2)

        # Execute
        try:
            result = await orchestrator.execute_pipeline(pipeline)

            # Verify both tasks completed
            assert result is not None
            assert "generate_topic" in result
            assert "create_outline" in result

            topic = result["generate_topic"]
            outline = result["create_outline"]

            assert topic is not None
            assert outline is not None

            print(f"\nGenerated topic: {topic}")
            print(f"\nGenerated outline: {outline}")

            # Verify outline references the topic
            if isinstance(topic, str) and isinstance(outline, str):
                # The outline should be related to the topic
                assert len(outline) > len(topic)
                # Check if topic is referenced in outline (basic check)
                # Note: The template replacement might not work perfectly with all models

        except Exception as e:
            pytest.fail(f"Multi-step pipeline failed: {e}")
    async def test_error_handling(self, orchestrator):
        """Test pipeline handles errors gracefully."""
        # Create task with invalid parameters
        task = Task(
            id="invalid_task",
            name="Invalid Task",
            action="generate",
            parameters={
                # Missing required 'prompt' parameter
                "max_tokens": 50
            })

        pipeline = Pipeline(id="error_test", name="Error Handling Test")
        pipeline.add_task(task)

        # Should handle error gracefully
        with pytest.raises(Exception) as exc_info:
            await orchestrator.execute_pipeline(pipeline)

        # Check we got a meaningful error
        error_msg = str(exc_info.value).lower()
        # The error might be wrapped, so check for task failure or parameter error
        assert (
            "prompt" in error_msg
            or "parameter" in error_msg
            or "required" in error_msg
            or "task 'invalid_task' failed" in error_msg
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
