"""Integration tests for simplified versions of complex YAML pipelines."""

import pytest

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
    model_registry = (
        orchestrator.control_system.model_registry
        if hasattr(orchestrator.control_system, "model_registry")
        else None
    )
    return YAMLCompiler(model_registry=model_registry)


class TestSimplifiedCodeAnalysis:
    """Test simplified version of code analysis pipeline."""

    @pytest.mark.timeout(90)
    async def test_simplified_code_analysis(self, orchestrator, yaml_compiler, tmp_path):
        """Test a simplified code analysis with just key steps."""
        # Create sample code file
        code_file = tmp_path / "sample.py"
        code_file.write_text(
            """
def calculate_sum(a, b):
    # Simple function to add two numbers
    return a + b

def main():
    result = calculate_sum(5, 3)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
"""
        )

        yaml_content = """
name: "Simplified Code Analysis"
description: "Minimal code analysis for testing"
model: "gpt-4o-mini"

inputs:
  repo_path:
    type: string
    required: true

steps:
  - id: analyze_code
    action: generate
    parameters:
      prompt: |
        Analyze this Python code and provide:
        1. Brief code quality assessment
        2. Any potential issues
        3. One improvement suggestion
        
        Code path: {{repo_path}}
        Keep response under 100 words.
      max_tokens: 150
    
  - id: generate_summary
    action: generate
    parameters:
      prompt: |
        Based on the analysis: {{analyze_code}}
        
        Provide a one-line summary of the code quality.
      max_tokens: 50
    depends_on: [analyze_code]
"""

        # Compile with context
        context = {"repo_path": str(code_file)}
        pipeline = await yaml_compiler.compile(yaml_content, context=context)

        # Execute
        result = await orchestrator.execute_pipeline(pipeline)

        # Verify
        assert result is not None
        assert "analyze_code" in result
        assert "generate_summary" in result
        assert len(result["analyze_code"]) > 0
        assert len(result["generate_summary"]) > 0

        print(f"\nAnalysis: {result['analyze_code']}")
        print(f"Summary: {result['generate_summary']}")


class TestSimplifiedCreativeWriting:
    """Test simplified version of creative writing pipeline."""

    @pytest.mark.timeout(60)
    async def test_simplified_story_generation(self, orchestrator, yaml_compiler):
        """Test simplified story generation with minimal steps."""
        yaml_content = """
name: "Simplified Story Generator"
description: "Minimal story generation for testing"
model: "gpt-4o-mini"

inputs:
  genre:
    type: string
    default: "sci-fi"
  
  topic:
    type: string
    default: "time travel"

steps:
  - id: generate_premise
    action: generate
    parameters:
      prompt: |
        Create a one-sentence {{genre}} story premise about {{topic}}.
      max_tokens: 50
  
  - id: write_opening
    action: generate
    parameters:
      prompt: |
        Write a 3-sentence opening for this story:
        Premise: {{generate_premise}}
      max_tokens: 100
    depends_on: [generate_premise]
"""

        # Compile with defaults
        pipeline = await yaml_compiler.compile(yaml_content)

        # Execute
        result = await orchestrator.execute_pipeline(pipeline)

        # Verify
        assert result is not None
        assert "generate_premise" in result
        assert "write_opening" in result
        assert len(result["generate_premise"]) > 0
        assert len(result["write_opening"]) > 0

        print(f"\nPremise: {result['generate_premise']}")
        print(f"Opening: {result['write_opening']}")


class TestSimplifiedDataProcessing:
    """Test simplified version of data processing pipeline."""

    @pytest.mark.timeout(60)
    async def test_simplified_data_analysis(self, orchestrator, yaml_compiler):
        """Test simplified data processing with minimal steps."""
        yaml_content = """
name: "Simplified Data Processor"
description: "Minimal data processing for testing"
model: "gpt-4o-mini"

inputs:
  data_description:
    type: string
    default: "sales data with 100 records containing date, product, and revenue"

steps:
  - id: analyze_data
    action: generate
    parameters:
      prompt: |
        Analyze this dataset: {{data_description}}
        Provide 2 key insights in bullet points.
      max_tokens: 100
  
  - id: suggest_visualization
    action: generate
    parameters:
      prompt: |
        Based on: {{data_description}}
        Suggest the best chart type for visualization (one line).
      max_tokens: 30
"""

        # Compile
        pipeline = await yaml_compiler.compile(yaml_content)

        # Execute
        result = await orchestrator.execute_pipeline(pipeline)

        # Verify
        assert result is not None
        assert "analyze_data" in result
        assert "suggest_visualization" in result

        print(f"\nAnalysis: {result['analyze_data']}")
        print(f"Visualization: {result['suggest_visualization']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
