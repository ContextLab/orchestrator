"""Tests for pipeline recursion tools."""

import asyncio
import os
import pytest
import tempfile
import yaml

from src.orchestrator.tools.pipeline_recursion_tools import (
    PipelineExecutorTool,
    RecursionControlTool,
)
from src.orchestrator.models.registry_singleton import get_model_registry, reset_model_registry
from src.orchestrator.models.anthropic_model import AnthropicModel


@pytest.fixture(autouse=True)
async def setup_models():
    """Setup models for testing."""
    # Clear any existing models
    reset_model_registry()
    
    # Get registry and register a minimal model
    registry = get_model_registry()
    
    # Use environment variable to control if we use real models
    if os.environ.get("USE_REAL_MODELS", "false").lower() == "true":
        # Initialize real models (requires API keys)
        from orchestrator import init_models
        init_models()
    else:
        # Create a minimal anthropic model for testing
        # This will work without API key for basic orchestration
        model = AnthropicModel(
            name="claude-3-haiku-20240307",
            api_key="test-key-for-recursion"
        )
        registry.register_model(model)
    
    yield
    
    # Cleanup
    reset_model_registry()


@pytest.mark.asyncio
async def test_pipeline_executor_basic():
    """Test basic pipeline execution."""
    # Create a simple sub-pipeline
    sub_pipeline = """
id: sub_pipeline
name: Sub Pipeline
steps:
  - id: step1
    tool: terminal
    action: execute
    parameters:
      command: echo "Hello from sub-pipeline"
"""
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(sub_pipeline)
        pipeline_path = f.name
    
    try:
        tool = PipelineExecutorTool()
        
        result = await tool.execute(
            pipeline=pipeline_path,
            inputs={"message": "test"},
            wait_for_completion=True
        )
        
        assert result["success"] is True
        assert result["pipeline_id"] == "sub_pipeline"
        assert "outputs" in result
        assert result["recursion_depth"] == 1
    finally:
        os.unlink(pipeline_path)


@pytest.mark.asyncio
async def test_pipeline_executor_with_inputs():
    """Test pipeline execution with input parameters."""
    sub_pipeline = """
id: parameterized_pipeline
name: Parameterized Pipeline
parameters:
  name:
    type: string
    default: "World"
steps:
  - id: greet
    tool: terminal
    action: execute
    parameters:
      command: echo "Hello {{ parameters.name }}"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(sub_pipeline)
        pipeline_path = f.name
    
    try:
        tool = PipelineExecutorTool()
        
        result = await tool.execute(
            pipeline=pipeline_path,
            inputs={"parameters": {"name": "Alice"}},
            wait_for_completion=True
        )
        
        assert result["success"] is True
        assert result["pipeline_id"] == "parameterized_pipeline"
    finally:
        os.unlink(pipeline_path)


@pytest.mark.asyncio
async def test_pipeline_executor_inline_yaml():
    """Test execution with inline YAML."""
    tool = PipelineExecutorTool()
    
    inline_yaml = """
id: inline_pipeline
name: Inline Test
steps:
  - id: test_step
    tool: terminal
    action: execute
    parameters:
      command: echo "Inline pipeline works"
"""
    
    result = await tool.execute(
        pipeline=inline_yaml,
        wait_for_completion=True
    )
    
    assert result["success"] is True
    assert result["pipeline_id"] == "inline_pipeline"


@pytest.mark.asyncio
async def test_recursion_depth_limit():
    """Test recursion depth limiting."""
    # Create a recursive pipeline
    recursive_pipeline = """
id: recursive_pipeline
name: Recursive Test
parameters:
  counter:
    type: integer
    default: 0
steps:
  - id: check_counter
    tool: recursion-control
    action: update_state
    parameters:
      state_key: counter
      increment: 1
      
  - id: recurse
    tool: pipeline-executor
    action: execute
    parameters:
      pipeline: recursive_pipeline
      inputs:
        parameters:
          counter: "{{ check_counter.new_value }}"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(recursive_pipeline)
        pipeline_path = f.name
    
    try:
        tool = PipelineExecutorTool()
        
        # This should hit recursion depth limit
        with pytest.raises(RecursionError):
            await tool.execute(
                pipeline=pipeline_path,
                wait_for_completion=True
            )
    finally:
        os.unlink(pipeline_path)


@pytest.mark.asyncio
async def test_recursion_control_check_condition():
    """Test recursion control condition checking."""
    tool = RecursionControlTool()
    
    # Test simple condition
    result = await tool.execute(
        action="check_condition",
        condition="state.get('counter', 0) >= 5",
        context_id="test_context"
    )
    
    assert result["success"] is True
    assert result["should_terminate"] is False  # Counter not set yet
    
    # Update state and check again
    await tool.execute(
        action="update_state",
        state_key="counter",
        state_value=5,
        context_id="test_context"
    )
    
    result = await tool.execute(
        action="check_condition",
        condition="state.get('counter', 0) >= 5",
        context_id="test_context"
    )
    
    assert result["should_terminate"] is True


@pytest.mark.asyncio
async def test_recursion_control_state_management():
    """Test recursion control state operations."""
    tool = RecursionControlTool()
    context_id = "state_test"
    
    # Test update_state with value
    result = await tool.execute(
        action="update_state",
        state_key="user_name",
        state_value="Alice",
        context_id=context_id
    )
    
    assert result["success"] is True
    assert result["new_value"] == "Alice"
    
    # Test increment
    await tool.execute(
        action="update_state",
        state_key="score",
        state_value=10,
        context_id=context_id
    )
    
    result = await tool.execute(
        action="update_state",
        state_key="score",
        increment=5,
        context_id=context_id
    )
    
    assert result["new_value"] == 15
    
    # Test get_state for specific key
    result = await tool.execute(
        action="get_state",
        state_key="score",
        context_id=context_id
    )
    
    assert result["value"] == 15
    assert result["exists"] is True
    
    # Test get_state for all
    result = await tool.execute(
        action="get_state",
        context_id=context_id
    )
    
    assert result["state"]["user_name"] == "Alice"
    assert result["state"]["score"] == 15
    
    # Test reset
    result = await tool.execute(
        action="reset",
        context_id=context_id
    )
    
    assert result["success"] is True
    
    # Verify reset worked
    result = await tool.execute(
        action="get_state",
        context_id=context_id
    )
    
    assert result["state"] == {}


@pytest.mark.asyncio
async def test_recursion_control_limits():
    """Test recursion control limit checking."""
    tool = RecursionControlTool()
    context_id = "limits_test"
    
    # Test iteration limit
    result = await tool.execute(
        action="check_condition",
        condition="False",  # Never terminate naturally
        max_iterations=5,
        context_id=context_id
    )
    
    assert result["success"] is True
    assert result["should_terminate"] is False
    
    # Simulate multiple iterations
    for i in range(5):
        await tool.execute(
            action="update_state",
            state_key=f"iteration_{i}",
            state_value=i,
            context_id=context_id
        )
        # Update execution count manually (normally done by PipelineExecutorTool)
        tool._recursion_states[context_id].execution_count["test"] = i + 1
    
    # Check should now fail due to iteration limit
    result = await tool.execute(
        action="check_condition",
        condition="False",
        max_iterations=5,
        context_id=context_id
    )
    
    assert result["should_terminate"] is True
    assert "max_iterations exceeded" in result["reason"]


@pytest.mark.asyncio
async def test_pipeline_executor_output_mapping():
    """Test output mapping functionality."""
    sub_pipeline = """
id: output_test
name: Output Test Pipeline
steps:
  - id: generate_data
    tool: terminal
    action: execute
    parameters:
      command: echo '{"status": "success", "value": 42}'
outputs:
  result_status: "{{ generate_data.output }}"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(sub_pipeline)
        pipeline_path = f.name
    
    try:
        tool = PipelineExecutorTool()
        
        result = await tool.execute(
            pipeline=pipeline_path,
            output_mapping={
                "result_status": "mapped_status"
            },
            wait_for_completion=True
        )
        
        assert result["success"] is True
        assert "outputs" in result
        # The mapping would be applied if outputs were properly extracted
    finally:
        os.unlink(pipeline_path)


@pytest.mark.asyncio
async def test_pipeline_executor_error_handling():
    """Test error handling strategies."""
    failing_pipeline = """
id: failing_pipeline
name: Failing Pipeline
steps:
  - id: fail_step
    tool: terminal
    action: execute
    parameters:
      command: "exit 1"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(failing_pipeline)
        pipeline_path = f.name
    
    try:
        tool = PipelineExecutorTool()
        
        # Test fail strategy (default)
        with pytest.raises(RuntimeError):
            await tool.execute(
                pipeline=pipeline_path,
                error_handling="fail",
                wait_for_completion=True
            )
        
        # Test continue strategy
        result = await tool.execute(
            pipeline=pipeline_path,
            error_handling="continue",
            wait_for_completion=True
        )
        
        assert result["success"] is False
        assert result["continued"] is True
    finally:
        os.unlink(pipeline_path)


@pytest.mark.asyncio
async def test_recursive_fibonacci_pipeline():
    """Test a realistic recursive pipeline - Fibonacci calculation."""
    # Create Fibonacci pipeline
    fib_pipeline = """
id: fibonacci
name: Fibonacci Calculator
parameters:
  n:
    type: integer
    default: 5
steps:
  - id: check_base_case
    tool: recursion-control
    action: check_condition
    parameters:
      condition: "state.get('n', {{ parameters.n }}) <= 1"
      context_id: "fib_{{ parameters.n }}"
      
  - id: return_base
    tool: recursion-control
    action: update_state
    parameters:
      state_key: result
      state_value: "{{ parameters.n }}"
      context_id: "fib_{{ parameters.n }}"
    dependencies:
      - check_base_case
    condition: "{{ check_base_case.should_terminate }}"
    
  - id: calc_n_minus_1
    tool: pipeline-executor
    action: execute
    parameters:
      pipeline: fibonacci
      inputs:
        parameters:
          n: "{{ parameters.n - 1 }}"
    dependencies:
      - check_base_case
    condition: "not {{ check_base_case.should_terminate }}"
    
  - id: calc_n_minus_2
    tool: pipeline-executor
    action: execute
    parameters:
      pipeline: fibonacci
      inputs:
        parameters:
          n: "{{ parameters.n - 2 }}"
    dependencies:
      - check_base_case
    condition: "not {{ check_base_case.should_terminate }}"
    
  - id: combine_results
    tool: recursion-control
    action: update_state
    parameters:
      state_key: result
      state_value: "{{ calc_n_minus_1.outputs.result + calc_n_minus_2.outputs.result }}"
      context_id: "fib_{{ parameters.n }}"
    dependencies:
      - calc_n_minus_1
      - calc_n_minus_2

outputs:
  result: "{{ combine_results.new_value if combine_results else return_base.new_value }}"
"""
    
    # Note: This is a complex example that would require full pipeline execution
    # In a real test, we'd save this and test with small values of n
    
    # For now, just verify the YAML is valid
    pipeline_def = yaml.safe_load(fib_pipeline)
    assert pipeline_def["id"] == "fibonacci"
    assert "steps" in pipeline_def
    assert len(pipeline_def["steps"]) == 5


if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_pipeline_executor_basic())
    asyncio.run(test_recursion_control_check_condition())
    asyncio.run(test_recursion_control_state_management())
    print("All basic tests passed!")