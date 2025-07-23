"""Simple tests for pipeline recursion tools that don't require full orchestration."""

import asyncio
import pytest
import yaml

from src.orchestrator.tools.pipeline_recursion_tools import (
    PipelineExecutorTool,
    RecursionControlTool,
    RecursionContext,
)


@pytest.mark.asyncio
async def test_pipeline_executor_resolve_pipeline():
    """Test pipeline resolution from different formats."""
    tool = PipelineExecutorTool()

    # Test inline YAML resolution
    inline_yaml = """
id: test_pipeline
name: Test Pipeline
steps:
  - id: step1
    tool: test
    action: test
"""

    pipeline_def = tool._resolve_pipeline(inline_yaml)
    assert pipeline_def["id"] == "test_pipeline"
    assert "steps" in pipeline_def

    # Test YAML dict resolution
    yaml_dict = yaml.safe_load(inline_yaml)
    pipeline_def2 = tool._resolve_pipeline(yaml.dump(yaml_dict))
    assert pipeline_def2["id"] == "test_pipeline"


@pytest.mark.asyncio
async def test_recursion_context():
    """Test RecursionContext functionality."""
    context = RecursionContext(max_depth=5)

    # Test initial state
    assert context.depth == 0
    assert context.call_stack == []
    assert context.execution_count == {}

    # Test depth tracking
    context.depth = 3
    context.call_stack.append("pipeline1")
    context.call_stack.append("pipeline2")

    assert context.depth == 3
    assert len(context.call_stack) == 2

    # Test execution counting
    context.execution_count["pipeline1"] = 5
    assert context.execution_count["pipeline1"] == 5


@pytest.mark.asyncio
async def test_recursion_limits():
    """Test recursion limit checking."""
    tool = PipelineExecutorTool()
    context = RecursionContext(max_depth=3, max_executions_per_pipeline=5)

    # Test depth limit
    context.depth = 2
    tool._check_recursion_limits("test_pipeline", context)  # Should not raise

    context.depth = 3
    with pytest.raises(RecursionError, match="Maximum recursion depth"):
        tool._check_recursion_limits("test_pipeline", context)

    # Test execution count limit
    context.depth = 1
    context.execution_count["test_pipeline"] = 4
    tool._check_recursion_limits("test_pipeline", context)  # Should not raise

    context.execution_count["test_pipeline"] = 5
    with pytest.raises(RecursionError, match="exceeded maximum executions"):
        tool._check_recursion_limits("test_pipeline", context)


@pytest.mark.asyncio
async def test_context_merging():
    """Test context merging functionality."""
    tool = PipelineExecutorTool()

    parent_context = {"var1": "parent", "var2": "original"}
    child_inputs = {"var2": "child", "var3": "new"}

    # Test with inheritance
    merged = tool._merge_contexts(parent_context, child_inputs, inherit=True)
    assert merged["var1"] == "parent"
    assert merged["var2"] == "child"  # Child overrides parent
    assert merged["var3"] == "new"

    # Test without inheritance
    merged_no_inherit = tool._merge_contexts(parent_context, child_inputs, inherit=False)
    assert merged_no_inherit == child_inputs
    assert "var1" not in merged_no_inherit


@pytest.mark.asyncio
async def test_output_mapping():
    """Test output mapping functionality."""
    tool = PipelineExecutorTool()

    pipeline_outputs = {"result": "success", "data": {"value": 42}, "status": "completed"}

    output_mapping = {
        "result": "final_result",
        "data": "processed_data",
        "missing_key": "should_not_appear",
    }

    mapped = tool._map_outputs(pipeline_outputs, output_mapping)

    assert mapped["final_result"] == "success"
    assert mapped["processed_data"] == {"value": 42}
    assert "should_not_appear" not in mapped
    assert "status" not in mapped  # Not in mapping

    # Test empty mapping
    mapped_empty = tool._map_outputs(pipeline_outputs, {})
    assert mapped_empty == pipeline_outputs


@pytest.mark.asyncio
async def test_recursion_control_conditions():
    """Test recursion control condition evaluation."""
    tool = RecursionControlTool()
    context = RecursionContext()

    # Set up test state
    context.shared_state = {"counter": 5, "items": [1, 2, 3], "flag": True}
    context.depth = 2
    context.execution_count = {"pipeline1": 3, "pipeline2": 1}

    # Test various conditions
    assert tool._evaluate_condition("state.get('counter') == 5", context) is True
    assert tool._evaluate_condition("state.get('counter') > 10", context) is False
    assert tool._evaluate_condition("len(state.get('items', [])) == 3", context) is True
    assert tool._evaluate_condition("state.get('flag') and depth < 5", context) is True
    assert tool._evaluate_condition("sum(executions.values()) == 4", context) is True

    # Test with missing keys
    assert tool._evaluate_condition("state.get('missing', 0) == 0", context) is True

    # Test complex conditions
    assert (
        tool._evaluate_condition(
            "state.get('counter') >= 5 and len(state.get('items', [])) > 2", context
        )
        is True
    )


@pytest.mark.asyncio
async def test_recursion_control_state_operations():
    """Test recursion control state management."""
    tool = RecursionControlTool()

    # Test state initialization
    result = await tool.execute(action="get_state", context_id="test1")
    assert result["success"] is True
    assert result["state"] == {}

    # Test state update with value
    result = await tool.execute(
        action="update_state", state_key="name", state_value="test_value", context_id="test1"
    )
    assert result["success"] is True
    assert result["new_value"] == "test_value"

    # Test increment
    await tool.execute(
        action="update_state", state_key="counter", state_value=10, context_id="test1"
    )

    result = await tool.execute(
        action="update_state", state_key="counter", increment=5, context_id="test1"
    )
    assert result["new_value"] == 15

    # Test get specific key
    result = await tool.execute(action="get_state", state_key="counter", context_id="test1")
    assert result["value"] == 15
    assert result["exists"] is True

    # Test reset
    result = await tool.execute(action="reset", context_id="test1")
    assert result["success"] is True

    # Verify reset
    result = await tool.execute(action="get_state", context_id="test1")
    assert result["state"] == {}


@pytest.mark.asyncio
async def test_active_contexts():
    """Test getting active contexts."""
    tool = RecursionControlTool()

    # Initially no contexts
    assert tool.get_active_contexts() == []

    # Create some contexts
    await tool.execute(action="update_state", state_key="test", state_value=1, context_id="ctx1")

    await tool.execute(action="update_state", state_key="test", state_value=2, context_id="ctx2")

    active = tool.get_active_contexts()
    assert len(active) == 2
    assert "ctx1" in active
    assert "ctx2" in active

    # Test context info
    info = tool.get_context_info("ctx1")
    assert info is not None
    assert info["context_id"] == "ctx1"
    assert "test" in info["state_keys"]

    # Non-existent context
    assert tool.get_context_info("non_existent") is None


if __name__ == "__main__":
    asyncio.run(test_pipeline_executor_resolve_pipeline())
    asyncio.run(test_recursion_context())
    asyncio.run(test_recursion_control_state_operations())
    print("Simple tests passed!")
