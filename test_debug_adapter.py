"""Test to debug the adapter issue."""

import asyncio
import pytest

@pytest.mark.asyncio 
async def test_langgraph_adapter_debug(populated_model_registry):
    """Test LangGraphAdapter task execution with extra debugging."""
    from orchestrator.adapters.langgraph_adapter import LangGraphAdapter
    from orchestrator.core.task import Task
    
    registry = populated_model_registry
    available_models = registry.list_models()
    print(f"\n[DEBUG] Available models: {len(available_models)}")
    
    config = {"name": "langgraph", "version": "1.0.0"}
    adapter = LangGraphAdapter(config, model_registry=registry)
    
    task = Task("test_task", "Test Task", "generate")
    task.parameters = {"prompt": "Say 'Hello, LangGraph!' in exactly 3 words"}
    
    print(f"[DEBUG] Task action: {task.action}")
    print(f"[DEBUG] Task parameters: {task.parameters}")
    
    try:
        # Let's check the internal execution control
        print(f"[DEBUG] Adapter execution_control: {adapter.execution_control}")
        print(f"[DEBUG] Registry in control: {adapter.execution_control.model_registry}")
        
        # Execute task
        result = await adapter.execute_task(task, {})
        
        print(f"[DEBUG] Result type: {type(result)}")
        print(f"[DEBUG] Result length: {len(result) if isinstance(result, str) else 'N/A'}")
        print(f"[DEBUG] Result content: {repr(result)}")
        
        # Verify we got actual AI-generated response
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert len(result) > 5, f"Result too short: {repr(result)}"
        
    except Exception as e:
        print(f"[DEBUG] Exception type: {type(e)}")
        print(f"[DEBUG] Exception: {e}")
        import traceback
        traceback.print_exc()
        raise