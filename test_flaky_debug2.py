"""Debug script to reproduce flaky test behavior in suite context."""

import asyncio
import sys
from orchestrator import init_models
from orchestrator.adapters.langgraph_adapter import LangGraphAdapter
from orchestrator.core.task import Task
from orchestrator.models.registry_singleton import get_model_registry

async def run_other_tests():
    """Simulate running other tests that might affect state."""
    # Initialize models multiple times like different tests might
    for i in range(3):
        init_models()
        registry = get_model_registry()
        
        # Use models in different ways
        models = await registry.get_available_models()
        print(f"Other test {i+1}: {len(models)} models available")
        
        # Select models with different criteria
        model1 = await registry.select_model({"tasks": ["analyze"]})
        model2 = await registry.select_model({"tasks": ["code"]})
        
        # Generate some text to potentially exhaust resources
        if model1:
            try:
                await model1.generate("Test prompt", max_tokens=10)
            except:
                pass

async def test_langgraph_execution_in_suite():
    """Test LangGraph adapter execution after other tests."""
    # First run some other tests
    print("=== Running other tests first ===")
    await run_other_tests()
    
    print("\n=== Now running LangGraph test ===")
    
    # Now run the actual test
    registry = get_model_registry()
    print(f"Available models: {len(registry.list_models())}")
    
    # Create adapter
    config = {"name": "langgraph", "version": "1.0.0"}
    adapter = LangGraphAdapter(config, model_registry=registry)
    
    # Create task
    task = Task("test_task", "Test Task", "generate")
    task.parameters = {"prompt": "Say 'Hello, LangGraph!' in exactly 3 words"}
    
    # Execute task
    try:
        result = await adapter.execute_task(task, {})
        print(f"Result length: {len(result)}")
        print(f"Result: {result[:100] if result else 'EMPTY'}")
        if not result or len(result) == 0:
            print("ERROR: Empty result!")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

async def main():
    """Main function."""
    success = await test_langgraph_execution_in_suite()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())