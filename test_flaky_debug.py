"""Debug script to reproduce flaky test behavior."""

import asyncio
import sys
from orchestrator import init_models
from orchestrator.adapters.langgraph_adapter import LangGraphAdapter
from orchestrator.core.task import Task
from orchestrator.models.registry_singleton import get_model_registry

async def test_langgraph_execution():
    """Test LangGraph adapter execution."""
    # Initialize models
    init_models()
    registry = get_model_registry()
    
    print(f"Available models: {len(registry.list_models())}")
    
    # Create adapter
    config = {"name": "langgraph", "version": "1.0.0"}
    adapter = LangGraphAdapter(config, model_registry=registry)
    
    # Create task
    task = Task("test_task", "Test Task", "generate")
    task.parameters = {"prompt": "Say 'Hello, LangGraph!' in exactly 3 words"}
    
    # Execute task multiple times
    for i in range(10):
        print(f"\nRun {i+1}:")
        try:
            result = await adapter.execute_task(task, {})
            print(f"Result length: {len(result)}")
            print(f"Result: {result[:100] if result else 'EMPTY'}")
            if not result:
                print("ERROR: Empty result!")
                return False
        except Exception as e:
            print(f"ERROR: {e}")
            return False
    
    return True

async def main():
    """Main function."""
    success = await test_langgraph_execution()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())