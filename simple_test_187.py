#!/usr/bin/env python3
"""Simple test for Issue 187 verification."""

import asyncio
from orchestrator.core.parallel_queue_task import ParallelQueueTask
from orchestrator.control_flow.parallel_queue_handler import ParallelQueueHandler
from orchestrator.control_flow.auto_resolver import ControlFlowAutoResolver
from orchestrator.core.loop_context import GlobalLoopContextManager
from orchestrator.models.model_registry import ModelRegistry

async def simple_test():
    """Simple test of parallel queue functionality."""
    
    print("Testing basic parallel queue functionality...")
    
    # Create handler
    model_registry = ModelRegistry()
    auto_resolver = ControlFlowAutoResolver(model_registry=model_registry)
    handler = ParallelQueueHandler(
        auto_resolver=auto_resolver,
        loop_context_manager=GlobalLoopContextManager(),
        model_registry=model_registry
    )
    
    # Create simple task
    task = ParallelQueueTask(
        id="simple_test",
        name="Simple Test",
        action="create_parallel_queue",
        on='["a", "b"]',  # Just 2 items
        max_parallel=1,
        action_loop=[{"action": "echo", "parameters": {"message": "Processing {{ item }}"}}]
    )
    
    try:
        # Test execution directly (this will call _generate_queue_items internally)
        result = await handler.execute_parallel_queue(task, {}, {})
        print(f"Total items: {result['total_items']}")
        print(f"Successful items: {result['successful_items']}")
        print(f"Results count: {len(result['results'])}")
        
        # Print actual results
        for i, res in enumerate(result['results']):
            print(f"Result {i}: item={res.get('item')}, status={res.get('status')}")
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(simple_test())
    if result:
        print("✅ Test completed")
    else:
        print("❌ Test failed")