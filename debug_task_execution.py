#!/usr/bin/env python3
"""Debug task execution after model selection."""

import asyncio
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

async def test_task_execution():
    """Test executing a task directly."""
    print("Testing task execution...")
    
    try:
        from orchestrator import Orchestrator, init_models
        from orchestrator.core.task import Task, TaskStatus
        
        # Initialize orchestrator
        model_registry = init_models()
        orchestrator = Orchestrator(model_registry=model_registry)
        
        # Create a simple task
        task = Task(
            id="test_task",
            action="generate_text",
            parameters={
                'prompt': 'Say hello',
                'model': 'gpt-3.5-turbo',
                'max_tokens': 5
            }
        )
        
        print("Executing task...")
        start = time.time()
        
        try:
            # This is the method that should be hanging
            result = await asyncio.wait_for(
                orchestrator.execute_task(task),
                timeout=20.0
            )
            print(f"✅ Task execution successful in {time.time() - start:.2f}s")
            print(f"   Result: {result}")
            return True
            
        except asyncio.TimeoutError:
            print(f"⏱️  Task execution timed out after 20s")
            print(f"   Task status: {task.status}")
            return False
        
    except Exception as e:
        print(f"❌ Task execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_action_execution():
    """Test action execution through control system."""
    print("Testing action execution...")
    
    try:
        from orchestrator import Orchestrator, init_models
        
        # Initialize orchestrator  
        model_registry = init_models()
        orchestrator = Orchestrator(model_registry=model_registry)
        
        # Test action execution directly
        print("Testing generate_text action...")
        start = time.time()
        
        try:
            result = await asyncio.wait_for(
                orchestrator.control_system.execute_action(
                    action="generate_text",
                    parameters={
                        'prompt': 'Say hello',
                        'model': 'gpt-3.5-turbo', 
                        'max_tokens': 5
                    },
                    context={}
                ),
                timeout=15.0
            )
            print(f"✅ Action execution successful in {time.time() - start:.2f}s")
            print(f"   Result: {result}")
            return True
            
        except asyncio.TimeoutError:
            print(f"⏱️  Action execution timed out after 15s")
            return False
        
    except Exception as e:
        print(f"❌ Action execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🔍 Debugging task execution pipeline...")
    print("=" * 60)
    
    # Test action execution first (lower level)
    print("\n[1/2] Testing action execution...")
    success = await test_action_execution()
    if not success:
        print("❌ Action execution failed")
        return
    print("✅ Action execution passed")
    
    # Test task execution
    print("\n[2/2] Testing task execution...")
    success = await test_task_execution()
    if not success:
        print("❌ Task execution failed")
        return
    print("✅ Task execution passed")
    
    print("\n🎉 All task execution tests completed!")

if __name__ == "__main__":
    asyncio.run(main())