#!/usr/bin/env python3
"""
Test script to verify Issue 187 implementation completeness.
This tests all the key requirements from the issue specification.
"""

import asyncio
import time
from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.core.parallel_queue_task import ParallelQueueTask
from orchestrator.control_flow.parallel_queue_handler import ParallelQueueHandler
from orchestrator.control_flow.auto_resolver import ControlFlowAutoResolver
from orchestrator.core.loop_context import GlobalLoopContextManager
from orchestrator.models.model_registry import ModelRegistry

async def test_issue_187_requirements():
    """Test all requirements from Issue 187."""
    
    print("🔍 Testing Issue 187: create_parallel_queue Action Implementation")
    print("=" * 70)
    
    # Test 1: YAML Compilation (from issue specification)
    print("\n1. Testing YAML Compilation with create_parallel_queue syntax...")
    
    yaml_content = '''
name: Issue 187 Verification Test
steps:
  - id: verify-sources
    create_parallel_queue:
      on: '["source1", "source2", "source3"]'
      max_parallel: 2
      action_loop:
        - action: debug
          parameters:
            message: "Processing source: {{ item }}"
            index: "{{ index }}"
    '''
    
    try:
        compiler = YAMLCompiler()
        pipeline = await compiler.compile(yaml_content)
        
        # Verify pipeline structure
        assert pipeline is not None, "Pipeline compilation failed"
        assert len(pipeline.tasks) == 1, f"Expected 1 task, got {len(pipeline.tasks)}"
        
        task = pipeline.tasks["verify-sources"]
        assert isinstance(task, ParallelQueueTask), f"Expected ParallelQueueTask, got {type(task)}"
        assert task.action == "create_parallel_queue", f"Expected create_parallel_queue action, got {task.action}"
        assert task.max_parallel == 2, f"Expected max_parallel=2, got {task.max_parallel}"
        assert len(task.action_loop) == 1, f"Expected 1 action in loop, got {len(task.action_loop)}"
        
        print("✅ YAML compilation successful")
        
    except Exception as e:
        print(f"❌ YAML compilation failed: {e}")
        return False
    
    # Test 2: Queue Generation 
    print("\n2. Testing Queue Generation...")
    
    try:
        model_registry = ModelRegistry()
        auto_resolver = ControlFlowAutoResolver(model_registry=model_registry)
        handler = ParallelQueueHandler(
            auto_resolver=auto_resolver,
            loop_context_manager=GlobalLoopContextManager(),
            model_registry=model_registry
        )
        
        # Test direct JSON list parsing
        task = ParallelQueueTask(
            id="test_queue_gen",
            name="Test Queue Generation",
            action="create_parallel_queue",
            on='["item1", "item2", "item3", "item4"]',
            max_parallel=2,
            action_loop=[{"action": "echo", "parameters": {"message": "Test {{ item }}"}}]
        )
        
        context = {}
        step_results = {}
        
        await handler._generate_queue_items(task, context, step_results)
        
        assert len(task.queue_items) == 4, f"Expected 4 queue items, got {len(task.queue_items)}"
        assert task.queue_items == ["item1", "item2", "item3", "item4"], f"Wrong queue items: {task.queue_items}"
        assert task.stats.queue_generation_time > 0, "Queue generation time not recorded"
        
        print("✅ Queue generation successful")
        
    except Exception as e:
        print(f"❌ Queue generation failed: {e}")
        return False
    
    # Test 3: Context Variables (from specification)
    print("\n3. Testing Context Variables ($item, $index, etc.)...")
    
    try:
        # Test context variable generation
        context_vars = task.get_context_variables(1)  # Second item (index 1)
        
        expected_vars = {
            "$item", "$index", "$queue", "$queue_size", "$is_first", "$is_last",
            "item", "index", "queue", "queue_size", "is_first", "is_last"  # Both syntaxes
        }
        
        for var in expected_vars:
            assert var in context_vars, f"Missing context variable: {var}"
        
        # Check specific values
        assert context_vars["$item"] == "item2", f"Wrong $item value: {context_vars['$item']}"
        assert context_vars["$index"] == 1, f"Wrong $index value: {context_vars['$index']}"
        assert context_vars["$is_first"] == False, f"Wrong $is_first value: {context_vars['$is_first']}"
        assert context_vars["$is_last"] == False, f"Wrong $is_last value: {context_vars['$is_last']}"
        assert context_vars["$queue_size"] == 4, f"Wrong $queue_size value: {context_vars['$queue_size']}"
        
        # Test first and last item
        first_vars = task.get_context_variables(0)
        last_vars = task.get_context_variables(3)
        
        assert first_vars["$is_first"] == True, "First item $is_first should be True"
        assert first_vars["$is_last"] == False, "First item $is_last should be False"
        assert last_vars["$is_first"] == False, "Last item $is_first should be False"
        assert last_vars["$is_last"] == True, "Last item $is_last should be True"
        
        print("✅ Context variables working correctly")
        
    except Exception as e:
        print(f"❌ Context variables failed: {e}")
        return False
    
    # Test 4: Parallel Execution with Concurrency Control
    print("\n4. Testing Parallel Execution with Concurrency Control...")
    
    try:
        start_time = time.time()
        result = await handler.execute_parallel_queue(task, context, step_results)
        execution_time = time.time() - start_time
        
        # Verify results structure
        assert "total_items" in result, "Missing total_items in result"
        assert "successful_items" in result, "Missing successful_items in result"
        assert "failed_items" in result, "Missing failed_items in result"
        assert "results" in result, "Missing results in result"
        assert "execution_stats" in result, "Missing execution_stats in result"
        
        # Verify execution success
        assert result["total_items"] == 4, f"Expected 4 total items, got {result['total_items']}"
        assert result["successful_items"] == 4, f"Expected 4 successful items, got {result['successful_items']}"
        assert result["failed_items"] == 0, f"Expected 0 failed items, got {result['failed_items']}"
        assert len(result["results"]) == 4, f"Expected 4 results, got {len(result['results'])}"
        
        # Verify concurrency was respected
        max_concurrent = result["execution_stats"]["max_concurrent_executions"]
        assert max_concurrent <= 2, f"Concurrency limit violated: {max_concurrent} > 2"
        
        # Verify all items were processed
        processed_items = [res["item"] for res in result["results"]]
        expected_items = ["item1", "item2", "item3", "item4"]
        assert set(processed_items) == set(expected_items), f"Items not processed correctly: {processed_items}"
        
        print(f"✅ Parallel execution successful (took {execution_time:.2f}s, max_concurrent: {max_concurrent})")
        
    except Exception as e:
        print(f"❌ Parallel execution failed: {e}")
        return False
    
    # Test 5: Error Handling (Edge Cases)
    print("\n5. Testing Error Handling...")
    
    try:
        # Test empty queue
        empty_task = ParallelQueueTask(
            id="empty_test",
            name="Empty Queue Test",
            action="create_parallel_queue",
            on='[]',
            max_parallel=2,
            action_loop=[{"action": "debug", "parameters": {"message": "Should not run"}}]
        )
        
        await handler._generate_queue_items(empty_task, {}, {})
        empty_result = await handler.execute_parallel_queue(empty_task, {}, {})
        
        assert empty_result["total_items"] == 0, "Empty queue should have 0 items"
        assert empty_result["successful_items"] == 0, "Empty queue should have 0 successful items"
        assert len(empty_result["results"]) == 0, "Empty queue should have 0 results"
        
        print("✅ Error handling working correctly")
        
    except Exception as e:
        print(f"❌ Error handling failed: {e}")
        return False
    
    # Test 6: Resource Management
    print("\n6. Testing Resource Management...")
    
    try:
        # Test tool specification
        tool_task = ParallelQueueTask(
            id="tool_test",
            name="Tool Resource Test", 
            action="create_parallel_queue",
            on='["tool1", "tool2"]',
            max_parallel=1,
            tool="debug_tool",  # Specify a tool
            action_loop=[{"action": "debug", "parameters": {"message": "Tool test: {{ item }}"}}]
        )
        
        tool_result = await handler.execute_parallel_queue(tool_task, {}, {})
        
        assert "resource_stats" in tool_result, "Missing resource_stats in result"
        assert tool_result["successful_items"] == 2, "Tool execution should succeed"
        
        print("✅ Resource management working correctly")
        
    except Exception as e:
        print(f"❌ Resource management failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED! Issue 187 implementation is COMPLETE")
    print("✅ YAML compilation with create_parallel_queue syntax")
    print("✅ Queue generation from expressions")  
    print("✅ Context variables ($item, $index, $is_first, $is_last, etc.)")
    print("✅ Parallel execution with concurrency control")
    print("✅ Error handling and edge cases")
    print("✅ Resource management and tool integration")
    print("✅ Result aggregation and statistics")
    print("✅ NO MOCKS - All real functionality")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_issue_187_requirements())
    exit(0 if success else 1)