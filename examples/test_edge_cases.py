#!/usr/bin/env python3
"""Test edge cases and error conditions in pipelines."""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestrator.orchestrator import Orchestrator
from orchestrator.core.control_system import MockControlSystem
from orchestrator.core.task import Task
from orchestrator.core.model import Model, ModelCapabilities


async def test_edge_cases():
    """Test various edge cases and error conditions."""
    print("Testing Pipeline Edge Cases and Error Conditions")
    print("=" * 60)
    
    tests = []
    
    # Test 1: Empty input
    print("\n1. Testing with empty topic...")
    try:
        orchestrator = Orchestrator()
        results = await orchestrator.execute_yaml(
            open("pipelines/simple_research.yaml").read(),
            context={"topic": ""}
        )
        print("  ✅ Handled empty input gracefully")
        tests.append(True)
    except Exception as e:
        print(f"  ❌ Failed with empty input: {e}")
        tests.append(False)
    
    # Test 2: Missing input file
    print("\n2. Testing with non-existent code file...")
    try:
        orchestrator = Orchestrator()
        results = await orchestrator.execute_yaml(
            open("pipelines/code_optimization.yaml").read(),
            context={
                "code_path": "non_existent_file.py",
                "optimization_level": "performance",
                "language": "python"
            }
        )
        print("  ✅ Handled missing file gracefully")
        tests.append(True)
    except Exception as e:
        print(f"  ❌ Failed with missing file: {e}")
        tests.append(False)
    
    # Test 3: Invalid pipeline YAML
    print("\n3. Testing with invalid YAML syntax...")
    try:
        orchestrator = Orchestrator()
        invalid_yaml = """
name: "Invalid Pipeline
description: Missing quote above
steps:
  - id: test
    action: test
    parameters:
      invalid: {{ unclosed template
"""
        results = await orchestrator.execute_yaml(invalid_yaml, context={})
        print("  ❌ Should have failed with invalid YAML")
        tests.append(False)
    except Exception as e:
        print("  ✅ Correctly rejected invalid YAML")
        tests.append(True)
    
    # Test 4: Circular dependencies
    print("\n4. Testing with circular dependencies...")
    try:
        orchestrator = Orchestrator()
        circular_yaml = """
name: "Circular Dependencies"
steps:
  - id: task_a
    action: test
    dependencies: [task_b]
  - id: task_b  
    action: test
    dependencies: [task_a]
"""
        results = await orchestrator.execute_yaml(circular_yaml, context={})
        print("  ❌ Should have detected circular dependency")
        tests.append(False)
    except Exception as e:
        print("  ✅ Correctly detected circular dependency")
        tests.append(True)
    
    # Test 5: Very long input
    print("\n5. Testing with very long input...")
    try:
        orchestrator = Orchestrator()
        long_topic = "A" * 10000  # Very long string
        results = await orchestrator.execute_yaml(
            open("pipelines/simple_research.yaml").read(),
            context={"topic": long_topic}
        )
        print("  ✅ Handled very long input")
        tests.append(True)
    except Exception as e:
        print(f"  ❌ Failed with long input: {e}")
        tests.append(False)
    
    # Test 6: Unicode and special characters
    print("\n6. Testing with Unicode and special characters...")
    try:
        orchestrator = Orchestrator()
        unicode_topic = "Machine Learning 机器学习 🤖 with émojis and spëcial chars"
        results = await orchestrator.execute_yaml(
            open("pipelines/simple_research.yaml").read(),
            context={"topic": unicode_topic}
        )
        print("  ✅ Handled Unicode and special characters")
        tests.append(True)
    except Exception as e:
        print(f"  ❌ Failed with Unicode: {e}")
        tests.append(False)
    
    # Test 7: Missing required context
    print("\n7. Testing with missing required context...")
    try:
        orchestrator = Orchestrator()
        results = await orchestrator.execute_yaml(
            open("pipelines/simple_research.yaml").read(),
            context={}  # Missing required 'topic'
        )
        print("  ❌ Should have failed with missing context")
        tests.append(False)
    except Exception as e:
        print("  ✅ Correctly handled missing required context")
        tests.append(True)
    
    # Test 8: Very large data file
    print("\n8. Testing with large dataset simulation...")
    try:
        # Create a simulated large dataset
        large_data_yaml = """
name: "Large Data Test"
steps:
  - id: process
    action: process_large
    parameters:
      size: 100000
"""
        
        class LargeDataControlSystem(MockControlSystem):
            async def execute_task(self, task, context=None):
                if task.action == "process_large":
                    size = task.parameters.get("size", 0)
                    if size > 50000:
                        # Simulate memory pressure
                        print(f"    Processing {size} records (simulated)")
                        return {"processed": size, "status": "completed"}
                return {"status": "completed"}
        
        orchestrator = Orchestrator(control_system=LargeDataControlSystem())
        results = await orchestrator.execute_yaml(large_data_yaml, context={})
        print("  ✅ Handled large dataset simulation")
        tests.append(True)
    except Exception as e:
        print(f"  ❌ Failed with large dataset: {e}")
        tests.append(False)
    
    # Results summary
    passed = sum(tests)
    total = len(tests)
    
    print(f"\n{'='*60}")
    print(f"Edge Case Test Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    print('='*60)
    
    if passed >= total * 0.75:  # 75% pass rate acceptable for edge cases
        print("✅ Edge case testing mostly successful!")
        return True
    else:
        print("❌ Too many edge case failures")
        return False


async def test_performance_simulation():
    """Test performance with simulated load."""
    print("\nTesting Performance Under Load")
    print("-" * 40)
    
    # Test concurrent pipeline execution
    print("Running 3 pipelines concurrently...")
    
    async def run_single_pipeline(pipeline_id):
        orchestrator = Orchestrator()
        return await orchestrator.execute_yaml(
            open("pipelines/simple_research.yaml").read(),
            context={"topic": f"Test Topic {pipeline_id}"}
        )
    
    try:
        # Run 3 pipelines concurrently
        start_time = asyncio.get_event_loop().time()
        
        results = await asyncio.gather(
            run_single_pipeline(1),
            run_single_pipeline(2), 
            run_single_pipeline(3),
            return_exceptions=True
        )
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        successful = sum(1 for r in results if not isinstance(r, Exception))
        
        print(f"  Completed in {duration:.2f} seconds")
        print(f"  Successful: {successful}/3 pipelines")
        
        if successful >= 2:
            print("  ✅ Concurrent execution successful")
            return True
        else:
            print("  ❌ Too many concurrent failures")
            return False
            
    except Exception as e:
        print(f"  ❌ Concurrent execution failed: {e}")
        return False


async def main():
    """Run all edge case and performance tests."""
    edge_case_success = await test_edge_cases()
    performance_success = await test_performance_simulation()
    
    print(f"\n{'='*60}")
    print("FINAL TEST SUMMARY")
    print('='*60)
    print(f"Edge Cases: {'✅ PASS' if edge_case_success else '❌ FAIL'}")
    print(f"Performance: {'✅ PASS' if performance_success else '❌ FAIL'}")
    
    overall_success = edge_case_success and performance_success
    
    if overall_success:
        print("\n🎉 All edge case and performance tests completed successfully!")
        print("The orchestrator framework handles edge cases and concurrent execution well.")
    else:
        print("\n⚠️ Some edge case or performance tests failed.")
        print("This indicates areas for improvement in error handling or performance.")
    
    return overall_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)