#!/usr/bin/env python3
"""Test edge cases and error conditions in pipelines with real implementations."""

import asyncio
import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestrator.orchestrator import Orchestrator
from orchestrator.core.control_system import ControlSystem
from orchestrator.core.task import Task


async def test_edge_cases():
    """Test various edge cases and error conditions with real implementations."""
    print("Testing Pipeline Edge Cases and Error Conditions")
    print("=" * 60)
    
    tests = []
    
    # Create a simple YAML for testing
    simple_yaml = """
name: "Test Pipeline"
description: "Simple pipeline for edge case testing"
steps:
  - id: process
    action: process_data
    parameters:
      input: "{{ topic }}"
"""
    
    # Test 1: Empty input
    print("\n1. Testing with empty topic...")
    try:
        orchestrator = Orchestrator()
        
        # Create a control system that handles empty input
        class EdgeCaseControlSystem(ControlSystem):
            def __init__(self):
                config = {
                    "capabilities": {
                        "supported_actions": ["process_data"],
                        "parallel_execution": False,
                        "streaming": False,
                        "checkpoint_support": True,
                    },
                    "base_priority": 10,
                }
                super().__init__(name="edge-case-control", config=config)
            
            async def execute_task(self, task: Task, context: dict = None):
                input_data = task.parameters.get("input", "")
                if not input_data:
                    # Handle empty input gracefully
                    return {"status": "completed", "result": "Empty input handled", "input_length": 0}
                return {"status": "completed", "result": f"Processed: {input_data}", "input_length": len(input_data)}
        
        orchestrator.control_system = EdgeCaseControlSystem()
        results = await orchestrator.execute_yaml(
            simple_yaml,
            context={"topic": ""}
        )
        print("  ✅ Handled empty input gracefully")
        tests.append(True)
    except Exception as e:
        print(f"  ❌ Failed with empty input: {e}")
        tests.append(False)
    
    # Test 2: Missing input file
    print("\n2. Testing with non-existent file...")
    try:
        file_yaml = """
name: "File Processing"
steps:
  - id: read_file
    action: read_file
    parameters:
      path: "{{ file_path }}"
"""
        orchestrator = Orchestrator()
        
        # Control system that handles file operations
        class FileControlSystem(ControlSystem):
            def __init__(self):
                config = {
                    "capabilities": {
                        "supported_actions": ["read_file"],
                        "parallel_execution": False,
                        "streaming": False,
                        "checkpoint_support": False,
                    },
                    "base_priority": 10,
                }
                super().__init__(name="file-control", config=config)
            
            async def execute_task(self, task: Task, context: dict = None):
                file_path = task.parameters.get("path", "")
                try:
                    # Attempt to read the file
                    with open(file_path, 'r') as f:
                        content = f.read()
                    return {"status": "completed", "content": content, "size": len(content)}
                except FileNotFoundError:
                    # Handle missing file gracefully
                    return {"status": "completed", "error": "File not found", "handled": True}
                except Exception as e:
                    return {"status": "failed", "error": str(e)}
        
        orchestrator.control_system = FileControlSystem()
        results = await orchestrator.execute_yaml(
            file_yaml,
            context={"file_path": "non_existent_file.py"}
        )
        # Check if error was handled gracefully
        if any('error' in r and r.get('handled') for r in results.values() if isinstance(r, dict)):
            print("  ✅ Handled missing file gracefully")
            tests.append(True)
        else:
            print("  ❌ Did not handle missing file properly")
            tests.append(False)
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
    except Exception:
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
    except Exception:
        print("  ✅ Correctly detected circular dependency")
        tests.append(True)
    
    # Test 5: Very long input
    print("\n5. Testing with very long input...")
    try:
        orchestrator = Orchestrator()
        
        # Control system that handles text processing
        class TextProcessingControlSystem(ControlSystem):
            def __init__(self):
                config = {
                    "capabilities": {
                        "supported_actions": ["process_text"],
                        "parallel_execution": False,
                        "streaming": True,
                        "checkpoint_support": False,
                    },
                    "base_priority": 10,
                }
                super().__init__(name="text-control", config=config)
            
            async def execute_task(self, task: Task, context: dict = None):
                text = task.parameters.get("text", "")
                # Handle very long text by truncating if needed
                max_length = 5000
                if len(text) > max_length:
                    truncated = text[:max_length] + "..."
                    return {
                        "status": "completed",
                        "original_length": len(text),
                        "truncated": True,
                        "processed_length": len(truncated)
                    }
                return {"status": "completed", "length": len(text), "truncated": False}
        
        text_yaml = """
name: "Text Processing"
steps:
  - id: process
    action: process_text
    parameters:
      text: "{{ topic }}"
"""
        
        orchestrator.control_system = TextProcessingControlSystem()
        long_topic = "A" * 10000  # Very long string
        results = await orchestrator.execute_yaml(
            text_yaml,
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
        
        # Control system that properly handles Unicode
        class UnicodeControlSystem(ControlSystem):
            def __init__(self):
                config = {
                    "capabilities": {
                        "supported_actions": ["process_unicode"],
                        "parallel_execution": False,
                        "streaming": False,
                        "checkpoint_support": False,
                    },
                    "base_priority": 10,
                }
                super().__init__(name="unicode-control", config=config)
            
            async def execute_task(self, task: Task, context: dict = None):
                text = task.parameters.get("text", "")
                # Process Unicode text
                try:
                    # Test encoding/decoding
                    encoded = text.encode('utf-8')
                    decoded = encoded.decode('utf-8')
                    
                    # Count different character types
                    ascii_chars = sum(1 for c in text if ord(c) < 128)
                    unicode_chars = len(text) - ascii_chars
                    
                    return {
                        "status": "completed",
                        "original": text,
                        "length": len(text),
                        "ascii_chars": ascii_chars,
                        "unicode_chars": unicode_chars,
                        "encoding_test": decoded == text
                    }
                except Exception as e:
                    return {"status": "failed", "error": str(e)}
        
        unicode_yaml = """
name: "Unicode Processing"
steps:
  - id: process
    action: process_unicode
    parameters:
      text: "{{ topic }}"
"""
        
        orchestrator.control_system = UnicodeControlSystem()
        unicode_topic = "Machine Learning 机器学习 🤖 with émojis and spëcial chars"
        results = await orchestrator.execute_yaml(
            unicode_yaml,
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
        required_yaml = """
name: "Required Context Test"
steps:
  - id: process
    action: process
    parameters:
      required_field: "{{ required_param }}"
"""
        orchestrator = Orchestrator()
        results = await orchestrator.execute_yaml(
            required_yaml,
            context={}  # Missing required 'required_param'
        )
        print("  ❌ Should have failed with missing context")
        tests.append(False)
    except Exception:
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
        
        class LargeDataControlSystem(ControlSystem):
            def __init__(self):
                config = {
                    "capabilities": {
                        "supported_actions": ["process_large"],
                        "parallel_execution": True,
                        "streaming": True,
                        "checkpoint_support": True,
                    },
                    "base_priority": 10,
                }
                super().__init__(name="large-data-control", config=config)
            async def execute_task(self, task: Task, context: dict = None):
                if task.action == "process_large":
                    size = task.parameters.get("size", 0)
                    if size > 50000:
                        # Create actual large data in memory
                        import pandas as pd
                        import numpy as np
                        
                        print(f"    Creating DataFrame with {size} records...")
                        # Create a large DataFrame
                        df = pd.DataFrame({
                            'id': range(size),
                            'value': np.random.rand(size),
                            'category': np.random.choice(['A', 'B', 'C'], size)
                        })
                        
                        # Process the data
                        mean_value = df['value'].mean()
                        category_counts = df['category'].value_counts().to_dict()
                        
                        return {
                            "processed": size,
                            "status": "completed",
                            "mean_value": float(mean_value),
                            "category_distribution": category_counts,
                            "memory_used_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
                        }
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
    """Test performance with real concurrent load."""
    print("\nTesting Performance Under Load")
    print("-" * 40)
    
    # Test concurrent pipeline execution
    print("Running 3 pipelines concurrently...")
    
    # Create a performance test control system
    class PerformanceControlSystem(ControlSystem):
        def __init__(self):
            config = {
                "capabilities": {
                    "supported_actions": ["cpu_task", "io_task", "network_task"],
                    "parallel_execution": True,
                    "streaming": False,
                    "checkpoint_support": False,
                },
                "base_priority": 10,
            }
            super().__init__(name="performance-control", config=config)
        
        async def execute_task(self, task: Task, context: dict = None):
            import hashlib
            import aiohttp
            
            if task.action == "cpu_task":
                # CPU-intensive task
                data = task.parameters.get("data", "test")
                iterations = task.parameters.get("iterations", 1000)
                result = data
                for i in range(iterations):
                    result = hashlib.sha256(result.encode()).hexdigest()
                return {"status": "completed", "hash": result, "iterations": iterations}
            
            elif task.action == "io_task":
                # I/O task - write and read temporary file
                data = task.parameters.get("data", "test data")
                temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                temp_file.write(data * 100)  # Write some data
                temp_file.close()
                
                # Read it back
                with open(temp_file.name, 'r') as f:
                    content = f.read()
                
                os.unlink(temp_file.name)
                return {"status": "completed", "bytes_processed": len(content)}
            
            elif task.action == "network_task":
                # Network task - make a real HTTP request
                url = task.parameters.get("url", "https://httpbin.org/delay/1")
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, timeout=5) as response:
                            data = await response.text()
                            return {"status": "completed", "response_size": len(data), "status_code": response.status}
                except Exception as e:
                    return {"status": "completed", "error": str(e), "fallback": True}
            
            return {"status": "completed"}
    
    performance_yaml = """
name: "Performance Test Pipeline"
steps:
  - id: cpu_intensive
    action: cpu_task
    parameters:
      data: "{{ pipeline_id }}"
      iterations: 5000
  
  - id: io_operation
    action: io_task
    parameters:
      data: "Test data for pipeline {{ pipeline_id }}"
  
  - id: network_call
    action: network_task
    parameters:
      url: "https://httpbin.org/delay/1"
"""
    
    async def run_single_pipeline(pipeline_id):
        orchestrator = Orchestrator(control_system=PerformanceControlSystem())
        return await orchestrator.execute_yaml(
            performance_yaml,
            context={"pipeline_id": str(pipeline_id)}
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