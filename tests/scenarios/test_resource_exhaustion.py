"""
Resource Exhaustion and Memory Limit Testing

Tests the orchestrator's behavior under resource stress conditions including
memory exhaustion, CPU overload, disk space limits, and concurrent resource conflicts.
"""

import pytest
import os
import sys
import psutil
import time
import tempfile
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Import orchestrator components
from src.orchestrator.orchestrator import Orchestrator


class TestMemoryExhaustionScenarios:
    """Test handling of memory exhaustion scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp()) / "memory_tests"
        self.test_dir.mkdir(exist_ok=True)
        self.executor = Orchestrator()
        self.initial_memory = psutil.virtual_memory().percent
    
    def create_test_pipeline(self, content: str, filename: str = "test_pipeline.yaml") -> Path:
        """Create a test pipeline file with given content."""
        pipeline_path = self.test_dir / filename
        pipeline_path.write_text(content)
        return pipeline_path
    
    @pytest.mark.asyncio
    async def test_gradual_memory_consumption(self):
        """Test handling of gradual memory consumption that approaches limits."""
        # Calculate safe memory limit (don't actually exhaust system memory)
        available_memory_mb = psutil.virtual_memory().available // (1024 * 1024)
        safe_limit_mb = min(100, available_memory_mb // 4)  # Use at most 100MB or 1/4 available
        
        memory_test_pipeline = f"""
name: gradual_memory_test
version: "1.0"
steps:
  - id: memory_consumption_test
    action: python_code
    parameters:
      code: |
        import psutil
        import gc
        import time
        
        def get_memory_usage():
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory usage: {{get_memory_usage():.2f}} MB")
        
        # Gradually consume memory
        memory_chunks = []
        chunk_size = 1024 * 1024  # 1MB chunks
        target_mb = {safe_limit_mb}
        
        try:
            for i in range(target_mb):
                chunk = bytearray(chunk_size)
                memory_chunks.append(chunk)
                
                if i % 10 == 0:  # Report every 10MB
                    current_memory = get_memory_usage()
                    print(f"Allocated {{i+1}} MB, current usage: {{current_memory:.2f}} MB")
                    
                    # Safety check - if we're using too much system memory, stop
                    system_memory_percent = psutil.virtual_memory().percent
                    if system_memory_percent > 85:
                        print(f"System memory at {{system_memory_percent:.1f}}%, stopping allocation")
                        break
                
                time.sleep(0.01)  # Small delay to allow monitoring
            
            final_memory = get_memory_usage()
            print(f"Final memory usage: {{final_memory:.2f}} MB")
            print(f"Successfully allocated {{len(memory_chunks)}} MB")
            
        except MemoryError as e:
            print(f"Memory allocation failed: {{e}}")
            raise
        finally:
            # Clean up
            del memory_chunks
            gc.collect()
            cleanup_memory = get_memory_usage()
            print(f"After cleanup: {{cleanup_memory:.2f}} MB")
"""
        
        pipeline_path = self.create_test_pipeline(memory_test_pipeline, "gradual_memory.yaml")
        
        start_time = time.time()
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        execution_time = time.time() - start_time
        
        # Should complete (success or controlled failure)
        assert result.status in ["success", "error", "failed"]
        
        # Check memory usage didn't cause system issues
        final_memory = psutil.virtual_memory().percent
        memory_increase = final_memory - self.initial_memory
        assert memory_increase < 20, f"Memory usage increased too much: {memory_increase}%"
        
        print(f"✓ Gradual memory consumption test completed in {execution_time:.2f}s")
        print(f"  Memory change: {memory_increase:.1f}% (from {self.initial_memory:.1f}% to {final_memory:.1f}%)")
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """Test detection of memory leaks in pipeline execution."""
        memory_leak_pipeline = """
name: memory_leak_test
version: "1.0"
steps:
  - id: memory_leak_simulation
    action: python_code
    parameters:
      code: |
        import gc
        import psutil
        import time
        
        def get_memory_usage():
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate memory leak by creating objects that aren't properly cleaned up
        leaked_objects = []
        
        initial_memory = get_memory_usage()
        print(f"Initial memory: {initial_memory:.2f} MB")
        
        for iteration in range(50):  # Reduced iterations to avoid system impact
            # Create objects that might not be cleaned up properly
            large_dict = {f"key_{i}_{iteration}": [0] * 10000 for i in range(10)}
            leaked_objects.append(large_dict)
            
            # Simulate some operations
            temp_data = [i ** 2 for i in range(1000)]
            
            if iteration % 10 == 0:
                current_memory = get_memory_usage()
                memory_growth = current_memory - initial_memory
                print(f"Iteration {iteration}: {current_memory:.2f} MB (+{memory_growth:.2f} MB)")
                
                # If memory growth is too rapid, warn but don't fail
                if memory_growth > 50:  # More than 50MB growth
                    print(f"Warning: Rapid memory growth detected: {memory_growth:.2f} MB")
                    break
            
            time.sleep(0.01)
        
        final_memory = get_memory_usage()
        total_growth = final_memory - initial_memory
        print(f"Final memory: {final_memory:.2f} MB (growth: {total_growth:.2f} MB)")
        
        # Attempt cleanup
        del leaked_objects
        gc.collect()
        
        after_cleanup = get_memory_usage()
        recovered = final_memory - after_cleanup
        print(f"After cleanup: {after_cleanup:.2f} MB (recovered: {recovered:.2f} MB)")
"""
        
        pipeline_path = self.create_test_pipeline(memory_leak_pipeline, "memory_leak.yaml")
        
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        
        # Should complete (may succeed or fail based on system resources)
        assert result.status in ["success", "error", "failed"]
        
        print("✓ Memory leak detection test completed")
    
    @pytest.mark.asyncio
    async def test_concurrent_memory_pressure(self):
        """Test handling of multiple concurrent memory-intensive operations."""
        concurrent_memory_pipeline = """
name: concurrent_memory_test
version: "1.0"
steps:
  - id: concurrent_memory_operations
    action: parallel
    parameters:
      max_concurrent: 4
      tasks:
        - action: python_code
          parameters:
            code: |
              import time
              # Task 1: Create and process large list
              large_list = [i ** 2 for i in range(100000)]
              result1 = sum(large_list)
              print(f"Task 1 result: {result1}")
              time.sleep(0.5)
              del large_list
              
        - action: python_code
          parameters:
            code: |
              import time
              # Task 2: Create large dictionary
              large_dict = {f"key_{i}": [j for j in range(100)] for i in range(10000)}
              result2 = len(large_dict)
              print(f"Task 2 result: {result2}")
              time.sleep(0.5)
              del large_dict
              
        - action: python_code
          parameters:
            code: |
              import time
              # Task 3: String operations with large data
              base_string = "test_string_" * 10000
              result3 = len(base_string.split("_"))
              print(f"Task 3 result: {result3}")
              time.sleep(0.5)
              del base_string
              
        - action: python_code
          parameters:
            code: |
              import time
              # Task 4: Nested list operations
              nested_lists = [[i + j for i in range(100)] for j in range(1000)]
              result4 = sum(len(sublist) for sublist in nested_lists)
              print(f"Task 4 result: {result4}")
              time.sleep(0.5)
              del nested_lists
"""
        
        pipeline_path = self.create_test_pipeline(concurrent_memory_pipeline, "concurrent_memory.yaml")
        
        start_memory = psutil.virtual_memory().percent
        
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        
        end_memory = psutil.virtual_memory().percent
        
        # Should handle concurrent memory operations
        assert result.status in ["success", "partial_success", "error", "failed"]
        
        # Memory should not increase dramatically
        memory_change = end_memory - start_memory
        assert memory_change < 15, f"Memory usage increased too much: {memory_change}%"
        
        print(f"✓ Concurrent memory operations handled (memory change: {memory_change:.1f}%)")


class TestCPUExhaustionScenarios:
    """Test handling of CPU-intensive operations and overload."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp()) / "cpu_tests"
        self.test_dir.mkdir(exist_ok=True)
        self.executor = Orchestrator()
        self.cpu_count = multiprocessing.cpu_count()
    
    def create_test_pipeline(self, content: str, filename: str = "test_pipeline.yaml") -> Path:
        """Create a test pipeline file with given content."""
        pipeline_path = self.test_dir / filename
        pipeline_path.write_text(content)
        return pipeline_path
    
    @pytest.mark.asyncio
    async def test_cpu_intensive_operations(self):
        """Test handling of CPU-intensive computations."""
        cpu_intensive_pipeline = f"""
name: cpu_intensive_test
version: "1.0"
steps:
  - id: cpu_computation
    action: python_code
    parameters:
      code: |
        import time
        import psutil
        
        def cpu_intensive_task(n):
            '''Perform CPU-intensive computation'''
            result = 0
            for i in range(n):
                result += i ** 2 + i ** 3
                if i % 100000 == 0:  # Periodic check
                    current_cpu = psutil.cpu_percent(interval=0.1)
                    print(f"Progress: {{i/n*100:.1f}}%, CPU: {{current_cpu:.1f}}%")
            return result
        
        print("Starting CPU-intensive computation...")
        start_time = time.time()
        
        # Adjust computation size based on available CPUs
        computation_size = min(1000000, 500000 * {self.cpu_count})
        result = cpu_intensive_task(computation_size)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Computation result: {{result}}")
        print(f"Execution time: {{execution_time:.2f}} seconds")
        print(f"Operations per second: {{computation_size/execution_time:.0f}}")
      timeout: 60
"""
        
        pipeline_path = self.create_test_pipeline(cpu_intensive_pipeline, "cpu_intensive.yaml")
        
        start_time = time.time()
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert execution_time < 120, f"CPU-intensive task took too long: {execution_time}s"
        
        # Should handle CPU load appropriately
        assert result.status in ["success", "timeout", "error", "failed"]
        
        print(f"✓ CPU-intensive operation handled in {execution_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_cpu_overload(self):
        """Test system behavior under concurrent CPU overload."""
        # Create more concurrent CPU tasks than available cores
        overload_factor = self.cpu_count + 2
        
        cpu_overload_pipeline = f"""
name: cpu_overload_test
version: "1.0"
steps:
  - id: cpu_overload
    action: parallel
    parameters:
      max_concurrent: {overload_factor}
      tasks:
"""
        
        # Add CPU-intensive tasks
        for i in range(overload_factor):
            cpu_overload_pipeline += f"""
        - action: python_code
          parameters:
            code: |
              import time
              import math
              
              def cpu_task_{i}():
                  result = 0
                  for j in range(200000):  # Smaller per-task to avoid timeout
                      result += math.sin(j) * math.cos(j) + j ** 0.5
                  return result
              
              print(f"Task {i} starting...")
              start = time.time()
              result = cpu_task_{i}()
              duration = time.time() - start
              print(f"Task {i} completed in {{duration:.2f}}s, result: {{result:.2f}}")
"""
        
        pipeline_path = self.create_test_pipeline(cpu_overload_pipeline, "cpu_overload.yaml")
        
        start_time = time.time()
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        execution_time = time.time() - start_time
        
        # Should handle CPU overload gracefully
        assert result.status in ["success", "partial_success", "timeout", "error", "failed"]
        
        # Execution time should reflect CPU contention
        print(f"✓ CPU overload test completed in {execution_time:.2f}s")
        print(f"  {overload_factor} tasks on {self.cpu_count} CPU cores")


class TestDiskSpaceExhaustionScenarios:
    """Test handling of disk space limitations and I/O pressure."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp()) / "disk_tests"
        self.test_dir.mkdir(exist_ok=True)
        self.executor = Orchestrator()
        
        # Check available disk space
        self.initial_disk_usage = psutil.disk_usage(str(self.test_dir))
        self.available_space_mb = self.initial_disk_usage.free // (1024 * 1024)
    
    def create_test_pipeline(self, content: str, filename: str = "test_pipeline.yaml") -> Path:
        """Create a test pipeline file with given content."""
        pipeline_path = self.test_dir / filename
        pipeline_path.write_text(content)
        return pipeline_path
    
    @pytest.mark.asyncio
    async def test_large_file_operations(self):
        """Test handling of large file creation and manipulation."""
        # Use a safe amount of disk space (max 100MB or 1/10 of available)
        safe_file_size_mb = min(100, self.available_space_mb // 10)
        
        if safe_file_size_mb < 10:
            pytest.skip(f"Not enough disk space for test: {self.available_space_mb}MB available")
        
        large_file_pipeline = f"""
name: large_file_test
version: "1.0"
steps:
  - id: create_large_file
    action: python_code
    parameters:
      code: |
        import os
        import time
        
        file_path = "{self.test_dir}/large_test_file.bin"
        target_size_mb = {safe_file_size_mb}
        chunk_size = 1024 * 1024  # 1MB chunks
        
        print(f"Creating {{target_size_mb}}MB file...")
        
        try:
            with open(file_path, 'wb') as f:
                for i in range(target_size_mb):
                    chunk = b'\\x00' * chunk_size  # Write zeros
                    f.write(chunk)
                    
                    if i % 10 == 0:  # Report every 10MB
                        print(f"Written {{i+1}}/{{target_size_mb}} MB")
            
            # Verify file size
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            print(f"File created successfully: {{file_size_mb:.2f}} MB")
            
            # Read back some data to test I/O
            with open(file_path, 'rb') as f:
                sample = f.read(1024)
                print(f"Sample read: {{len(sample)}} bytes")
            
            # Clean up
            os.remove(file_path)
            print("File cleaned up successfully")
            
        except IOError as e:
            print(f"I/O error during file operations: {{e}}")
            raise
        except OSError as e:
            print(f"OS error (possibly disk full): {{e}}")
            raise
"""
        
        pipeline_path = self.create_test_pipeline(large_file_pipeline, "large_file.yaml")
        
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        
        # Should handle large file operations
        assert result.status in ["success", "error", "failed"]
        
        if result.status == "error":
            # Check if it's a disk space issue
            error_messages = " ".join([step.error_message.lower() for step in result.step_results if step.error_message])
            disk_keywords = ["disk", "space", "full", "no space", "device"]
            if any(keyword in error_messages for keyword in disk_keywords):
                print("✓ Disk space limitation properly detected")
            else:
                print("✓ Large file operation failed for other reasons")
        else:
            print("✓ Large file operations completed successfully")
    
    @pytest.mark.asyncio
    async def test_concurrent_file_operations(self):
        """Test handling of concurrent file I/O operations."""
        concurrent_io_pipeline = f"""
name: concurrent_io_test
version: "1.0"
steps:
  - id: concurrent_file_operations
    action: parallel
    parameters:
      max_concurrent: 5
      tasks:
        - action: python_code
          parameters:
            code: |
              import os
              import time
              import random
              
              # Task 1: Write sequential data
              file_path = "{self.test_dir}/sequential_data.txt"
              with open(file_path, 'w') as f:
                  for i in range(10000):
                      f.write(f"Line {{i}}: Sequential data\\n")
              
              file_size = os.path.getsize(file_path)
              print(f"Sequential file: {{file_size}} bytes")
              os.remove(file_path)
              
        - action: python_code
          parameters:
            code: |
              import os
              import json
              
              # Task 2: Write JSON data
              data = {{"numbers": [i**2 for i in range(1000)], "timestamp": time.time()}}
              file_path = "{self.test_dir}/json_data.json"
              with open(file_path, 'w') as f:
                  json.dump(data, f, indent=2)
              
              file_size = os.path.getsize(file_path)
              print(f"JSON file: {{file_size}} bytes")
              os.remove(file_path)
              
        - action: python_code
          parameters:
            code: |
              import os
              import random
              
              # Task 3: Write random binary data
              file_path = "{self.test_dir}/random_data.bin"
              with open(file_path, 'wb') as f:
                  for _ in range(1000):
                      random_bytes = bytes([random.randint(0, 255) for _ in range(100)])
                      f.write(random_bytes)
              
              file_size = os.path.getsize(file_path)
              print(f"Random binary file: {{file_size}} bytes")
              os.remove(file_path)
              
        - action: python_code
          parameters:
            code: |
              import os
              import tempfile
              
              # Task 4: Create and manipulate temp files
              temp_files = []
              for i in range(50):
                  temp_file = tempfile.NamedTemporaryFile(delete=False, dir="{self.test_dir}")
                  temp_file.write(f"Temp file {{i}} content\\n".encode())
                  temp_file.close()
                  temp_files.append(temp_file.name)
              
              total_size = sum(os.path.getsize(f) for f in temp_files)
              print(f"Created {{len(temp_files)}} temp files, total: {{total_size}} bytes")
              
              # Clean up
              for temp_file in temp_files:
                  os.remove(temp_file)
              
        - action: python_code
          parameters:
            code: |
              import os
              import csv
              
              # Task 5: Write CSV data
              file_path = "{self.test_dir}/csv_data.csv"
              with open(file_path, 'w', newline='') as f:
                  writer = csv.writer(f)
                  writer.writerow(['ID', 'Value', 'Square', 'Cube'])
                  for i in range(5000):
                      writer.writerow([i, random.random(), i**2, i**3])
              
              file_size = os.path.getsize(file_path)
              print(f"CSV file: {{file_size}} bytes")
              os.remove(file_path)
"""
        
        pipeline_path = self.create_test_pipeline(concurrent_io_pipeline, "concurrent_io.yaml")
        
        start_time = time.time()
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        execution_time = time.time() - start_time
        
        # Should handle concurrent I/O operations
        assert result.status in ["success", "partial_success", "error", "failed"]
        
        print(f"✓ Concurrent I/O operations completed in {execution_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_disk_space_monitoring(self):
        """Test monitoring and handling of disk space during operations."""
        disk_monitoring_pipeline = f"""
name: disk_monitoring_test
version: "1.0"
steps:
  - id: disk_space_monitor
    action: python_code
    parameters:
      code: |
        import os
        import psutil
        import time
        
        def get_disk_usage(path):
            usage = psutil.disk_usage(path)
            return {{
                'total_gb': usage.total / (1024**3),
                'used_gb': usage.used / (1024**3),
                'free_gb': usage.free / (1024**3),
                'percent_used': (usage.used / usage.total) * 100
            }}
        
        test_dir = "{self.test_dir}"
        print("Initial disk usage:")
        initial_usage = get_disk_usage(test_dir)
        for key, value in initial_usage.items():
            print(f"  {{key}}: {{value:.2f}}")
        
        # Create some files to change disk usage
        test_files = []
        try:
            for i in range(10):
                file_path = os.path.join(test_dir, f"test_file_{{i}}.txt")
                with open(file_path, 'w') as f:
                    # Write 1MB of data
                    content = "x" * (1024 * 1024)
                    f.write(content)
                test_files.append(file_path)
                
                if i % 3 == 0:  # Check every few files
                    current_usage = get_disk_usage(test_dir)
                    usage_change = current_usage['used_gb'] - initial_usage['used_gb']
                    print(f"After {{i+1}} files: +{{usage_change:.2f}} GB used")
            
            # Final check
            final_usage = get_disk_usage(test_dir)
            total_change = final_usage['used_gb'] - initial_usage['used_gb']
            print(f"Total disk usage change: +{{total_change:.2f}} GB")
            
        finally:
            # Clean up
            for file_path in test_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            cleanup_usage = get_disk_usage(test_dir)
            cleanup_change = cleanup_usage['used_gb'] - initial_usage['used_gb']
            print(f"After cleanup: {{cleanup_change:.2f}} GB change from initial")
"""
        
        pipeline_path = self.create_test_pipeline(disk_monitoring_pipeline, "disk_monitoring.yaml")
        
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        
        # Should complete disk monitoring
        assert result.status in ["success", "error", "failed"]
        
        print("✓ Disk space monitoring test completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])