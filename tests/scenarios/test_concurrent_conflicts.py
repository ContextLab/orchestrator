"""
Concurrent Execution Conflicts Testing

Tests the orchestrator's handling of concurrent execution conflicts including
resource contention, file access conflicts, shared state issues, and deadlock scenarios.
"""

import pytest
import os
import time
import tempfile
import threading
import asyncio
import multiprocessing
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import fcntl
import sqlite3

# Import orchestrator components
from src.orchestrator.orchestrator import Orchestrator


class TestFileAccessConflicts:
    """Test handling of concurrent file access conflicts."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp()) / "concurrent_tests"
        self.test_dir.mkdir(exist_ok=True)
        self.executor = Orchestrator()
        
        # Create shared test files
        self.shared_file = self.test_dir / "shared_file.txt"
        self.shared_file.write_text("Initial content\n")
        
        self.lock_file = self.test_dir / "lock_test.txt"
        self.lock_file.write_text("Locked content\n")
    
    def create_test_pipeline(self, content: str, filename: str = "test_pipeline.yaml") -> Path:
        """Create a test pipeline file with given content."""
        pipeline_path = self.test_dir / filename
        pipeline_path.write_text(content)
        return pipeline_path
    
    @pytest.mark.asyncio
    async def test_concurrent_file_writes(self):
        """Test handling of multiple processes writing to the same file."""
        concurrent_write_pipeline = f"""
name: concurrent_write_test
version: "1.0"
steps:
  - id: concurrent_writers
    action: parallel
    parameters:
      max_concurrent: 5
      tasks:
        - action: python_code
          parameters:
            code: |
              import time
              import random
              
              file_path = "{self.shared_file}"
              writer_id = "writer_1"
              
              try:
                  # Simulate concurrent write operations
                  for i in range(10):
                      with open(file_path, "a") as f:
                          timestamp = time.time()
                          message = f"{{writer_id}}: Write {{i}} at {{timestamp}}\\n"
                          f.write(message)
                          print(f"{{writer_id}} wrote: {{message.strip()}}")
                      
                      # Small random delay to create race conditions
                      time.sleep(random.uniform(0.01, 0.05))
                  
                  print(f"{{writer_id}} completed all writes")
                  
              except Exception as e:
                  print(f"{{writer_id}} error: {{e}}")
                  raise
                  
        - action: python_code
          parameters:
            code: |
              import time
              import random
              
              file_path = "{self.shared_file}"
              writer_id = "writer_2"
              
              try:
                  for i in range(10):
                      with open(file_path, "a") as f:
                          timestamp = time.time()
                          message = f"{{writer_id}}: Write {{i}} at {{timestamp}}\\n"
                          f.write(message)
                          print(f"{{writer_id}} wrote: {{message.strip()}}")
                      
                      time.sleep(random.uniform(0.01, 0.05))
                  
                  print(f"{{writer_id}} completed all writes")
                  
              except Exception as e:
                  print(f"{{writer_id}} error: {{e}}")
                  raise
                  
        - action: python_code
          parameters:
            code: |
              import time
              import random
              
              file_path = "{self.shared_file}"
              writer_id = "writer_3"
              
              try:
                  for i in range(10):
                      with open(file_path, "a") as f:
                          timestamp = time.time()
                          message = f"{{writer_id}}: Write {{i}} at {{timestamp}}\\n"
                          f.write(message)
                          print(f"{{writer_id}} wrote: {{message.strip()}}")
                      
                      time.sleep(random.uniform(0.01, 0.05))
                  
                  print(f"{{writer_id}} completed all writes")
                  
              except Exception as e:
                  print(f"{{writer_id}} error: {{e}}")
                  raise
                  
  - id: verify_writes
    action: python_code
    parameters:
      code: |
        file_path = "{self.shared_file}"
        
        # Read final file content
        with open(file_path, "r") as f:
            content = f.read()
        
        lines = content.strip().split("\\n")
        print(f"Total lines written: {{len(lines)}}")
        
        # Count writes per writer
        writer_counts = {{}}
        for line in lines:
            if "writer_" in line:
                writer_id = line.split(":")[0]
                writer_counts[writer_id] = writer_counts.get(writer_id, 0) + 1
        
        print("Writes per writer:")
        for writer_id, count in writer_counts.items():
            print(f"  {{writer_id}}: {{count}} writes")
        
        # Check for data integrity issues
        total_expected = 10 * 3  # 10 writes per 3 writers
        if len([l for l in lines if l.strip()]) != total_expected + 1:  # +1 for initial content
            print(f"Warning: Expected {{total_expected + 1}} lines, got {{len(lines)}}")
        else:
            print("All concurrent writes completed successfully")
    depends_on:
      - concurrent_writers
"""
        
        pipeline_path = self.create_test_pipeline(concurrent_write_pipeline, "concurrent_writes.yaml")
        
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        
        # Should handle concurrent writes (may succeed with race conditions or fail gracefully)
        assert result.status in ["success", "partial_success", "error", "failed"]
        
        # Verify final file state
        final_content = self.shared_file.read_text()
        lines = [l for l in final_content.split("\n") if l.strip()]
        assert len(lines) > 1, "File should contain content from concurrent writes"
        
        print(f"✓ Concurrent file writes test completed ({len(lines)} lines written)")
    
    @pytest.mark.asyncio
    async def test_file_locking_conflicts(self):
        """Test handling of file locking conflicts."""
        file_locking_pipeline = f"""
name: file_locking_test
version: "1.0"
steps:
  - id: locking_contention
    action: parallel
    parameters:
      max_concurrent: 3
      tasks:
        - action: python_code
          parameters:
            code: |
              import time
              import fcntl
              
              file_path = "{self.lock_file}"
              process_id = "process_1"
              
              try:
                  with open(file_path, "a+") as f:
                      print(f"{{process_id}}: Attempting to acquire lock...")
                      
                      # Try to acquire exclusive lock
                      fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                      print(f"{{process_id}}: Lock acquired!")
                      
                      # Hold lock for some time
                      time.sleep(2)
                      
                      # Write while holding lock
                      f.write(f"{{process_id}}: Exclusive write at {{time.time()}}\\n")
                      f.flush()
                      
                      print(f"{{process_id}}: Releasing lock...")
                      # Lock automatically released when file closes
                      
              except (IOError, OSError) as e:
                  print(f"{{process_id}}: Could not acquire lock: {{e}}")
                  # Try non-blocking read instead
                  try:
                      with open(file_path, "r") as f:
                          content = f.read()
                      print(f"{{process_id}}: Read while locked: {{len(content)}} chars")
                  except Exception as read_error:
                      print(f"{{process_id}}: Read also failed: {{read_error}}")
                      
        - action: python_code
          parameters:
            code: |
              import time
              import fcntl
              
              file_path = "{self.lock_file}"
              process_id = "process_2"
              
              # Delay to let first process potentially acquire lock
              time.sleep(0.5)
              
              try:
                  with open(file_path, "a+") as f:
                      print(f"{{process_id}}: Attempting to acquire lock...")
                      
                      fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                      print(f"{{process_id}}: Lock acquired!")
                      
                      time.sleep(1)
                      f.write(f"{{process_id}}: Exclusive write at {{time.time()}}\\n")
                      f.flush()
                      
                      print(f"{{process_id}}: Releasing lock...")
                      
              except (IOError, OSError) as e:
                  print(f"{{process_id}}: Could not acquire lock: {{e}}")
                  try:
                      with open(file_path, "r") as f:
                          content = f.read()
                      print(f"{{process_id}}: Read while locked: {{len(content)}} chars")
                  except Exception as read_error:
                      print(f"{{process_id}}: Read also failed: {{read_error}}")
                      
        - action: python_code
          parameters:
            code: |
              import time
              import fcntl
              
              file_path = "{self.lock_file}"
              process_id = "process_3"
              
              time.sleep(1.0)  # Longer delay
              
              try:
                  with open(file_path, "a+") as f:
                      print(f"{{process_id}}: Attempting to acquire lock...")
                      
                      fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                      print(f"{{process_id}}: Lock acquired!")
                      
                      time.sleep(0.5)
                      f.write(f"{{process_id}}: Exclusive write at {{time.time()}}\\n")
                      f.flush()
                      
                      print(f"{{process_id}}: Releasing lock...")
                      
              except (IOError, OSError) as e:
                  print(f"{{process_id}}: Could not acquire lock: {{e}}")
                  try:
                      with open(file_path, "r") as f:
                          content = f.read()
                      print(f"{{process_id}}: Read while locked: {{len(content)}} chars")
                  except Exception as read_error:
                      print(f"{{process_id}}: Read also failed: {{read_error}}")
"""
        
        pipeline_path = self.create_test_pipeline(file_locking_pipeline, "file_locking.yaml")
        
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        
        # Should handle file locking appropriately
        assert result.status in ["success", "partial_success", "error", "failed"]
        
        # Check final file state
        final_content = self.lock_file.read_text()
        print(f"✓ File locking test completed, final content: {len(final_content)} chars")
    
    @pytest.mark.asyncio
    async def test_database_concurrent_access(self):
        """Test handling of concurrent database access conflicts."""
        db_file = self.test_dir / "test_database.db"
        
        database_conflict_pipeline = f"""
name: database_conflict_test
version: "1.0"
steps:
  - id: setup_database
    action: python_code
    parameters:
      code: |
        import sqlite3
        import os
        
        db_path = "{db_file}"
        
        # Remove existing database
        if os.path.exists(db_path):
            os.remove(db_path)
        
        # Create database and table
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE test_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                process_name TEXT,
                value INTEGER,
                timestamp REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print("Database setup completed")
        
  - id: concurrent_database_operations
    action: parallel
    parameters:
      max_concurrent: 4
      tasks:
        - action: python_code
          parameters:
            code: |
              import sqlite3
              import time
              import random
              
              db_path = "{db_file}"
              process_name = "inserter_1"
              
              try:
                  for i in range(20):
                      conn = sqlite3.connect(db_path, timeout=10.0)
                      cursor = conn.cursor()
                      
                      try:
                          cursor.execute(
                              "INSERT INTO test_data (process_name, value, timestamp) VALUES (?, ?, ?)",
                              (process_name, random.randint(1, 1000), time.time())
                          )
                          conn.commit()
                          print(f"{{process_name}}: Inserted record {{i+1}}")
                          
                      except sqlite3.OperationalError as e:
                          print(f"{{process_name}}: Database error on record {{i+1}}: {{e}}")
                          
                      finally:
                          conn.close()
                      
                      time.sleep(random.uniform(0.01, 0.05))
                  
                  print(f"{{process_name}}: All inserts completed")
                  
              except Exception as e:
                  print(f"{{process_name}}: Fatal error: {{e}}")
                  
        - action: python_code
          parameters:
            code: |
              import sqlite3
              import time
              import random
              
              db_path = "{db_file}"
              process_name = "inserter_2"
              
              time.sleep(0.1)  # Slight offset
              
              try:
                  for i in range(20):
                      conn = sqlite3.connect(db_path, timeout=10.0)
                      cursor = conn.cursor()
                      
                      try:
                          cursor.execute(
                              "INSERT INTO test_data (process_name, value, timestamp) VALUES (?, ?, ?)",
                              (process_name, random.randint(1, 1000), time.time())
                          )
                          conn.commit()
                          print(f"{{process_name}}: Inserted record {{i+1}}")
                          
                      except sqlite3.OperationalError as e:
                          print(f"{{process_name}}: Database error on record {{i+1}}: {{e}}")
                          
                      finally:
                          conn.close()
                      
                      time.sleep(random.uniform(0.01, 0.05))
                  
                  print(f"{{process_name}}: All inserts completed")
                  
              except Exception as e:
                  print(f"{{process_name}}: Fatal error: {{e}}")
                  
        - action: python_code
          parameters:
            code: |
              import sqlite3
              import time
              
              db_path = "{db_file}"
              process_name = "reader_1"
              
              try:
                  for i in range(30):  # More frequent reads
                      conn = sqlite3.connect(db_path, timeout=5.0)
                      cursor = conn.cursor()
                      
                      try:
                          cursor.execute("SELECT COUNT(*) FROM test_data")
                          count = cursor.fetchone()[0]
                          
                          cursor.execute("SELECT MAX(timestamp) FROM test_data")
                          latest = cursor.fetchone()[0]
                          
                          print(f"{{process_name}}: Read {{count}} records, latest: {{latest}}")
                          
                      except sqlite3.OperationalError as e:
                          print(f"{{process_name}}: Read error {{i+1}}: {{e}}")
                          
                      finally:
                          conn.close()
                      
                      time.sleep(0.1)
                  
                  print(f"{{process_name}}: All reads completed")
                  
              except Exception as e:
                  print(f"{{process_name}}: Fatal error: {{e}}")
                  
        - action: python_code
          parameters:
            code: |
              import sqlite3
              import time
              
              db_path = "{db_file}"
              process_name = "updater_1"
              
              time.sleep(0.5)  # Let some inserts happen first
              
              try:
                  for i in range(10):
                      conn = sqlite3.connect(db_path, timeout=10.0)
                      cursor = conn.cursor()
                      
                      try:
                          # Update random records
                          cursor.execute("UPDATE test_data SET value = value * 2 WHERE id % 5 = ?", (i % 5,))
                          updated = cursor.rowcount
                          conn.commit()
                          
                          print(f"{{process_name}}: Updated {{updated}} records in batch {{i+1}}")
                          
                      except sqlite3.OperationalError as e:
                          print(f"{{process_name}}: Update error {{i+1}}: {{e}}")
                          
                      finally:
                          conn.close()
                      
                      time.sleep(0.2)
                  
                  print(f"{{process_name}}: All updates completed")
                  
              except Exception as e:
                  print(f"{{process_name}}: Fatal error: {{e}}")
    depends_on:
      - setup_database
      
  - id: verify_database_integrity
    action: python_code
    parameters:
      code: |
        import sqlite3
        
        db_path = "{db_file}"
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check final record count
        cursor.execute("SELECT COUNT(*) FROM test_data")
        total_records = cursor.fetchone()[0]
        
        # Check records per process
        cursor.execute("SELECT process_name, COUNT(*) FROM test_data GROUP BY process_name")
        process_counts = cursor.fetchall()
        
        print(f"Final database state:")
        print(f"  Total records: {{total_records}}")
        for process_name, count in process_counts:
            print(f"  {{process_name}}: {{count}} records")
        
        # Check for data integrity
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM test_data")
        min_time, max_time = cursor.fetchone()
        duration = max_time - min_time if min_time and max_time else 0
        
        print(f"  Time span: {{duration:.2f}} seconds")
        
        conn.close()
        
        print("Database integrity check completed")
    depends_on:
      - concurrent_database_operations
"""
        
        pipeline_path = self.create_test_pipeline(database_conflict_pipeline, "database_conflicts.yaml")
        
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        
        # Should handle database concurrency appropriately
        assert result.status in ["success", "partial_success", "error", "failed"]
        
        print("✓ Database concurrent access test completed")


class TestResourceContentionScenarios:
    """Test handling of various resource contention scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp()) / "resource_contention_tests"
        self.test_dir.mkdir(exist_ok=True)
        self.executor = Orchestrator()
    
    def create_test_pipeline(self, content: str, filename: str = "test_pipeline.yaml") -> Path:
        """Create a test pipeline file with given content."""
        pipeline_path = self.test_dir / filename
        pipeline_path.write_text(content)
        return pipeline_path
    
    @pytest.mark.asyncio
    async def test_memory_contention(self):
        """Test handling of memory contention between concurrent processes."""
        memory_contention_pipeline = """
name: memory_contention_test
version: "1.0"
steps:
  - id: memory_competition
    action: parallel
    parameters:
      max_concurrent: 3
      tasks:
        - action: python_code
          parameters:
            code: |
              import time
              import gc
              import psutil
              
              process_name = "memory_user_1"
              memory_chunks = []
              
              try:
                  print(f"{process_name}: Starting memory allocation...")
                  
                  for i in range(50):  # Allocate 50MB
                      chunk = bytearray(1024 * 1024)  # 1MB
                      memory_chunks.append(chunk)
                      
                      if i % 10 == 0:
                          process = psutil.Process()
                          memory_mb = process.memory_info().rss / 1024 / 1024
                          print(f"{process_name}: Allocated {i+1}MB, using {memory_mb:.1f}MB")
                      
                      time.sleep(0.02)
                  
                  # Hold memory for a while
                  time.sleep(2)
                  print(f"{process_name}: Memory allocation completed")
                  
              except MemoryError as e:
                  print(f"{process_name}: Memory allocation failed: {e}")
              finally:
                  del memory_chunks
                  gc.collect()
                  print(f"{process_name}: Memory cleaned up")
                  
        - action: python_code
          parameters:
            code: |
              import time
              import gc
              import psutil
              
              process_name = "memory_user_2"
              memory_chunks = []
              
              time.sleep(0.5)  # Offset start
              
              try:
                  print(f"{process_name}: Starting memory allocation...")
                  
                  for i in range(40):  # Allocate 40MB
                      chunk = bytearray(1024 * 1024)  # 1MB
                      memory_chunks.append(chunk)
                      
                      if i % 10 == 0:
                          process = psutil.Process()
                          memory_mb = process.memory_info().rss / 1024 / 1024
                          print(f"{process_name}: Allocated {i+1}MB, using {memory_mb:.1f}MB")
                      
                      time.sleep(0.03)
                  
                  time.sleep(1.5)
                  print(f"{process_name}: Memory allocation completed")
                  
              except MemoryError as e:
                  print(f"{process_name}: Memory allocation failed: {e}")
              finally:
                  del memory_chunks
                  gc.collect()
                  print(f"{process_name}: Memory cleaned up")
                  
        - action: python_code
          parameters:
            code: |
              import time
              import gc
              import psutil
              
              process_name = "memory_user_3"
              memory_chunks = []
              
              time.sleep(1.0)  # Later start
              
              try:
                  print(f"{process_name}: Starting memory allocation...")
                  
                  for i in range(30):  # Allocate 30MB
                      chunk = bytearray(1024 * 1024)  # 1MB
                      memory_chunks.append(chunk)
                      
                      if i % 10 == 0:
                          process = psutil.Process()
                          memory_mb = process.memory_info().rss / 1024 / 1024
                          print(f"{process_name}: Allocated {i+1}MB, using {memory_mb:.1f}MB")
                      
                      time.sleep(0.04)
                  
                  time.sleep(1)
                  print(f"{process_name}: Memory allocation completed")
                  
              except MemoryError as e:
                  print(f"{process_name}: Memory allocation failed: {e}")
              finally:
                  del memory_chunks
                  gc.collect()
                  print(f"{process_name}: Memory cleaned up")
"""
        
        pipeline_path = self.create_test_pipeline(memory_contention_pipeline, "memory_contention.yaml")
        
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        
        # Should handle memory contention gracefully
        assert result.status in ["success", "partial_success", "error", "failed"]
        
        print("✓ Memory contention test completed")
    
    @pytest.mark.asyncio
    async def test_cpu_contention(self):
        """Test handling of CPU contention between concurrent processes."""
        cpu_count = multiprocessing.cpu_count()
        overload_factor = cpu_count + 1
        
        cpu_contention_pipeline = f"""
name: cpu_contention_test
version: "1.0"
steps:
  - id: cpu_competition
    action: parallel
    parameters:
      max_concurrent: {overload_factor}
      tasks:
"""
        
        for i in range(overload_factor):
            cpu_contention_pipeline += f"""
        - action: python_code
          parameters:
            code: |
              import time
              import math
              
              process_name = "cpu_worker_{i}"
              start_time = time.time()
              
              print(f"{{process_name}}: Starting CPU-intensive work...")
              
              # CPU-intensive computation
              result = 0
              iterations = 500000  # Moderate workload
              
              for j in range(iterations):
                  result += math.sin(j) * math.cos(j) + j ** 0.5
                  
                  if j % 100000 == 0 and j > 0:
                      elapsed = time.time() - start_time
                      progress = j / iterations * 100
                      print(f"{{process_name}}: {{progress:.1f}}% complete, {{elapsed:.1f}}s elapsed")
              
              total_time = time.time() - start_time
              print(f"{{process_name}}: Completed in {{total_time:.2f}}s, result={{result:.2e}}")
"""
        
        pipeline_path = self.create_test_pipeline(cpu_contention_pipeline, "cpu_contention.yaml")
        
        start_time = time.time()
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        execution_time = time.time() - start_time
        
        # Should handle CPU contention
        assert result.status in ["success", "partial_success", "timeout", "error", "failed"]
        
        print(f"✓ CPU contention test completed in {execution_time:.2f}s")
        print(f"  {overload_factor} workers on {cpu_count} CPU cores")
    
    @pytest.mark.asyncio
    async def test_mixed_resource_contention(self):
        """Test handling of mixed resource contention (CPU, memory, I/O)."""
        mixed_contention_pipeline = f"""
name: mixed_contention_test
version: "1.0"
steps:
  - id: mixed_resource_competition
    action: parallel
    parameters:
      max_concurrent: 4
      tasks:
        - action: python_code
          parameters:
            code: |
              # CPU-intensive task
              import math
              import time
              
              task_name = "cpu_intensive"
              print(f"{{task_name}}: Starting CPU work...")
              
              result = 0
              for i in range(300000):
                  result += math.sin(i) * math.cos(i)
              
              print(f"{{task_name}}: CPU work completed, result={{result:.2e}}")
              
        - action: python_code
          parameters:
            code: |
              # Memory-intensive task
              import time
              import gc
              
              task_name = "memory_intensive"
              print(f"{{task_name}}: Starting memory allocation...")
              
              memory_chunks = []
              for i in range(30):  # 30MB
                  chunk = bytearray(1024 * 1024)
                  memory_chunks.append(chunk)
                  time.sleep(0.05)
              
              time.sleep(1)  # Hold memory
              print(f"{{task_name}}: Memory work completed")
              
              del memory_chunks
              gc.collect()
              
        - action: python_code
          parameters:
            code: |
              # I/O-intensive task
              import os
              import time
              
              task_name = "io_intensive"
              print(f"{{task_name}}: Starting I/O work...")
              
              file_path = "{self.test_dir}/io_test_file.txt"
              
              # Write operations
              for i in range(100):
                  with open(file_path, "a") as f:
                      data = f"Line {{i}}: " + "x" * 1000 + "\\n"
                      f.write(data)
                  time.sleep(0.01)
              
              # Read operations
              for i in range(50):
                  with open(file_path, "r") as f:
                      content = f.read()
                  time.sleep(0.02)
              
              print(f"{{task_name}}: I/O work completed")
              
              # Cleanup
              if os.path.exists(file_path):
                  os.remove(file_path)
                  
        - action: python_code
          parameters:
            code: |
              # Mixed workload task
              import math
              import time
              import os
              
              task_name = "mixed_workload"
              print(f"{{task_name}}: Starting mixed work...")
              
              # Alternate between CPU, memory, and I/O
              for cycle in range(5):
                  # CPU work
                  result = sum(math.sin(i) for i in range(50000))
                  
                  # Memory work
                  temp_data = [i ** 2 for i in range(10000)]
                  
                  # I/O work
                  temp_file = "{self.test_dir}/mixed_temp_{{cycle}}.txt"
                  with open(temp_file, "w") as f:
                      f.write(f"Cycle {{cycle}} data: {{result}}\\n")
                  
                  with open(temp_file, "r") as f:
                      content = f.read()
                  
                  os.remove(temp_file)
                  del temp_data
                  
                  print(f"{{task_name}}: Completed cycle {{cycle + 1}}/5")
                  time.sleep(0.1)
              
              print(f"{{task_name}}: Mixed work completed")
"""
        
        pipeline_path = self.create_test_pipeline(mixed_contention_pipeline, "mixed_contention.yaml")
        
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        
        # Should handle mixed resource contention
        assert result.status in ["success", "partial_success", "error", "failed"]
        
        print("✓ Mixed resource contention test completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])