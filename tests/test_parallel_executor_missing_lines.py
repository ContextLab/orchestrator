"""
Tests for missing coverage lines in parallel_executor.py.

This test file specifically targets:
- Line 111: return {} when no stats
- Lines 197-198: asyncio.TimeoutError in _execute_thread
- Lines 205-212: _execute_process method
- Line 272: break on critical failures
- Line 287: ValueError when task not found
- Line 309: skip task with unmet dependencies
- Lines 323-324: error handling in _execute_single_task
- Lines 339-340: TimeoutError handling
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from orchestrator.executor.parallel_executor import ParallelExecutor, ExecutionConfig, ExecutionMode, ExecutionResult, ResourceMonitor, WorkerPool
from orchestrator.core.pipeline import Pipeline
from orchestrator.core.task import Task, TaskStatus


class TestParallelExecutorMissingLines:
    """Test cases for achieving 100% coverage on parallel_executor.py."""

    def test_get_statistics_empty_line_104(self):
        """Test line 104: return {} when no stats in ResourceMonitor."""
        monitor = ResourceMonitor()
        
        # Ensure usage_stats is empty
        monitor.usage_stats.clear()
        
        stats = monitor.get_statistics()
        
        # Should trigger line 104 (was line 111 in coverage report)
        assert stats == {}
    
    @pytest.mark.asyncio
    async def test_execute_thread_timeout_lines_197_198(self):
        """Test lines 197-198: asyncio.TimeoutError in _execute_thread."""
        config = ExecutionConfig(timeout=0.001)  # Very short timeout
        worker_pool = WorkerPool(config)
        
        # Initialize thread pool
        worker_pool.thread_pool = ThreadPoolExecutor(max_workers=1)
        
        # Create a task that takes too long
        task = Task(id="slow_task", name="Slow Task", action="generate")
        
        # Mock executor function that sleeps longer than timeout
        def slow_executor(t):
            import time
            time.sleep(1)  # Sleep longer than timeout
            return "result"
        
        # Test timeout
        with pytest.raises(TimeoutError) as exc_info:
            await worker_pool._execute_thread(task, slow_executor)
        
        assert "timed out after 0.001s" in str(exc_info.value)
        
        # Cleanup
        worker_pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_execute_process_not_initialized_lines_202_203(self):
        """Test lines 202-203: RuntimeError when process pool not initialized."""
        worker_pool = WorkerPool(ExecutionConfig())
        
        # Ensure process pool is None
        worker_pool.process_pool = None
        
        task = Task(id="test_task", name="Test Task", action="generate")
        
        with pytest.raises(RuntimeError) as exc_info:
            await worker_pool._execute_process(task, lambda t: "result")
        
        assert "Process pool not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio 
    async def test_execute_process_success_lines_205_210(self):
        """Test lines 205-210: successful process execution."""
        config = ExecutionConfig(execution_mode=ExecutionMode.PROCESS)
        worker_pool = WorkerPool(config)
        
        # Mock the process pool execution to avoid pickle issues
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value="Processed test_task")
            
            # Set process pool to a truthy value
            worker_pool.process_pool = MagicMock()
            
            task = Task(id="test_task", name="Test Task", action="generate")
            
            # Simple executor function (won't actually be pickled)
            def executor_func(t):
                return f"Processed {t.id}"
            
            result = await worker_pool._execute_process(task, executor_func)
            
            assert result == "Processed test_task"
            
            # Verify the execution path was taken
            mock_loop.return_value.run_in_executor.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_process_timeout_lines_211_212(self):
        """Test lines 211-212: TimeoutError in _execute_process."""
        config = ExecutionConfig(execution_mode=ExecutionMode.PROCESS, timeout=0.1)
        worker_pool = WorkerPool(config)
        
        # Mock to simulate timeout
        with patch('asyncio.wait_for') as mock_wait_for:
            mock_wait_for.side_effect = asyncio.TimeoutError()
            
            # Set process pool to a truthy value
            worker_pool.process_pool = MagicMock()
            
            task = Task(id="slow_task", name="Slow Task", action="generate")
            
            # Executor function (won't actually run)
            def slow_executor(t):
                return "result"
            
            with pytest.raises(TimeoutError) as exc_info:
                await worker_pool._execute_process(task, slow_executor)
            
            assert "timed out after 0.1s" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_execute_critical_failure_line_272(self):
        """Test line 272: break on critical failures."""
        executor = ParallelExecutor()
        
        # Create pipeline with multiple levels
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        
        # Level 1 tasks
        task1 = Task(id="task1", name="Task 1", action="generate")
        task2 = Task(id="task2", name="Task 2", action="generate")
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        
        # Level 2 tasks (depend on level 1)
        task3 = Task(id="task3", name="Task 3", action="generate", dependencies=["task1"])
        task4 = Task(id="task4", name="Task 4", action="generate", dependencies=["task2"])
        pipeline.add_task(task3)
        pipeline.add_task(task4)
        
        # Mock executor function that fails for task2
        async def mock_executor(task):
            if task.id == "task2":
                raise Exception("Critical failure")
            return f"Result for {task.id}"
        
        # Override _is_critical_failure to mark task2 failure as critical
        executor._is_critical_failure = Mock(return_value=True)
        
        # Execute pipeline
        results = await executor.execute_pipeline(pipeline, mock_executor)
        
        # Verify execution stopped after critical failure
        assert "task1" in results
        assert "task2" in results
        assert results["task2"].status == "failed"
        # Level 2 should not be executed due to critical failure
        assert "task3" not in results
        assert "task4" not in results
    
    @pytest.mark.asyncio
    async def test_execute_level_task_not_found_line_287(self):
        """Test line 287: ValueError when task not found in pipeline."""
        executor = ParallelExecutor()
        
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        
        # Add a real task
        task = Task(id="real_task", name="Real Task", action="generate")
        pipeline.add_task(task)
        
        # Try to execute level with non-existent task
        with pytest.raises(ValueError) as exc_info:
            await executor._execute_level(
                ["real_task", "missing_task"],
                pipeline,
                lambda t: "result",
                0
            )
        
        assert "Task 'missing_task' not found in pipeline" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_execute_level_unmet_dependencies_line_309(self):
        """Test line 309: skip task with unmet dependencies."""
        executor = ParallelExecutor()
        
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        
        # Create tasks with dependencies
        task1 = Task(id="task1", name="Task 1", action="generate")
        task2 = Task(id="task2", name="Task 2", action="generate", dependencies=["task1"])
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        
        # Mark task1 as failed to make task2's dependencies unmet
        task1.fail(Exception("Task 1 failed"))
        
        # Execute level containing task2
        results = await executor._execute_level(
            ["task2"],
            pipeline,
            lambda t: "result",
            1
        )
        
        # Verify task2 was skipped
        assert task2.status == TaskStatus.SKIPPED
        assert "task2" in results
        assert results["task2"].status == "skipped"
        assert "Unmet dependencies" in results["task2"].error
    
    @pytest.mark.asyncio
    async def test_execute_level_exception_lines_323_324(self):
        """Test lines 323-324: exception handling in _execute_level."""
        executor = ParallelExecutor()
        
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task = Task(id="failing_task", name="Failing Task", action="generate")
        pipeline.add_task(task)
        
        # Mock worker pool execute_task to return an exception
        executor.worker_pool.execute_task = AsyncMock(side_effect=ValueError("Task execution failed"))
        
        # Execute level
        results = await executor._execute_level(
            ["failing_task"],
            pipeline,
            lambda t: "result",
            0
        )
        
        # Verify task was marked as failed (line 323)
        assert task.status == TaskStatus.FAILED
        assert "failing_task" in results
        assert results["failing_task"].success is False
        assert isinstance(results["failing_task"].error, ValueError)
    
    @pytest.mark.asyncio
    async def test_unexpected_result_type_lines_339_340(self):
        """Test lines 339-340: Unexpected result type handling."""
        executor = ParallelExecutor()
        
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task = Task(id="task1", name="Task 1", action="generate")
        pipeline.add_task(task)
        
        # Mock the _execute_single_task to return an unexpected type
        original_execute_single = executor._execute_single_task
        async def mock_execute_single(task, executor_func):
            if task.id == "task1":
                # Return a string instead of ExecutionResult
                return "unexpected_string_result"
            return await original_execute_single(task, executor_func)
        
        executor._execute_single_task = mock_execute_single
        
        # Execute
        results = await executor.execute_pipeline(pipeline, lambda t: "result")
        
        # Check that task was marked as failed due to unexpected result type
        assert task.status == TaskStatus.FAILED
        assert "task1" in results
        assert results["task1"].success is False
        assert "Unexpected result type" in str(results["task1"].error)
        
        executor._execute_single_task = original_execute_single