"""Tests for parallel execution engine."""

import asyncio
import time
from collections import defaultdict
from unittest.mock import patch

import pytest

from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.task import Task
from src.orchestrator.executor.parallel_executor import (
    ExecutionConfig,
    ExecutionMode,
    ExecutionResult,
    ParallelExecutor,
    ResourceMonitor,
    WorkerPool,
)


class TestExecutionMode:
    """Test cases for ExecutionMode enum."""

    def test_execution_mode_values(self):
        """Test execution mode enum values."""
        assert ExecutionMode.ASYNC.value == "async"
        assert ExecutionMode.THREAD.value == "thread"
        assert ExecutionMode.PROCESS.value == "process"


class TestExecutionConfig:
    """Test cases for ExecutionConfig class."""

    def test_execution_config_default_creation(self):
        """Test default execution config creation."""
        config = ExecutionConfig()

        assert config.max_workers == 4
        assert config.execution_mode == ExecutionMode.ASYNC
        assert config.timeout == 300.0
        assert "memory_limit" in config.resource_limits
        assert "cpu_limit" in config.resource_limits
        assert config.resource_limits["memory_limit"] == "512m"
        assert config.resource_limits["cpu_limit"] == 1.0

    def test_execution_config_custom_creation(self):
        """Test custom execution config creation."""
        config = ExecutionConfig(
            max_workers=8,
            execution_mode=ExecutionMode.THREAD,
            timeout=600.0,
            resource_limits={"memory_limit": "1g", "cpu_limit": 2.0},
        )

        assert config.max_workers == 8
        assert config.execution_mode == ExecutionMode.THREAD
        assert config.timeout == 600.0
        assert config.resource_limits["memory_limit"] == "1g"
        assert config.resource_limits["cpu_limit"] == 2.0

    def test_execution_config_post_init(self):
        """Test execution config post-init behavior."""
        config = ExecutionConfig(resource_limits={})

        # Should set default resource limits
        assert config.resource_limits["memory_limit"] == "512m"
        assert config.resource_limits["cpu_limit"] == 1.0


class TestExecutionResult:
    """Test cases for ExecutionResult class."""

    def test_execution_result_success(self):
        """Test successful execution result."""
        result = ExecutionResult(
            task_id="test_task",
            success=True,
            result={"data": "test"},
            execution_time=1.5,
            worker_id="worker_1",
        )

        assert result.task_id == "test_task"
        assert result.success is True
        assert result.result == {"data": "test"}
        assert result.error is None
        assert result.execution_time == 1.5
        assert result.worker_id == "worker_1"

    def test_execution_result_failure(self):
        """Test failed execution result."""
        error = ValueError("Test error")
        result = ExecutionResult(
            task_id="test_task", success=False, error=error, execution_time=0.5
        )

        assert result.task_id == "test_task"
        assert result.success is False
        assert result.result is None
        assert result.error == error
        assert result.execution_time == 0.5
        assert result.worker_id is None


class TestResourceMonitor:
    """Test cases for ResourceMonitor class."""

    def test_resource_monitor_creation(self):
        """Test resource monitor creation."""
        monitor = ResourceMonitor()

        assert hasattr(monitor, "usage_stats")
        assert hasattr(monitor, "active_tasks")
        assert isinstance(monitor.usage_stats, defaultdict)
        assert monitor.active_tasks == {}

    def test_resource_monitor_start_monitoring(self):
        """Test starting task monitoring."""
        monitor = ResourceMonitor()

        with (
            patch.object(monitor, "_get_memory_usage", return_value=100.0),
            patch.object(monitor, "_get_cpu_usage", return_value=25.0),
        ):

            monitor.start_monitoring("task1")

            assert "task1" in monitor.active_tasks
            assert "start_time" in monitor.active_tasks["task1"]
            assert monitor.active_tasks["task1"]["memory_start"] == 100.0
            assert monitor.active_tasks["task1"]["cpu_start"] == 25.0

    def test_resource_monitor_stop_monitoring(self):
        """Test stopping task monitoring."""
        monitor = ResourceMonitor()

        # Start monitoring first
        start_time = time.time()
        monitor.active_tasks["task1"] = {
            "start_time": start_time,
            "memory_start": 100.0,
            "cpu_start": 25.0,
        }

        with (
            patch.object(monitor, "_get_memory_usage", return_value=120.0),
            patch.object(monitor, "_get_cpu_usage", return_value=30.0),
        ):

            # Allow some time to pass
            time.sleep(0.1)
            usage = monitor.stop_monitoring("task1")

            assert "task1" not in monitor.active_tasks
            assert "execution_time" in usage
            assert "memory_used" in usage
            assert "cpu_used" in usage
            assert usage["memory_used"] == 20.0  # 120 - 100
            assert usage["cpu_used"] == 5.0  # 30 - 25
            assert usage["execution_time"] > 0

    def test_resource_monitor_stop_monitoring_nonexistent(self):
        """Test stopping monitoring for non-existent task."""
        monitor = ResourceMonitor()

        usage = monitor.stop_monitoring("nonexistent")
        assert usage == {}

    def test_resource_monitor_get_memory_usage_without_psutil(self):
        """Test memory usage when psutil is not available."""
        monitor = ResourceMonitor()

        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'psutil'")
        ):
            usage = monitor._get_memory_usage()
            assert usage == 0.0

    def test_resource_monitor_get_cpu_usage_without_psutil(self):
        """Test CPU usage when psutil is not available."""
        monitor = ResourceMonitor()

        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'psutil'")
        ):
            usage = monitor._get_cpu_usage()
            assert usage == 0.0

    def test_resource_monitor_get_statistics_empty(self):
        """Test getting statistics when no data exists."""
        monitor = ResourceMonitor()

        stats = monitor.get_statistics()
        assert stats == {}

    def test_resource_monitor_get_statistics_with_data(self):
        """Test getting statistics with monitoring data."""
        monitor = ResourceMonitor()

        # Add some test data
        monitor.usage_stats["task1"] = [
            {"execution_time": 1.0, "memory_used": 10.0, "cpu_used": 5.0},
            {"execution_time": 2.0, "memory_used": 20.0, "cpu_used": 10.0},
        ]
        monitor.usage_stats["task2"] = [
            {"execution_time": 1.5, "memory_used": 15.0, "cpu_used": 7.5}
        ]
        monitor.active_tasks["task3"] = {"start_time": time.time()}

        stats = monitor.get_statistics()

        assert stats["total_executions"] == 3
        assert stats["avg_execution_time"] == 1.5  # (1.0 + 2.0 + 1.5) / 3
        assert stats["avg_memory_used"] == 15.0  # (10.0 + 20.0 + 15.0) / 3
        assert stats["avg_cpu_used"] == 7.5  # (5.0 + 10.0 + 7.5) / 3
        assert stats["max_execution_time"] == 2.0
        assert stats["max_memory_used"] == 20.0
        assert stats["active_tasks"] == 1


class TestWorkerPool:
    """Test cases for WorkerPool class."""

    def test_worker_pool_creation(self):
        """Test worker pool creation."""
        config = ExecutionConfig()
        pool = WorkerPool(config)

        assert pool.config == config
        assert pool.thread_pool is None
        assert pool.process_pool is None
        assert isinstance(pool.resource_monitor, ResourceMonitor)

    def test_worker_pool_initialize_async_mode(self):
        """Test worker pool initialization for async mode."""
        config = ExecutionConfig(execution_mode=ExecutionMode.ASYNC)
        pool = WorkerPool(config)

        pool.initialize()

        # Async mode doesn't create pools
        assert pool.thread_pool is None
        assert pool.process_pool is None

    def test_worker_pool_initialize_thread_mode(self):
        """Test worker pool initialization for thread mode."""
        config = ExecutionConfig(execution_mode=ExecutionMode.THREAD, max_workers=2)
        pool = WorkerPool(config)

        pool.initialize()

        assert pool.thread_pool is not None
        assert pool.thread_pool._max_workers == 2
        assert pool.process_pool is None

        # Cleanup
        pool.shutdown()

    def test_worker_pool_initialize_process_mode(self):
        """Test worker pool initialization for process mode."""
        config = ExecutionConfig(execution_mode=ExecutionMode.PROCESS, max_workers=2)
        pool = WorkerPool(config)

        pool.initialize()

        assert pool.process_pool is not None
        assert pool.process_pool._max_workers == 2
        assert pool.thread_pool is None

        # Cleanup
        pool.shutdown()

    @pytest.mark.asyncio
    async def test_worker_pool_execute_task_async_success(self):
        """Test successful task execution in async mode."""
        config = ExecutionConfig(execution_mode=ExecutionMode.ASYNC)
        pool = WorkerPool(config)
        pool.initialize()

        task = Task(id="test_task", name="Test Task", action="test_action")

        async def mock_executor(t):
            await asyncio.sleep(0.01)
            return {"result": "success"}

        result = await pool.execute_task(task, mock_executor)

        assert isinstance(result, ExecutionResult)
        assert result.task_id == "test_task"
        assert result.success is True
        assert result.result == {"result": "success"}
        assert result.error is None
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_worker_pool_execute_task_async_failure(self):
        """Test failed task execution in async mode."""
        config = ExecutionConfig(execution_mode=ExecutionMode.ASYNC)
        pool = WorkerPool(config)
        pool.initialize()

        task = Task(id="test_task", name="Test Task", action="test_action")

        async def mock_executor(t):
            raise ValueError("Test error")

        result = await pool.execute_task(task, mock_executor)

        assert isinstance(result, ExecutionResult)
        assert result.task_id == "test_task"
        assert result.success is False
        assert result.result is None
        assert isinstance(result.error, ValueError)
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_worker_pool_execute_task_timeout(self):
        """Test task execution timeout."""
        config = ExecutionConfig(execution_mode=ExecutionMode.ASYNC, timeout=0.1)
        pool = WorkerPool(config)
        pool.initialize()

        task = Task(id="test_task", name="Test Task", action="test_action")

        async def mock_executor(t):
            await asyncio.sleep(0.2)  # Longer than timeout
            return {"result": "success"}

        result = await pool.execute_task(task, mock_executor)

        assert isinstance(result, ExecutionResult)
        assert result.task_id == "test_task"
        assert result.success is False
        assert isinstance(result.error, TimeoutError)

    @pytest.mark.asyncio
    async def test_worker_pool_execute_task_thread_mode(self):
        """Test task execution in thread mode."""
        config = ExecutionConfig(execution_mode=ExecutionMode.THREAD, max_workers=2)
        pool = WorkerPool(config)
        pool.initialize()

        task = Task(id="test_task", name="Test Task", action="test_action")

        def mock_executor(t):
            time.sleep(0.01)
            return {"result": "thread_success"}

        result = await pool.execute_task(task, mock_executor)

        assert isinstance(result, ExecutionResult)
        assert result.task_id == "test_task"
        assert result.success is True
        assert result.result == {"result": "thread_success"}

        # Cleanup
        pool.shutdown()

    @pytest.mark.asyncio
    async def test_worker_pool_execute_task_thread_not_initialized(self):
        """Test task execution when thread pool not initialized."""
        config = ExecutionConfig(execution_mode=ExecutionMode.THREAD)
        pool = WorkerPool(config)
        # Don't call initialize()

        task = Task(id="test_task", name="Test Task", action="test_action")

        def mock_executor(t):
            return {"result": "success"}

        result = await pool.execute_task(task, mock_executor)

        assert isinstance(result, ExecutionResult)
        assert result.success is False
        assert isinstance(result.error, RuntimeError)
        assert "Thread pool not initialized" in str(result.error)

    @pytest.mark.asyncio
    async def test_worker_pool_execute_task_process_not_initialized(self):
        """Test task execution when process pool not initialized."""
        config = ExecutionConfig(execution_mode=ExecutionMode.PROCESS)
        pool = WorkerPool(config)
        # Don't call initialize()

        task = Task(id="test_task", name="Test Task", action="test_action")

        def mock_executor(t):
            return {"result": "success"}

        result = await pool.execute_task(task, mock_executor)

        assert isinstance(result, ExecutionResult)
        assert result.success is False
        assert isinstance(result.error, RuntimeError)
        assert "Process pool not initialized" in str(result.error)

    @pytest.mark.asyncio
    async def test_worker_pool_execute_task_unsupported_mode(self):
        """Test task execution with unsupported execution mode."""
        config = ExecutionConfig()
        config.execution_mode = "unsupported"  # Invalid mode
        pool = WorkerPool(config)
        pool.initialize()

        task = Task(id="test_task", name="Test Task", action="test_action")

        async def mock_executor(t):
            return {"result": "success"}

        result = await pool.execute_task(task, mock_executor)

        assert isinstance(result, ExecutionResult)
        assert result.success is False
        assert isinstance(result.error, ValueError)
        assert "Unsupported execution mode" in str(result.error)

    def test_worker_pool_shutdown(self):
        """Test worker pool shutdown."""
        config = ExecutionConfig(execution_mode=ExecutionMode.THREAD, max_workers=2)
        pool = WorkerPool(config)
        pool.initialize()

        # Verify pool is initialized
        assert pool.thread_pool is not None

        # Shutdown
        pool.shutdown()

        # Pool should still exist but be shut down
        assert pool.thread_pool is not None

    def test_worker_pool_get_statistics(self):
        """Test getting worker pool statistics."""
        config = ExecutionConfig(
            execution_mode=ExecutionMode.ASYNC, max_workers=4, timeout=300.0
        )
        pool = WorkerPool(config)

        # Add some test data to resource monitor
        pool.resource_monitor.usage_stats["task1"] = [
            {"execution_time": 1.0, "memory_used": 10.0, "cpu_used": 5.0}
        ]

        stats = pool.get_statistics()

        assert stats["execution_mode"] == "async"
        assert stats["max_workers"] == 4
        assert stats["timeout"] == 300.0
        assert "resource_usage" in stats


class TestParallelExecutor:
    """Test cases for ParallelExecutor class."""

    def test_parallel_executor_creation_default(self):
        """Test parallel executor creation with default config."""
        executor = ParallelExecutor()

        assert isinstance(executor.config, ExecutionConfig)
        assert isinstance(executor.worker_pool, WorkerPool)
        assert executor.execution_graph == {}
        assert executor.task_results == {}
        assert isinstance(executor.execution_stats, defaultdict)

    def test_parallel_executor_creation_custom_config(self):
        """Test parallel executor creation with custom config."""
        config = ExecutionConfig(max_workers=8, execution_mode=ExecutionMode.THREAD)
        executor = ParallelExecutor(config)

        assert executor.config == config
        assert executor.worker_pool.config == config

    def test_parallel_executor_initialize(self):
        """Test parallel executor initialization."""
        executor = ParallelExecutor()

        with patch.object(executor.worker_pool, "initialize") as mock_init:
            executor.initialize()
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_executor_execute_pipeline_simple(self):
        """Test executing a simple pipeline."""
        executor = ParallelExecutor()
        executor.initialize()

        # Create simple pipeline
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="test_action")
        task2 = Task(
            id="task2", name="Task 2", action="test_action", dependencies=["task1"]
        )

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        # Mock executor function
        async def mock_executor(task):
            await asyncio.sleep(0.01)
            return f"result_{task.id}"

        # Execute pipeline
        results = await executor.execute_pipeline(pipeline, mock_executor)

        assert len(results) == 2
        assert "task1" in results
        assert "task2" in results
        assert results["task1"].success is True
        assert results["task2"].success is True
        assert results["task1"].result == "result_task1"
        assert results["task2"].result == "result_task2"

    @pytest.mark.asyncio
    async def test_parallel_executor_execute_pipeline_parallel_tasks(self):
        """Test executing pipeline with parallel tasks."""
        executor = ParallelExecutor()
        executor.initialize()

        # Create pipeline with parallel tasks
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="test_action")
        task2 = Task(
            id="task2", name="Task 2", action="test_action"
        )  # No dependencies, can run in parallel
        task3 = Task(
            id="task3",
            name="Task 3",
            action="test_action",
            dependencies=["task1", "task2"],
        )

        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)

        # Mock executor function
        async def mock_executor(task):
            await asyncio.sleep(0.01)
            return f"result_{task.id}"

        # Execute pipeline
        results = await executor.execute_pipeline(pipeline, mock_executor)

        assert len(results) == 3
        assert all(results[task_id].success for task_id in results)

        # Verify execution statistics
        stats = executor.get_execution_statistics()
        assert stats["total_tasks"] == 3
        assert stats["successful_tasks"] == 3
        assert stats["failed_tasks"] == 0

    @pytest.mark.asyncio
    async def test_parallel_executor_execute_pipeline_with_failure(self):
        """Test executing pipeline with task failure."""
        executor = ParallelExecutor()
        executor.initialize()

        # Create simple pipeline
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="test_action")
        task2 = Task(
            id="task2", name="Task 2", action="test_action", dependencies=["task1"]
        )

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        # Mock executor function that fails for task1
        async def mock_executor(task):
            if task.id == "task1":
                raise ValueError("Task 1 failed")
            await asyncio.sleep(0.01)
            return f"result_{task.id}"

        # Execute pipeline
        results = await executor.execute_pipeline(pipeline, mock_executor)

        assert len(results) == 2
        assert results["task1"].success is False
        assert isinstance(results["task1"].error, ValueError)
        assert results["task2"].success is False  # Should fail due to dependency

    def test_parallel_executor_build_execution_graph(self):
        """Test building execution dependency graph."""
        executor = ParallelExecutor()

        # Create pipeline
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="test_action")
        task2 = Task(
            id="task2", name="Task 2", action="test_action", dependencies=["task1"]
        )
        task3 = Task(
            id="task3",
            name="Task 3",
            action="test_action",
            dependencies=["task1", "task2"],
        )

        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)

        graph = executor._build_execution_graph(pipeline)

        assert graph["task1"] == set()
        assert graph["task2"] == {"task1"}
        assert graph["task3"] == {"task1", "task2"}

    def test_parallel_executor_are_dependencies_satisfied_true(self):
        """Test checking dependencies when they are satisfied."""
        executor = ParallelExecutor()

        # Mock successful dependency results
        executor.task_results = {
            "dep1": ExecutionResult("dep1", True, "result1"),
            "dep2": ExecutionResult("dep2", True, "result2"),
        }

        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task = Task(
            id="task1",
            name="Task 1",
            action="test_action",
            dependencies=["dep1", "dep2"],
        )

        satisfied = executor._are_dependencies_satisfied(task, pipeline)
        assert satisfied is True

    def test_parallel_executor_are_dependencies_satisfied_false_missing(self):
        """Test checking dependencies when some are missing."""
        executor = ParallelExecutor()

        # Only one dependency result
        executor.task_results = {"dep1": ExecutionResult("dep1", True, "result1")}

        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task = Task(
            id="task1",
            name="Task 1",
            action="test_action",
            dependencies=["dep1", "dep2"],
        )

        satisfied = executor._are_dependencies_satisfied(task, pipeline)
        assert satisfied is False

    def test_parallel_executor_are_dependencies_satisfied_false_failed(self):
        """Test checking dependencies when some failed."""
        executor = ParallelExecutor()

        # One dependency failed
        executor.task_results = {
            "dep1": ExecutionResult("dep1", True, "result1"),
            "dep2": ExecutionResult("dep2", False, error=ValueError("Failed")),
        }

        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task = Task(
            id="task1",
            name="Task 1",
            action="test_action",
            dependencies=["dep1", "dep2"],
        )

        satisfied = executor._are_dependencies_satisfied(task, pipeline)
        assert satisfied is False

    def test_parallel_executor_is_critical_failure(self):
        """Test determining if failure is critical."""
        executor = ParallelExecutor()
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        result = ExecutionResult("task1", False, error=ValueError("Test error"))

        # Current implementation always returns False
        is_critical = executor._is_critical_failure(result, pipeline)
        assert is_critical is False

    @pytest.mark.asyncio
    async def test_parallel_executor_create_immediate_result(self):
        """Test creating immediate result."""
        executor = ParallelExecutor()
        result = ExecutionResult("task1", True, "test_result")

        immediate_result = await executor._create_immediate_result(result)
        assert immediate_result == result

    def test_parallel_executor_get_execution_statistics_empty(self):
        """Test getting execution statistics when no executions."""
        executor = ParallelExecutor()

        stats = executor.get_execution_statistics()
        assert stats == {}

    def test_parallel_executor_get_execution_statistics_with_data(self):
        """Test getting execution statistics with execution data."""
        executor = ParallelExecutor()

        # Add mock execution stats
        executor.execution_stats["task1"] = {
            "level": 0,
            "execution_time": 1.0,
            "success": True,
            "worker_id": "worker1",
        }
        executor.execution_stats["task2"] = {
            "level": 0,
            "execution_time": 2.0,
            "success": True,
            "worker_id": "worker2",
        }
        executor.execution_stats["task3"] = {
            "level": 1,
            "execution_time": 1.5,
            "success": False,
            "worker_id": "worker1",
        }

        # Mock worker pool statistics
        with patch.object(
            executor.worker_pool, "get_statistics", return_value={"test": "data"}
        ):
            stats = executor.get_execution_statistics()

        assert stats["total_tasks"] == 3
        assert stats["successful_tasks"] == 2
        assert stats["failed_tasks"] == 1
        assert stats["total_execution_time"] == 4.5
        assert stats["avg_task_time"] == 1.5
        assert stats["levels_executed"] == 2

        # Check level statistics
        assert 0 in stats["level_statistics"]
        assert 1 in stats["level_statistics"]
        assert stats["level_statistics"][0]["task_count"] == 2
        assert stats["level_statistics"][0]["total_time"] == 3.0
        assert stats["level_statistics"][1]["task_count"] == 1
        assert stats["level_statistics"][1]["total_time"] == 1.5

    def test_parallel_executor_shutdown(self):
        """Test parallel executor shutdown."""
        executor = ParallelExecutor()

        with patch.object(executor.worker_pool, "shutdown") as mock_shutdown:
            executor.shutdown()
            mock_shutdown.assert_called_once()


class TestParallelExecutorIntegration:
    """Integration tests for ParallelExecutor."""

    @pytest.mark.asyncio
    async def test_full_pipeline_execution_integration(self):
        """Test full pipeline execution integration."""
        # Create executor with async mode for fastest execution
        config = ExecutionConfig(execution_mode=ExecutionMode.ASYNC, max_workers=4)
        executor = ParallelExecutor(config)
        executor.initialize()

        # Create a complex pipeline
        pipeline = Pipeline(id="integration_test", name="Integration Test Pipeline")

        # Level 0: Independent tasks
        task_a = Task(id="task_a", name="Task A", action="test_action")
        task_b = Task(id="task_b", name="Task B", action="test_action")

        # Level 1: Depends on level 0
        task_c = Task(
            id="task_c", name="Task C", action="test_action", dependencies=["task_a"]
        )
        task_d = Task(
            id="task_d", name="Task D", action="test_action", dependencies=["task_b"]
        )

        # Level 2: Depends on level 1
        task_e = Task(
            id="task_e",
            name="Task E",
            action="test_action",
            dependencies=["task_c", "task_d"],
        )

        pipeline.add_task(task_a)
        pipeline.add_task(task_b)
        pipeline.add_task(task_c)
        pipeline.add_task(task_d)
        pipeline.add_task(task_e)

        # Execution counter to verify parallel execution
        execution_order = []
        execution_times = {}

        async def tracking_executor(task):
            start_time = time.time()
            execution_order.append(task.id)
            await asyncio.sleep(0.05)  # Simulate some work
            execution_times[task.id] = time.time() - start_time
            return f"result_{task.id}"

        # Execute pipeline
        start_time = time.time()
        results = await executor.execute_pipeline(pipeline, tracking_executor)
        total_time = time.time() - start_time

        # Verify all tasks completed successfully
        assert len(results) == 5
        assert all(results[task_id].success for task_id in results)

        # Verify execution order respects dependencies
        # task_a and task_b should execute first (in parallel)
        # task_c and task_d should execute after their dependencies
        # task_e should execute last

        a_index = execution_order.index("task_a")
        b_index = execution_order.index("task_b")
        c_index = execution_order.index("task_c")
        d_index = execution_order.index("task_d")
        e_index = execution_order.index("task_e")

        # Level 0 tasks execute before level 1
        assert a_index < c_index
        assert b_index < d_index

        # Level 1 tasks execute before level 2
        assert c_index < e_index
        assert d_index < e_index

        # Verify parallel execution efficiency
        # Total time should be less than sum of all execution times (proving parallelism)
        sum_execution_times = sum(execution_times.values())
        assert total_time < sum_execution_times

        # Verify statistics
        stats = executor.get_execution_statistics()
        assert stats["total_tasks"] == 5
        assert stats["successful_tasks"] == 5
        assert stats["failed_tasks"] == 0
        assert stats["levels_executed"] == 3  # 3 dependency levels

        # Cleanup
        executor.shutdown()
