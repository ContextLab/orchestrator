"""Test process pool execution with real picklable functions."""

from concurrent.futures import ProcessPoolExecutor

import pytest

from src.orchestrator.core.task import Task
from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
from src.orchestrator.executor.parallel_executor import (

    ExecutionConfig,
    ExecutionMode,
    WorkerPool)


# Module-level function that can be pickled
def process_task(task):
    """Simple task processor that can be pickled."""
    return f"Processed {task.id} in process pool"


def slow_process_task(task):
    """Slow task processor for timeout testing."""
    import time

    time.sleep(2)  # Sleep longer than timeout
    return f"Processed {task.id}"


class TestProcessPoolExecution:
    """Test real process pool execution."""

    @pytest.mark.asyncio
    async def test_process_pool_success(self):
        """Test successful process pool execution."""
        config = ExecutionConfig(execution_mode=ExecutionMode.PROCESS)
        worker_pool = WorkerPool(config)

        # Initialize real process pool
        worker_pool.process_pool = ProcessPoolExecutor(max_workers=1)

        try:
            task = Task(id="test_task", name="Test Task", action="generate")

            # Use module-level function
            result = await worker_pool._execute_process(task, process_task)

            assert result == "Processed test_task in process pool"
        finally:
            worker_pool.shutdown()

    @pytest.mark.asyncio
    async def test_process_pool_timeout(self):
        """Test process pool timeout."""
        config = ExecutionConfig(execution_mode=ExecutionMode.PROCESS, timeout=0.5)
        worker_pool = WorkerPool(config)

        # Initialize real process pool
        worker_pool.process_pool = ProcessPoolExecutor(max_workers=1)

        try:
            task = Task(id="slow_task", name="Slow Task", action="generate")

            with pytest.raises(TimeoutError) as exc_info:
                await worker_pool._execute_process(task, slow_process_task)

            assert "timed out after 0.5s" in str(exc_info.value)
        finally:
            worker_pool.shutdown()

    @pytest.mark.asyncio
    async def test_process_pool_not_initialized(self):
        """Test error when process pool not initialized."""
        worker_pool = WorkerPool(ExecutionConfig())
        worker_pool.process_pool = None

        task = Task(id="test_task", name="Test Task", action="generate")

        with pytest.raises(RuntimeError) as exc_info:
            await worker_pool._execute_process(task, process_task)

        assert "Process pool not initialized" in str(exc_info.value)
