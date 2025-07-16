"""Parallel execution engine for task orchestration."""

import asyncio
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from ..core.pipeline import Pipeline
from ..core.task import Task


# Module-level function for picklability in process pools
def _process_pool_executor_wrapper(executor_func_and_task):
    """Wrapper function that can be pickled for process pool execution."""
    executor_func, task = executor_func_and_task
    return executor_func(task)


class ExecutionMode(Enum):
    """Execution mode for parallel processing."""

    ASYNC = "async"
    THREAD = "thread"
    PROCESS = "process"


@dataclass
class ExecutionConfig:
    """Configuration for parallel execution."""

    max_workers: int = 4
    execution_mode: ExecutionMode = ExecutionMode.ASYNC
    timeout: float = 300.0  # 5 minutes
    resource_limits: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.resource_limits:
            self.resource_limits = {"memory_limit": "512m", "cpu_limit": 1.0}


@dataclass
class ExecutionResult:
    """Result of parallel execution."""

    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    worker_id: Optional[str] = None


class ResourceMonitor:
    """Monitor resource usage during execution."""

    def __init__(self):
        self.usage_stats = defaultdict(list)
        self.active_tasks = {}

    def start_monitoring(self, task_id: str):
        """Start monitoring a task."""
        self.active_tasks[task_id] = {
            "start_time": time.time(),
            "memory_start": self._get_memory_usage(),
            "cpu_start": self._get_cpu_usage(),
        }

    def stop_monitoring(self, task_id: str) -> Dict[str, float]:
        """Stop monitoring and return usage stats."""
        if task_id not in self.active_tasks:
            return {}

        stats = self.active_tasks.pop(task_id)
        execution_time = time.time() - stats["start_time"]
        memory_used = self._get_memory_usage() - stats["memory_start"]
        cpu_used = self._get_cpu_usage() - stats["cpu_start"]

        usage = {
            "execution_time": execution_time,
            "memory_used": memory_used,
            "cpu_used": cpu_used,
        }

        self.usage_stats[task_id].append(usage)
        return usage

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil

            return psutil.cpu_percent()
        except ImportError:
            return 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        if not self.usage_stats:
            return {}

        all_stats = []
        for task_stats in self.usage_stats.values():
            all_stats.extend(task_stats)

        if not all_stats:
            return {}

        return {
            "total_executions": len(all_stats),
            "avg_execution_time": sum(s["execution_time"] for s in all_stats)
            / len(all_stats),
            "avg_memory_used": sum(s["memory_used"] for s in all_stats)
            / len(all_stats),
            "avg_cpu_used": sum(s["cpu_used"] for s in all_stats) / len(all_stats),
            "max_execution_time": max(s["execution_time"] for s in all_stats),
            "max_memory_used": max(s["memory_used"] for s in all_stats),
            "active_tasks": len(self.active_tasks),
        }


class WorkerPool:
    """Manages worker pools for different execution modes."""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.thread_pool = None
        self.process_pool = None
        self.resource_monitor = ResourceMonitor()

    def initialize(self):
        """Initialize worker pools."""
        if self.config.execution_mode == ExecutionMode.THREAD:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        elif self.config.execution_mode == ExecutionMode.PROCESS:
            self.process_pool = ProcessPoolExecutor(max_workers=self.config.max_workers)

    async def execute_task(
        self, task: Task, executor_func: Callable
    ) -> ExecutionResult:
        """Execute task using appropriate worker pool."""
        self.resource_monitor.start_monitoring(task.id)
        start_time = time.time()

        try:
            if self.config.execution_mode == ExecutionMode.ASYNC:
                result = await self._execute_async(task, executor_func)
            elif self.config.execution_mode == ExecutionMode.THREAD:
                result = await self._execute_thread(task, executor_func)
            elif self.config.execution_mode == ExecutionMode.PROCESS:
                result = await self._execute_process(task, executor_func)
            else:
                raise ValueError(
                    f"Unsupported execution mode: {self.config.execution_mode}"
                )

            execution_time = time.time() - start_time
            resource_usage = self.resource_monitor.stop_monitoring(task.id)

            return ExecutionResult(
                task_id=task.id,
                success=True,
                result=result,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.resource_monitor.stop_monitoring(task.id)

            return ExecutionResult(
                task_id=task.id, success=False, error=e, execution_time=execution_time
            )

    async def _execute_async(self, task: Task, executor_func: Callable) -> Any:
        """Execute task asynchronously."""
        try:
            return await asyncio.wait_for(
                executor_func(task), timeout=self.config.timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task {task.id} timed out after {self.config.timeout}s")

    async def _execute_thread(self, task: Task, executor_func: Callable) -> Any:
        """Execute task in thread pool."""
        if not self.thread_pool:
            raise RuntimeError("Thread pool not initialized")

        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(self.thread_pool, lambda: executor_func(task)),
                timeout=self.config.timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task {task.id} timed out after {self.config.timeout}s")

    async def _execute_process(self, task: Task, executor_func: Callable) -> Any:
        """Execute task in process pool."""
        if not self.process_pool:
            raise RuntimeError("Process pool not initialized")

        loop = asyncio.get_event_loop()
        try:
            # Use the module-level wrapper function that can be pickled
            return await asyncio.wait_for(
                loop.run_in_executor(
                    self.process_pool,
                    _process_pool_executor_wrapper,
                    (executor_func, task),
                ),
                timeout=self.config.timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task {task.id} timed out after {self.config.timeout}s")

    def shutdown(self):
        """Shutdown worker pools."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)

    def get_statistics(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        return {
            "execution_mode": self.config.execution_mode.value,
            "max_workers": self.config.max_workers,
            "timeout": self.config.timeout,
            "resource_usage": self.resource_monitor.get_statistics(),
        }


class ParallelExecutor:
    """Parallel execution engine for pipelines."""

    def __init__(self, config: ExecutionConfig = None):
        self.config = config or ExecutionConfig()
        self.worker_pool = WorkerPool(self.config)
        self.execution_graph = {}
        self.task_results = {}
        self.execution_stats = defaultdict(dict)

    def initialize(self):
        """Initialize the parallel executor."""
        self.worker_pool.initialize()

    async def execute_pipeline(
        self, pipeline: Pipeline, executor_func: Callable[[Task], Awaitable[Any]]
    ) -> Dict[str, ExecutionResult]:
        """Execute pipeline tasks in parallel where possible."""
        # Build execution graph
        self.execution_graph = self._build_execution_graph(pipeline)
        self.task_results = {}

        # Get execution levels (parallel groups)
        execution_levels = pipeline.get_execution_levels()

        # Execute each level in parallel
        for level_index, task_ids in enumerate(execution_levels):
            level_results = await self._execute_level(
                task_ids, pipeline, executor_func, level_index
            )
            self.task_results.update(level_results)

            # Check for failures that should stop execution
            failed_tasks = [r for r in level_results.values() if not r.success]
            if failed_tasks:
                critical_failures = [
                    r for r in failed_tasks if self._is_critical_failure(r, pipeline)
                ]
                if critical_failures:
                    # Stop execution on critical failures
                    break

        return self.task_results

    async def _execute_level(
        self,
        task_ids: List[str],
        pipeline: Pipeline,
        executor_func: Callable,
        level_index: int,
    ) -> Dict[str, ExecutionResult]:
        """Execute all tasks in a level concurrently."""
        tasks = []

        for task_id in task_ids:
            task = pipeline.get_task(task_id)
            if task is None:
                raise ValueError(f"Task '{task_id}' not found in pipeline")

            # Check if dependencies are satisfied
            if self._are_dependencies_satisfied(task, pipeline):
                # Start task execution
                task.start()

                # Create execution coroutine
                execution_coro = self.worker_pool.execute_task(task, executor_func)
                tasks.append((task_id, execution_coro))
            else:
                # Skip task if dependencies not satisfied
                result = ExecutionResult(
                    task_id=task_id,
                    success=False,
                    error=Exception("Dependencies not satisfied"),
                    execution_time=0.0,
                )
                tasks.append((task_id, self._create_immediate_result(result)))

        # Execute all tasks in this level concurrently
        if not tasks:
            return {}

        results = await asyncio.gather(
            *[coro for _, coro in tasks], return_exceptions=True
        )

        # Process results
        level_results = {}
        for (task_id, _), result in zip(tasks, results):
            task = pipeline.get_task(task_id)

            if isinstance(result, Exception):
                # Handle execution exception
                task.fail(result)
                execution_result = ExecutionResult(
                    task_id=task_id, success=False, error=result, execution_time=0.0
                )
            elif isinstance(result, ExecutionResult):
                # Process execution result
                if result.success:
                    task.complete(result.result)
                else:
                    task.fail(result.error)
                execution_result = result
            else:
                # Unexpected result type
                task.fail(Exception(f"Unexpected result type: {type(result)}"))
                execution_result = ExecutionResult(
                    task_id=task_id,
                    success=False,
                    error=Exception("Unexpected result type"),
                    execution_time=0.0,
                )

            level_results[task_id] = execution_result

            # Record execution statistics
            self.execution_stats[task_id] = {
                "level": level_index,
                "execution_time": execution_result.execution_time,
                "success": execution_result.success,
                "worker_id": execution_result.worker_id,
            }

        return level_results

    def _build_execution_graph(self, pipeline: Pipeline) -> Dict[str, Set[str]]:
        """Build execution dependency graph."""
        graph = {}
        for task_id in pipeline:
            task = pipeline.get_task(task_id)
            graph[task_id] = set(task.dependencies)
        return graph

    def _are_dependencies_satisfied(self, task: Task, pipeline: Pipeline) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in self.task_results:
                return False

            dep_result = self.task_results[dep_id]
            if not dep_result.success:
                return False

        return True

    def _is_critical_failure(self, result: ExecutionResult, pipeline: Pipeline) -> bool:
        """Determine if failure is critical and should stop execution."""
        # For now, consider all failures as non-critical
        # This can be enhanced with task metadata or configuration
        return False

    async def _create_immediate_result(
        self, result: ExecutionResult
    ) -> ExecutionResult:
        """Create an immediate result for synchronous returns."""
        return result

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_stats:
            return {}

        total_time = sum(
            stats["execution_time"] for stats in self.execution_stats.values()
        )

        successful_tasks = sum(
            1 for stats in self.execution_stats.values() if stats["success"]
        )

        level_stats = defaultdict(list)
        for stats in self.execution_stats.values():
            level_stats[stats["level"]].append(stats["execution_time"])

        return {
            "total_tasks": len(self.execution_stats),
            "successful_tasks": successful_tasks,
            "failed_tasks": len(self.execution_stats) - successful_tasks,
            "total_execution_time": total_time,
            "avg_task_time": total_time / len(self.execution_stats),
            "levels_executed": len(level_stats),
            "level_statistics": {
                level: {
                    "task_count": len(times),
                    "total_time": sum(times),
                    "avg_time": sum(times) / len(times),
                    "max_time": max(times),
                }
                for level, times in level_stats.items()
            },
            "worker_pool_stats": self.worker_pool.get_statistics(),
        }

    def shutdown(self):
        """Shutdown the parallel executor."""
        self.worker_pool.shutdown()
