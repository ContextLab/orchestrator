"""
Tests for final missing lines in parallel_executor.py.

Targeting:
- Line 118: return {} when no stats
- Line 321: return {} when no tasks
- Lines 351-352: unexpected result type handling
"""

import pytest
from unittest.mock import AsyncMock

from orchestrator.executor.parallel_executor import ParallelExecutor, ResourceMonitor
from orchestrator.core.pipeline import Pipeline
from orchestrator.core.task import Task, TaskStatus


class TestParallelExecutorFinalLines:
    """Test cases for final missing lines in parallel executor."""
    
    def test_resource_monitor_empty_usage_stats_line_111(self):
        """Test line 111: return {} when usage_stats is empty."""
        monitor = ResourceMonitor()
        monitor.usage_stats.clear()
        stats = monitor.get_statistics()
        assert stats == {}
    
    def test_resource_monitor_no_stats_line_118(self):
        """Test line 118: return {} when all_stats is empty but usage_stats is not."""
        monitor = ResourceMonitor()
        
        # usage_stats with empty lists (line 118)
        monitor.usage_stats['task1'] = []
        monitor.usage_stats['task2'] = []
        monitor.usage_stats['task3'] = []
        
        # This should trigger line 118 - usage_stats is not empty but all_stats is
        stats = monitor.get_statistics()
        assert stats == {}
        
    @pytest.mark.asyncio
    async def test_execute_level_no_tasks_line_321(self):
        """Test line 321: return {} when no tasks to execute."""
        executor = ParallelExecutor()
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        
        # Empty task list
        result = await executor._execute_level(
            [],  # Empty task list
            pipeline,
            lambda t: "result",
            0
        )
        
        assert result == {}
        
    @pytest.mark.asyncio
    async def test_execute_level_unexpected_result_lines_351_352(self):
        """Test lines 351-352: handling unexpected result type."""
        executor = ParallelExecutor()
        
        # Create pipeline with task
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task = Task(id="task1", name="Task 1", action="generate")
        pipeline.add_task(task)
        
        # Mock worker pool to return unexpected type
        executor.worker_pool.execute_task = AsyncMock(return_value="unexpected_string_result")
        
        # Execute level
        result = await executor._execute_level(
            ["task1"],
            pipeline,
            lambda t: "result",
            0
        )
        
        # Check task was marked as failed
        assert task.status == TaskStatus.FAILED
        assert "Unexpected result type" in str(task.error)
        
        # Check result
        assert "task1" in result
        assert result["task1"].success is False
        assert "Unexpected result type" in str(result["task1"].error)