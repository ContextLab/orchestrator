"""Tests to cover specific missing lines in Orchestrator."""

from unittest.mock import AsyncMock, Mock

import pytest

from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.task import Task, TaskStatus
from src.orchestrator.orchestrator import Orchestrator


class TestOrchestratorMissingLines:
    """Tests to cover specific missing lines in orchestrator."""

    @pytest.mark.asyncio
    async def test_task_not_found_in_pipeline_line_207(self):
        """Test line 207: raise ValueError(f"Task '{task_id}' not found in pipeline")."""
        orchestrator = Orchestrator()

        # Create pipeline with no tasks
        pipeline = Pipeline(id="test", name="Test")

        # Try to execute level with non-existent task ID
        context = {"execution_id": "test_123", "current_level": 0}

        # This should trigger line 207 - task not found error
        with pytest.raises(
            ValueError, match="Task 'nonexistent_task' not found in pipeline"
        ):
            await orchestrator._execute_level(
                pipeline, ["nonexistent_task"], context, {}
            )

    @pytest.mark.asyncio
    async def test_skipped_task_not_in_results_line_272(self):
        """Test line 272: results[task_id] = {"status": "skipped"}."""

        orchestrator = Orchestrator()

        # Create a task
        task = Task(id="test_task", name="Test Task", action="generate")

        # Create pipeline
        pipeline = Pipeline(id="test", name="Test")
        pipeline.add_task(task)

        # Mock resource allocation to fail so task won't be scheduled for execution
        orchestrator.resource_allocator.request_resources = AsyncMock(
            return_value=False
        )
        orchestrator.resource_allocator.release_resources = AsyncMock()

        context = {"execution_id": "test_123", "current_level": 0}

        # Manually skip the task after it doesn't get scheduled
        task.skip("Resource allocation failed")

        # Execute the level - task won't be in main execution due to resource failure
        # but will be caught in the cleanup loop (line 272)
        results = await orchestrator._execute_level(
            pipeline, ["test_task"], context, {}
        )

        # Task should be in results via line 272 cleanup
        assert "test_task" in results
        assert results["test_task"]["status"] == "skipped"
        assert task.status == TaskStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_health_check_warning_overall_lines_641_642(self):
        """Test lines 641-642: elif any(status == "warning"...)."""
        orchestrator = Orchestrator()

        # Mock resource allocator to return warning status (high CPU)
        orchestrator.resource_allocator.get_utilization = AsyncMock(
            return_value={"cpu_usage": 0.95}  # 95% triggers warning (>0.9)
        )

        # Mock other components as healthy
        orchestrator.model_registry.get_available_models = AsyncMock(
            return_value=["model1"]
        )
        orchestrator.control_system.get_capabilities = Mock(
            return_value={"test": "value"}
        )
        orchestrator.state_manager.is_healthy = AsyncMock(return_value=True)

        health = await orchestrator.health_check()

        # Should trigger lines 641-642: warning overall status
        assert health["overall"] == "warning"
        assert health["resource_allocator"] == "warning"

    @pytest.mark.asyncio
    async def test_health_check_healthy_overall_lines_643_644(self):
        """Test lines 643-644: else: health_status["overall"] = "healthy"."""
        orchestrator = Orchestrator()

        # Mock all components as healthy
        orchestrator.resource_allocator.get_utilization = AsyncMock(
            return_value={"cpu_usage": 0.3}  # 30% is healthy
        )
        orchestrator.model_registry.get_available_models = AsyncMock(
            return_value=["model1", "model2"]
        )
        orchestrator.control_system.get_capabilities = Mock(
            return_value={"test": "value"}
        )
        orchestrator.state_manager.is_healthy = AsyncMock(return_value=True)

        health = await orchestrator.health_check()

        # Should trigger lines 643-644: healthy overall status
        assert health["overall"] == "healthy"
        assert all(
            status in ["healthy", "warning"]
            for status in health.values()
            if status != "healthy"
        )
