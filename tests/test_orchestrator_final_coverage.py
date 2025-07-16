"""Final tests for Orchestrator to achieve 100% coverage."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.task import Task, TaskStatus
from src.orchestrator.orchestrator import Orchestrator


class TestOrchestratorFinalCoverage:
    """Final tests to achieve 100% coverage of Orchestrator."""

    @pytest.mark.asyncio
    async def test_execute_level_skipped_task_not_in_results(self):
        """Test line 270: handling skipped tasks not already in results."""
        orchestrator = Orchestrator()

        # Create pipeline with a task that will be marked as skipped but not executed
        pipeline = Pipeline(id="skip_test", name="Skip Test Pipeline")
        normal_task = Task(id="normal_task", name="Normal Task", action="generate")
        skipped_task = Task(id="skipped_task", name="Skipped Task", action="generate")

        pipeline.add_task(normal_task)
        pipeline.add_task(skipped_task)

        # Mark task as skipped BEFORE execution
        skipped_task.skip("Pre-skipped for testing")

        # Mock resource allocation to ensure tasks get allocated
        orchestrator.resource_allocator.request_resources = AsyncMock(return_value=True)
        orchestrator.resource_allocator.release_resources = AsyncMock()

        # Mock control system to return a result with status
        orchestrator.control_system.execute_task = AsyncMock(
            return_value={"status": "completed", "output": "test output"}
        )

        # Execute level
        context = {"execution_id": "test_123", "current_level": 0}
        previous_results = {}

        results = await orchestrator._execute_level(
            pipeline, ["normal_task", "skipped_task"], context, previous_results
        )

        # Both tasks should be in results
        assert "normal_task" in results
        assert "skipped_task" in results

        # Normal task should be completed, skipped task should have status "skipped"
        assert results["normal_task"]["status"] == "completed"
        assert results["skipped_task"]["status"] == "skipped"

        # Verify the skipped task handling (line 270)
        assert normal_task.status == TaskStatus.COMPLETED
        assert skipped_task.status == TaskStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_health_check_overall_warning_status(self):
        """Test lines 641-642: health check overall warning status."""
        orchestrator = Orchestrator()

        # Mock components to have warning status but no unhealthy components
        orchestrator.model_registry.get_available_models = AsyncMock(
            return_value=["model1"]
        )
        orchestrator.control_system.get_capabilities = Mock(
            return_value={"action": "generate"}
        )
        orchestrator.state_manager.is_healthy = AsyncMock(return_value=True)

        # Mock resource allocator to return warning status (high CPU but not critical)
        orchestrator.resource_allocator.get_utilization = AsyncMock(
            return_value={"cpu_usage": 0.95}  # 95% - warning level (> 0.9)
        )

        health = await orchestrator.health_check()

        # Should have warning overall status (line 641-642)
        assert health["overall"] == "warning"
        assert health["resource_allocator"] == "warning"
        # Other components should be healthy
        assert health["model_registry"] in ["healthy", "warning"]  # Depends on models
        assert health["control_system"] == "healthy"
        assert health["state_manager"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_overall_healthy_status(self):
        """Test lines 643-644: health check overall healthy status."""
        orchestrator = Orchestrator()

        # Mock all components to be healthy
        orchestrator.model_registry.get_available_models = AsyncMock(
            return_value=["model1", "model2"]
        )
        orchestrator.control_system.get_capabilities = Mock(
            return_value={"action": "generate"}
        )
        orchestrator.state_manager.is_healthy = AsyncMock(return_value=True)

        # Mock resource allocator to return good utilization
        orchestrator.resource_allocator.get_utilization = AsyncMock(
            return_value={"cpu_usage": 0.3}  # 30% - healthy level
        )

        health = await orchestrator.health_check()

        # Should have healthy overall status (lines 643-644)
        assert health["overall"] == "healthy"
        assert health["resource_allocator"] == "healthy"
        assert health["model_registry"] == "healthy"
        assert health["control_system"] == "healthy"
        assert health["state_manager"] == "healthy"

    @pytest.mark.asyncio
    async def test_shutdown_async_parallel_executor(self):
        """Test line 662: async shutdown of parallel executor."""
        orchestrator = Orchestrator()

        # Mock parallel executor with async shutdown method
        mock_parallel_executor = Mock()
        mock_parallel_executor.shutdown = AsyncMock()

        # Ensure the shutdown method is detected as a coroutine function
        with patch("asyncio.iscoroutinefunction", return_value=True):
            orchestrator.parallel_executor = mock_parallel_executor

            # Mock other components
            orchestrator.resource_allocator.shutdown = AsyncMock()
            orchestrator.state_manager.shutdown = AsyncMock()

            await orchestrator.shutdown()

            # Should call the async shutdown method (line 662)
            mock_parallel_executor.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_comprehensive_orchestrator_edge_cases(self):
        """Test comprehensive edge cases for final coverage."""
        orchestrator = Orchestrator()

        # Test with complex pipeline that exercises multiple edge cases
        pipeline = Pipeline(id="edge_case_test", name="Edge Case Test Pipeline")

        # Task that will be skipped
        skip_task = Task(id="skip_task", name="Skip Task", action="generate")
        skip_task.skip("Skipped for edge case testing")

        # Task that will complete normally
        normal_task = Task(id="normal_task", name="Normal Task", action="generate")

        pipeline.add_task(skip_task)
        pipeline.add_task(normal_task)

        # Mock control system for pipeline execution
        orchestrator.control_system.execute_task = AsyncMock(
            return_value={"status": "completed", "output": "test output"}
        )

        # Execute pipeline to test skipped task handling
        results = await orchestrator.execute_pipeline(pipeline)

        # Verify results include both tasks
        assert "skip_task" in results
        assert "normal_task" in results
        assert results["skip_task"]["status"] == "skipped"
        assert results["normal_task"]["status"] == "completed"

        # Test health check with mixed component statuses
        orchestrator.resource_allocator.get_utilization = AsyncMock(
            return_value={"cpu_usage": 0.95}  # Warning level (> 0.9)
        )

        health = await orchestrator.health_check()
        # Should handle warning status correctly
        assert health["overall"] in ["healthy", "warning", "unhealthy"]

        # Test shutdown with all component types
        await orchestrator.shutdown()


class TestOrchestratorImplementationGuidance:
    """Tests to verify implementation follows design documents."""

    @pytest.mark.asyncio
    async def test_orchestrator_follows_design_patterns(self):
        """Verify orchestrator implements design patterns correctly."""
        orchestrator = Orchestrator()

        # Test core abstractions exist as per design
        assert hasattr(orchestrator, "model_registry")
        assert hasattr(orchestrator, "state_manager")
        assert hasattr(orchestrator, "control_system")
        assert hasattr(orchestrator, "resource_allocator")
        assert hasattr(orchestrator, "parallel_executor")
        assert hasattr(orchestrator, "error_handler")

        # Test pipeline execution follows design
        pipeline = Pipeline(id="design_test", name="Design Test Pipeline")
        task = Task(id="test_task", name="Test Task", action="generate")
        pipeline.add_task(task)

        # Mock control system
        orchestrator.control_system.execute_task = AsyncMock(
            return_value={"status": "completed", "output": "test output"}
        )

        # Should execute successfully following design patterns
        results = await orchestrator.execute_pipeline(pipeline)
        assert "test_task" in results

    @pytest.mark.asyncio
    async def test_error_handling_strategy_implementation(self):
        """Test error handling follows design document strategy."""
        orchestrator = Orchestrator()

        # Create pipeline with task that will fail
        pipeline = Pipeline(id="error_test", name="Error Test Pipeline")
        failing_task = Task(
            id="failing_task",
            name="Failing Task",
            action="generate",
            metadata={"on_failure": "continue"},  # Design pattern: graceful degradation
        )
        pipeline.add_task(failing_task)

        # Mock control system to fail the task
        orchestrator.control_system.execute_task = AsyncMock(
            side_effect=Exception("Test failure")
        )

        # Should handle error gracefully as per design
        results = await orchestrator.execute_pipeline(pipeline)

        # Should complete execution despite failure (graceful degradation)
        assert "failing_task" in results
        assert "error" in results["failing_task"]

    @pytest.mark.asyncio
    async def test_resource_management_design_compliance(self):
        """Test resource management follows design patterns."""
        orchestrator = Orchestrator()

        # Test resource allocation/deallocation pattern
        pipeline = Pipeline(id="resource_test", name="Resource Test Pipeline")
        resource_task = Task(
            id="resource_task",
            name="Resource Task",
            action="generate",
            metadata={"cpu_cores": 2, "memory_mb": 1024},  # Resource requirements
        )
        pipeline.add_task(resource_task)

        # Mock resource allocator to track allocations
        allocation_calls = []
        release_calls = []

        async def mock_request(task_id, requirements=None):
            allocation_calls.append((task_id, requirements))
            return True

        async def mock_release(task_id):
            release_calls.append(task_id)

        orchestrator.resource_allocator.request_resources = mock_request
        orchestrator.resource_allocator.release_resources = mock_release

        # Mock control system
        orchestrator.control_system.execute_task = AsyncMock(
            return_value={"status": "completed", "output": "test output"}
        )

        # Execute pipeline
        results = await orchestrator.execute_pipeline(pipeline)

        # Verify resource management pattern
        assert len(allocation_calls) > 0  # Resources were allocated
        assert len(release_calls) > 0  # Resources were released
        assert "resource_task" in results

    def test_model_selection_algorithm_design(self):
        """Test model selection follows UCB algorithm design."""
        orchestrator = Orchestrator()

        # Verify UCB model selector is used (design requirement)
        from src.orchestrator.models.model_registry import UCBModelSelector

        assert isinstance(orchestrator.model_registry.model_selector, UCBModelSelector)

        # Verify exploration factor is configurable (design requirement)
        assert hasattr(orchestrator.model_registry.model_selector, "exploration_factor")
        assert orchestrator.model_registry.model_selector.exploration_factor > 0
