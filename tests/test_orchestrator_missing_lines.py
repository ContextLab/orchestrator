"""Tests to cover specific missing lines in Orchestrator."""

from typing import Dict, Any, Optional, List

import pytest

from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.task import Task, TaskStatus
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.core.resource_allocator import ResourceAllocator
from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.core.control_system import ControlSystem
from src.orchestrator.state.state_manager import StateManager


class TestableResourceAllocator(ResourceAllocator):
    """Testable resource allocator with configurable behavior."""
    
    def __init__(self):
        super().__init__()
        self._test_allocation_results = {}
        self._test_utilization = {"cpu_usage": 0.3}
        self.allocation_calls = []
        self.release_calls = []
        
    def set_allocation_result(self, task_id: str, result: bool):
        """Set allocation result for a task."""
        self._test_allocation_results[task_id] = result
        
    def set_utilization(self, utilization: Dict[str, float]):
        """Set test utilization metrics."""
        self._test_utilization = utilization
        
    async def request_resources(self, task_id: str, requirements: Dict[str, Any] = None) -> bool:
        """Test version of request_resources."""
        self.allocation_calls.append((task_id, requirements))
        return self._test_allocation_results.get(task_id, True)
        
    async def release_resources(self, task_id: str):
        """Test version of release_resources."""
        self.release_calls.append(task_id)
        
    async def get_utilization(self) -> Dict[str, float]:
        """Test version of get_utilization."""
        return self._test_utilization


class TestableModelRegistry(ModelRegistry):
    """Testable model registry."""
    
    def __init__(self):
        super().__init__()
        self._test_models = []
        self.call_history = []
        
    def set_test_models(self, models: List[str]):
        """Set test models."""
        self._test_models = models
        
    async def get_available_models(self) -> List[str]:
        """Test version of get_available_models."""
        self.call_history.append('get_available_models')
        return self._test_models


class TestableControlSystem(ControlSystem):
    """Testable control system."""
    
    def __init__(self, name="test-control", config=None):
        super().__init__(name, config)
        self._test_capabilities = {}
        self.call_history = []
        
    def set_capabilities(self, capabilities: Dict[str, Any]):
        """Set test capabilities."""
        self._test_capabilities = capabilities
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Return test capabilities."""
        self.call_history.append('get_capabilities')
        return self._test_capabilities


class TestableStateManager(StateManager):
    """Testable state manager."""
    
    def __init__(self):
        super().__init__()
        self._test_healthy = True
        self.call_history = []
        
    def set_healthy(self, healthy: bool):
        """Set health status."""
        self._test_healthy = healthy
        
    async def is_healthy(self) -> bool:
        """Test version of is_healthy."""
        self.call_history.append('is_healthy')
        return self._test_healthy


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
        # Create testable resource allocator
        test_allocator = TestableResourceAllocator()
        test_allocator.set_allocation_result("test_task", False)  # Fail allocation
        
        orchestrator = Orchestrator(resource_allocator=test_allocator)

        # Create a task
        task = Task(id="test_task", name="Test Task", action="generate")

        # Create pipeline
        pipeline = Pipeline(id="test", name="Test")
        pipeline.add_task(task)

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
        # Create testable components
        test_allocator = TestableResourceAllocator()
        test_allocator.set_utilization({"cpu_usage": 0.95})  # 95% triggers warning
        
        test_registry = TestableModelRegistry()
        test_registry.set_test_models(["model1"])  # Only one model
        
        test_control = TestableControlSystem()
        test_control.set_capabilities({"test": "value"})
        
        test_state = TestableStateManager()
        test_state.set_healthy(True)
        
        orchestrator = Orchestrator(
            resource_allocator=test_allocator,
            model_registry=test_registry,
            control_system=test_control,
            state_manager=test_state
        )

        health = await orchestrator.health_check()

        # Should trigger lines 641-642: warning overall status
        assert health["overall"] == "warning"
        assert health["resource_allocator"] == "warning"

    @pytest.mark.asyncio
    async def test_health_check_healthy_overall_lines_643_644(self):
        """Test lines 643-644: else: health_status["overall"] = "healthy"."""
        # Create all healthy components
        test_allocator = TestableResourceAllocator()
        test_allocator.set_utilization({"cpu_usage": 0.3})  # 30% is healthy
        
        test_registry = TestableModelRegistry()
        test_registry.set_test_models(["model1", "model2"])  # Multiple models - healthy
        
        test_control = TestableControlSystem()
        test_control.set_capabilities({"test": "value"})
        
        test_state = TestableStateManager()
        test_state.set_healthy(True)
        
        orchestrator = Orchestrator(
            resource_allocator=test_allocator,
            model_registry=test_registry,
            control_system=test_control,
            state_manager=test_state
        )

        health = await orchestrator.health_check()

        # Should trigger lines 643-644: healthy overall status
        assert health["overall"] == "healthy"
        assert all(
            status in ["healthy", "warning"]
            for status in health.values()
            if status != "healthy"
        )
