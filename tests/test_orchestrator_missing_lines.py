"""Tests to cover specific missing lines in Orchestrator."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.task import Task, TaskStatus


class TestOrchestratorMissingLines:
    """Tests to cover specific missing lines in orchestrator."""
    
    @pytest.mark.asyncio
    async def test_skipped_task_not_in_results_line_270(self):
        """Test line 270: results[task_id] = {"status": "skipped"}."""
        orchestrator = Orchestrator()
        
        # Create a task and mark it as skipped
        task = Task(id="skipped_task", name="Skipped Task", action="generate")
        task.skip("Skipped for testing")
        
        # Create pipeline
        pipeline = Pipeline(id="test", name="Test")
        pipeline.add_task(task)
        
        # Mock resource allocation
        orchestrator.resource_allocator.request_resources = AsyncMock(return_value=True)
        orchestrator.resource_allocator.release_resources = AsyncMock()
        
        # Execute level with the skipped task
        context = {"execution_id": "test_123", "current_level": 0}
        results = await orchestrator._execute_level(
            pipeline, 
            ["skipped_task"], 
            context, 
            {}
        )
        
        # Should have the skipped task in results (line 270)
        assert "skipped_task" in results
        assert results["skipped_task"]["status"] == "skipped"
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
        orchestrator.model_registry.get_available_models = AsyncMock(return_value=["model1"])
        orchestrator.control_system.get_capabilities = Mock(return_value={"test": "value"})
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
        orchestrator.model_registry.get_available_models = AsyncMock(return_value=["model1", "model2"])
        orchestrator.control_system.get_capabilities = Mock(return_value={"test": "value"})
        orchestrator.state_manager.is_healthy = AsyncMock(return_value=True)
        
        health = await orchestrator.health_check()
        
        # Should trigger lines 643-644: healthy overall status
        assert health["overall"] == "healthy"
        assert all(status in ["healthy", "warning"] for status in health.values() if status != "healthy")