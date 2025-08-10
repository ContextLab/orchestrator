"""Integration test for Orchestrator with automatic checkpointing - Issue #205

This test validates that the orchestrator correctly integrates with the new
automatic checkpointing system.
"""

import asyncio
import pytest
import logging

from orchestrator.orchestrator import Orchestrator
from orchestrator.core.pipeline import Pipeline
from orchestrator.core.task import Task
from orchestrator.checkpointing import ExecutionRecoveryStrategy
from orchestrator.control_systems.hybrid_control_system import HybridControlSystem
from orchestrator.models.registry_singleton import get_model_registry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_orchestrator_automatic_checkpointing_integration():
    """Test orchestrator with automatic checkpointing enabled."""
    logger.info("Starting orchestrator checkpointing integration test")
    
    # Get model registry and create control system
    model_registry = get_model_registry()
    # Add a simple mock model for testing
    if not model_registry.models:
        class MockModel:
            def __init__(self):
                self.model_id = "test_model"
            
            async def generate_response(self, prompt, **kwargs):
                return "Mock response"
                
        model_registry.models["test_model"] = MockModel()
    
    control_system = HybridControlSystem(model_registry)
    
    # Create orchestrator with automatic checkpointing enabled
    orchestrator = Orchestrator(
        control_system=control_system,
        use_langgraph_state=True,
        use_automatic_checkpointing=True,
        checkpoint_frequency="every_step",
        max_checkpoint_overhead_ms=100.0,
        recovery_strategy=ExecutionRecoveryStrategy.RESUME_FROM_LAST_CHECKPOINT,
        max_recovery_attempts=2
    )
    
    # Verify checkpointing is enabled
    assert orchestrator.is_automatic_checkpointing_enabled()
    
    # Create a simple test pipeline
    pipeline = Pipeline(
        id="integration_test_pipeline",
        name="Integration Test Pipeline"
    )
    
    # Add test tasks
    tasks = [
        Task("task_1", "test", {"output": "Task 1 result"}),
        Task("task_2", "test", {"output": "Task 2 result"}),
        Task("task_3", "test", {"output": "Task 3 result"}),
    ]
    
    for i, task in enumerate(tasks):
        pipeline.add_task(task)
        if i > 0:
            task.dependencies = [f"task_{i}"]
    
    # Execute pipeline with checkpointing
    try:
        result = await orchestrator.execute_pipeline(
            pipeline=pipeline,
            checkpoint_enabled=True
        )
        
        # Validate result
        assert result["status"] == "success"
        assert "checkpoint_count" in result
        assert result["checkpoint_count"] > 0
        assert "metadata" in result
        assert result["metadata"]["automatic_checkpointing"] == True
        
        logger.info(f"✅ Integration test passed! Checkpoints created: {result['checkpoint_count']}")
        
        # Get checkpoint performance metrics
        metrics = orchestrator.get_checkpoint_performance_metrics()
        assert metrics["automatic_checkpointing_enabled"] == True
        assert "total_executions" in metrics
        
        logger.info(f"Performance metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        raise
    
    logger.info("✅ test_orchestrator_automatic_checkpointing_integration PASSED")


@pytest.mark.asyncio 
async def test_orchestrator_without_checkpointing():
    """Test orchestrator with automatic checkpointing disabled."""
    logger.info("Starting orchestrator without checkpointing test")
    
    # Get model registry and create control system
    model_registry = get_model_registry()
    # Add a simple mock model for testing
    if not model_registry.models:
        class MockModel:
            def __init__(self):
                self.model_id = "test_model"
            
            async def generate_response(self, prompt, **kwargs):
                return "Mock response"
                
        model_registry.models["test_model"] = MockModel()
    
    control_system = HybridControlSystem(model_registry)
    
    # Create orchestrator without automatic checkpointing
    orchestrator = Orchestrator(
        control_system=control_system,
        use_langgraph_state=False,  # Disable LangGraph
        use_automatic_checkpointing=False
    )
    
    # Verify checkpointing is disabled
    assert not orchestrator.is_automatic_checkpointing_enabled()
    
    # Create simple pipeline
    pipeline = Pipeline(
        id="no_checkpoint_test",
        name="No Checkpoint Test"
    )
    
    task = Task("simple_task", "test", {"output": "Simple result"})
    pipeline.add_task(task)
    
    # Execute without checkpointing
    result = await orchestrator.execute_pipeline(pipeline)
    
    # Should not have checkpoint information
    assert "checkpoint_count" not in result
    assert result.get("metadata", {}).get("automatic_checkpointing") != True
    
    # Metrics should indicate checkpointing is disabled
    metrics = orchestrator.get_checkpoint_performance_metrics()
    assert metrics["automatic_checkpointing_enabled"] == False
    
    logger.info("✅ test_orchestrator_without_checkpointing PASSED")


if __name__ == "__main__":
    import asyncio
    
    async def run_manual_integration():
        """Run integration tests manually."""
        print("Running manual integration tests...")
        
        try:
            await test_orchestrator_automatic_checkpointing_integration()
            await test_orchestrator_without_checkpointing()
            print("✅ All integration tests passed!")
        except Exception as e:
            print(f"❌ Integration tests failed: {e}")
    
    asyncio.run(run_manual_integration())