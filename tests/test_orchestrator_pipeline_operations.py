"""Tests for enhanced pipeline operations with LangGraph state management."""

import pytest
import time
from src.orchestrator import init_models
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.task import Task


@pytest.mark.asyncio
class TestOrchestratorPipelineOperations:
    """Test enhanced pipeline operations with LangGraph integration."""
    
    async def test_enhanced_pipeline_state_capture(self):
        """Test that enhanced pipeline state is captured when using LangGraph."""
        # Initialize models
        init_models()
        
        # Create orchestrator with LangGraph enabled
        orchestrator = Orchestrator(use_langgraph_state=True)
        
        # Create a simple pipeline
        pipeline = Pipeline(
            id="test_enhanced_pipeline",
            name="Enhanced Pipeline Test",
            context={"input_text": "test input"},
            metadata={"description": "Enhanced state capture test"}
        )
        
        # Add a simple task
        task = Task(
            id="enhanced_task",
            name="Enhanced Task",
            action="prompt",
            parameters={
                "prompt": "Process: {{input_text}}",
                "model": "test_model"
            }
        )
        pipeline.add_task(task)
        
        # Get enhanced pipeline state
        enhanced_state = orchestrator._get_pipeline_state(pipeline)
        
        # Verify enhanced metadata is included
        assert "execution_metadata" in enhanced_state
        assert "performance_metrics" in enhanced_state
        
        # Verify enhanced metadata content
        exec_meta = enhanced_state["execution_metadata"]
        assert "orchestrator_version" in exec_meta
        assert "model_registry_size" in exec_meta
        assert "control_system_type" in exec_meta
        assert "langgraph_storage_type" in exec_meta
        
        perf_metrics = enhanced_state["performance_metrics"]
        assert "running_pipelines_count" in perf_metrics
        assert "total_execution_history" in perf_metrics
        
    async def test_legacy_pipeline_state_capture(self):
        """Test that legacy pipeline state is captured when not using LangGraph."""
        # Initialize models
        init_models()
        
        # Create orchestrator with legacy state management
        orchestrator = Orchestrator(use_langgraph_state=False)
        
        # Create a simple pipeline  
        pipeline = Pipeline(
            id="test_legacy_pipeline",
            name="Legacy Pipeline Test",
            context={"input_text": "test input"},
            metadata={"description": "Legacy state capture test"}
        )
        
        # Add a simple task
        task = Task(
            id="legacy_task",
            name="Legacy Task",
            action="prompt",
            parameters={
                "prompt": "Process: {{input_text}}",
                "model": "test_model"
            }
        )
        pipeline.add_task(task)
        
        # Get basic pipeline state
        basic_state = orchestrator._get_pipeline_state(pipeline)
        
        # Verify enhanced metadata is NOT included
        assert "execution_metadata" not in basic_state
        assert "performance_metrics" not in basic_state
        
        # Verify basic fields are present
        assert "id" in basic_state
        assert "tasks" in basic_state
        assert "context" in basic_state
        assert "metadata" in basic_state
        
    async def test_enhanced_result_processing(self):
        """Test processing of enhanced task results."""
        # Initialize models
        init_models()
        
        orchestrator = Orchestrator(use_langgraph_state=True)
        
        # Test enhanced result processing
        enhanced_results = {
            "task1": {
                "result": "actual output",
                "task_metadata": {
                    "task_id": "task1",
                    "execution_time": 1.5,
                    "status": "completed"
                }
            },
            "task2": {
                "result": "another output",
                "task_metadata": {
                    "task_id": "task2", 
                    "execution_time": 0.8,
                    "status": "completed"
                }
            }
        }
        
        processed = orchestrator._process_enhanced_results(enhanced_results)
        
        # Verify actual results are extracted
        assert processed["task1"] == "actual output"
        assert processed["task2"] == "another output"
        
    async def test_legacy_result_processing(self):
        """Test that legacy results pass through unchanged."""
        # Initialize models
        init_models()
        
        orchestrator = Orchestrator(use_langgraph_state=False)
        
        # Test legacy result processing
        legacy_results = {
            "task1": "direct output",
            "task2": "another direct output"
        }
        
        processed = orchestrator._process_enhanced_results(legacy_results)
        
        # Verify results pass through unchanged
        assert processed == legacy_results
        
    async def test_global_state_methods_integration(self):
        """Test that the new global state methods work correctly."""
        # Initialize models
        init_models()
        
        orchestrator = Orchestrator(use_langgraph_state=True)
        
        # Test methods that require actual execution
        # These should not crash and should return appropriate types
        
        # Test get_pipeline_global_state with non-existent execution
        state = await orchestrator.get_pipeline_global_state("nonexistent_execution")
        assert state is None
        
        # Test get_pipeline_metrics with non-existent execution
        metrics = await orchestrator.get_pipeline_metrics("nonexistent_execution")
        assert metrics is None
        
        # Test create_named_checkpoint with non-existent execution
        checkpoint_id = await orchestrator.create_named_checkpoint(
            "nonexistent_execution", 
            "test_checkpoint", 
            "Test checkpoint"
        )
        assert checkpoint_id is None
        
    async def test_global_state_methods_legacy_error(self):
        """Test that global state methods raise errors in legacy mode."""
        # Initialize models
        init_models()
        
        orchestrator = Orchestrator(use_langgraph_state=False)
        
        # Test that methods raise appropriate errors in legacy mode
        with pytest.raises(ValueError, match="Global state only available"):
            await orchestrator.get_pipeline_global_state("test_execution")
            
        with pytest.raises(ValueError, match="Named checkpoints only available"):
            await orchestrator.create_named_checkpoint("test_execution", "test", "desc")
            
        with pytest.raises(ValueError, match="Pipeline metrics only available"):
            await orchestrator.get_pipeline_metrics("test_execution")