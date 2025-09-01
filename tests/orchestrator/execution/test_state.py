"""
Comprehensive tests for the State Management and Execution Context System.

Tests cover execution context management, state persistence, checkpoints,
metrics tracking, and integration scenarios.
"""

import pytest
import tempfile
import shutil
import json
import pickle
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.orchestrator.execution.state import (
    ExecutionContext,
    FileStateManager,
    ExecutionStatus,
    ExecutionMetrics,
    Checkpoint,
    PersistenceFormat,
    create_execution_context,
    load_execution_context
)


class TestExecutionMetrics:
    """Tests for ExecutionMetrics class."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = ExecutionMetrics()
        
        assert isinstance(metrics.start_time, datetime)
        assert metrics.end_time is None
        assert metrics.duration is None
        assert metrics.steps_completed == 0
        assert metrics.steps_failed == 0
        assert metrics.steps_skipped == 0
        assert metrics.steps_total == 0
        assert metrics.variables_created == 0
        assert metrics.variables_updated == 0
    
    def test_mark_completed(self):
        """Test marking execution as completed."""
        metrics = ExecutionMetrics()
        original_start = metrics.start_time
        
        metrics.mark_completed()
        
        assert metrics.end_time is not None
        assert isinstance(metrics.end_time, datetime)
        assert metrics.end_time >= original_start
        assert metrics.duration is not None
        assert isinstance(metrics.duration, timedelta)
    
    def test_completion_percentage(self):
        """Test completion percentage calculation."""
        metrics = ExecutionMetrics()
        
        # No steps
        assert metrics.completion_percentage() == 0.0
        
        # Some steps completed
        metrics.steps_total = 10
        metrics.steps_completed = 3
        assert metrics.completion_percentage() == 30.0
        
        # All steps completed
        metrics.steps_completed = 10
        assert metrics.completion_percentage() == 100.0
    
    def test_success_rate(self):
        """Test success rate calculation."""
        metrics = ExecutionMetrics()
        
        # No attempts
        assert metrics.success_rate() == 0.0
        
        # Some successes and failures
        metrics.steps_completed = 7
        metrics.steps_failed = 3
        assert metrics.success_rate() == 70.0
        
        # All successes
        metrics.steps_failed = 0
        assert metrics.success_rate() == 100.0


class TestCheckpoint:
    """Tests for Checkpoint class."""
    
    def test_checkpoint_creation(self):
        """Test checkpoint creation."""
        timestamp = datetime.now()
        checkpoint = Checkpoint(
            id="test_checkpoint",
            timestamp=timestamp,
            execution_id="test_execution",
            step_id="test_step",
            status=ExecutionStatus.RUNNING
        )
        
        assert checkpoint.id == "test_checkpoint"
        assert checkpoint.timestamp == timestamp
        assert checkpoint.execution_id == "test_execution"
        assert checkpoint.step_id == "test_step"
        assert checkpoint.status == ExecutionStatus.RUNNING
    
    def test_auto_id_generation(self):
        """Test automatic ID generation."""
        checkpoint = Checkpoint(
            id="",  # Empty ID should trigger auto-generation
            timestamp=datetime.now(),
            execution_id="test_execution"
        )
        
        assert checkpoint.id != ""
        assert len(checkpoint.id) == 8  # MD5 hash truncated to 8 chars


class TestFileStateManager:
    """Tests for FileStateManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test state files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def state_manager(self, temp_dir):
        """Create FileStateManager with temporary directory."""
        return FileStateManager(
            state_dir=temp_dir,
            format=PersistenceFormat.JSON,
            auto_cleanup_days=0  # Disable auto-cleanup for tests
        )
    
    def test_initialization(self, temp_dir):
        """Test state manager initialization."""
        manager = FileStateManager(
            state_dir=temp_dir,
            format=PersistenceFormat.COMPRESSED_JSON,
            auto_cleanup_days=7
        )
        
        assert manager.state_dir == temp_dir
        assert manager.format == PersistenceFormat.COMPRESSED_JSON
        assert manager.auto_cleanup_days == 7
        assert temp_dir.exists()
    
    def test_json_persistence(self, temp_dir):
        """Test JSON format persistence."""
        manager = FileStateManager(
            state_dir=temp_dir,
            format=PersistenceFormat.JSON
        )
        
        # Create test execution context
        context = ExecutionContext("test_exec", "test_pipeline")
        context.variable_manager.set_variable("test_var", "test_value")
        
        # Save state
        result = manager.save_state(context)
        assert result is True
        
        # Check file was created
        json_files = list(temp_dir.glob("*.json"))
        assert len(json_files) == 1
        
        # Load state
        loaded_context = manager.load_state("test_exec")
        assert loaded_context is not None
        assert loaded_context.execution_id == "test_exec"
        assert loaded_context.variable_manager.get_variable("test_var") == "test_value"
    
    def test_pickle_persistence(self, temp_dir):
        """Test pickle format persistence."""
        manager = FileStateManager(
            state_dir=temp_dir,
            format=PersistenceFormat.PICKLE
        )
        
        # Create test execution context
        context = ExecutionContext("test_exec", "test_pipeline")
        context.variable_manager.set_variable("complex_var", {"nested": {"data": [1, 2, 3]}})
        
        # Save state
        result = manager.save_state(context)
        assert result is True
        
        # Check file was created
        pickle_files = list(temp_dir.glob("*.pkl"))
        assert len(pickle_files) == 1
        
        # Load state
        loaded_context = manager.load_state("test_exec")
        assert loaded_context is not None
        assert loaded_context.variable_manager.get_variable("complex_var") == {"nested": {"data": [1, 2, 3]}}
    
    def test_compressed_formats(self, temp_dir):
        """Test compressed format persistence."""
        for format_type in [PersistenceFormat.COMPRESSED_JSON, PersistenceFormat.COMPRESSED_PICKLE]:
            manager = FileStateManager(
                state_dir=temp_dir,
                format=format_type
            )
            
            # Create test execution context
            context = ExecutionContext(f"test_exec_{format_type.value}", "test_pipeline")
            context.variable_manager.set_variable("test_var", "test_value" * 100)  # Larger data for compression
            
            # Save state
            result = manager.save_state(context)
            assert result is True
            
            # Load state
            loaded_context = manager.load_state(f"test_exec_{format_type.value}")
            assert loaded_context is not None
            assert loaded_context.variable_manager.get_variable("test_var") == "test_value" * 100
    
    def test_list_states(self, state_manager, temp_dir):
        """Test listing available states."""
        # Initially empty
        states = state_manager.list_states()
        assert states == []
        
        # Create multiple execution contexts
        for i in range(3):
            context = ExecutionContext(f"exec_{i}", "test_pipeline")
            state_manager.save_state(context)
        
        # List states
        states = state_manager.list_states()
        assert len(states) == 3
        assert "exec_0" in states
        assert "exec_1" in states
        assert "exec_2" in states
    
    def test_delete_state(self, state_manager):
        """Test state deletion."""
        # Create and save context
        context = ExecutionContext("to_delete", "test_pipeline")
        state_manager.save_state(context)
        
        # Verify it exists
        states = state_manager.list_states()
        assert "to_delete" in states
        
        # Delete state
        result = state_manager.delete_state("to_delete")
        assert result is True
        
        # Verify it's gone
        states = state_manager.list_states()
        assert "to_delete" not in states
        
        # Test deleting non-existent state
        result = state_manager.delete_state("nonexistent")
        assert result is False


class TestExecutionContext:
    """Tests for ExecutionContext class."""
    
    @pytest.fixture
    def execution_context(self):
        """Create a fresh ExecutionContext for each test."""
        return ExecutionContext("test_execution", "test_pipeline")
    
    def test_initialization(self, execution_context):
        """Test execution context initialization."""
        assert execution_context.execution_id == "test_execution"
        assert execution_context.pipeline_id == "test_pipeline"
        assert execution_context.status == ExecutionStatus.PENDING
        assert execution_context.parent_context is None
        assert isinstance(execution_context.metrics, ExecutionMetrics)
        assert execution_context.current_step_id is None
        assert len(execution_context.checkpoints) == 0
    
    def test_execution_lifecycle(self, execution_context):
        """Test complete execution lifecycle."""
        # Start execution
        execution_context.start_execution()
        assert execution_context.status == ExecutionStatus.RUNNING
        
        # Complete execution successfully
        execution_context.complete_execution(success=True)
        assert execution_context.status == ExecutionStatus.COMPLETED
        assert execution_context.metrics.end_time is not None
        assert execution_context.metrics.duration is not None
    
    def test_execution_pause_resume(self, execution_context):
        """Test execution pause and resume."""
        execution_context.start_execution()
        
        # Pause execution
        result = execution_context.pause_execution()
        assert result is True
        assert execution_context.status == ExecutionStatus.PAUSED
        
        # Resume execution
        result = execution_context.resume_execution()
        assert result is True
        assert execution_context.status == ExecutionStatus.RUNNING
        
        # Test pausing when not running
        execution_context.complete_execution()
        result = execution_context.pause_execution()
        assert result is False
    
    def test_execution_cancellation(self, execution_context):
        """Test execution cancellation."""
        execution_context.start_execution()
        
        execution_context.cancel_execution()
        assert execution_context.status == ExecutionStatus.CANCELLED
        assert execution_context.metrics.end_time is not None
    
    def test_step_management(self, execution_context):
        """Test step execution management."""
        execution_context.start_execution()
        
        # Start step
        execution_context.start_step("step1")
        assert execution_context.current_step_id == "step1"
        assert execution_context.metrics.steps_total >= 1
        
        # Complete step successfully
        execution_context.complete_step("step1", success=True)
        assert execution_context.metrics.steps_completed == 1
        assert execution_context.current_step_id is None
        
        # Start and fail step
        execution_context.start_step("step2")
        execution_context.complete_step("step2", success=False)
        assert execution_context.metrics.steps_failed == 1
        
        # Skip step
        execution_context.skip_step("step3")
        assert execution_context.metrics.steps_skipped == 1
    
    def test_checkpoint_management(self, execution_context):
        """Test checkpoint creation and restoration."""
        execution_context.start_execution()
        execution_context.variable_manager.set_variable("test_var", "checkpoint_value")
        
        # Create checkpoint
        checkpoint = execution_context.create_checkpoint("test checkpoint")
        assert checkpoint.id != ""
        assert checkpoint.execution_id == "test_execution"
        assert len(execution_context.checkpoints) == 1
        
        # Modify state
        execution_context.variable_manager.set_variable("test_var", "modified_value")
        execution_context.complete_step("step1", success=True)
        
        # Restore checkpoint
        result = execution_context.restore_checkpoint(checkpoint.id)
        assert result is True
        assert execution_context.variable_manager.get_variable("test_var") == "checkpoint_value"
        
        # Test restoring non-existent checkpoint
        result = execution_context.restore_checkpoint("nonexistent")
        assert result is False
    
    def test_nested_context_management(self, execution_context):
        """Test nested context creation and management."""
        # Create nested context
        nested_context = execution_context.create_nested_context("child")
        assert nested_context.execution_id == "test_execution.child"
        assert nested_context.parent_context == execution_context
        assert "child" in execution_context.nested_contexts
        
        # Test nested context independence
        nested_context.variable_manager.set_variable("child_var", "child_value")
        assert nested_context.variable_manager.get_variable("child_var") == "child_value"
        assert execution_context.variable_manager.get_variable("child_var") is None
        
        # Destroy nested context
        execution_context.destroy_nested_context("child")
        assert "child" not in execution_context.nested_contexts
    
    def test_state_persistence(self, execution_context):
        """Test state persistence with state manager."""
        # Create temporary state manager
        with tempfile.TemporaryDirectory() as temp_dir:
            state_manager = FileStateManager(temp_dir)
            execution_context.set_state_manager(state_manager)
            
            # Set up context state
            execution_context.start_execution()
            execution_context.variable_manager.set_variable("persistent_var", "persistent_value")
            execution_context.start_step("step1")
            execution_context.complete_step("step1", success=True)
            
            # Save state
            result = execution_context.save_state()
            assert result is True
            
            # Load state in new context
            loaded_context = ExecutionContext.load_state("test_execution", state_manager)
            assert loaded_context is not None
            assert loaded_context.execution_id == "test_execution"
            assert loaded_context.variable_manager.get_variable("persistent_var") == "persistent_value"
            assert loaded_context.metrics.steps_completed == 1
    
    def test_state_export_import(self, execution_context):
        """Test state export and import."""
        # Set up context state
        execution_context.start_execution()
        execution_context.variable_manager.set_variable("export_var", "export_value")
        execution_context.start_step("step1")
        execution_context.complete_step("step1", success=True)
        execution_context.create_checkpoint("export test")
        
        # Export state
        state_data = execution_context.export_state()
        assert "execution_id" in state_data
        assert "pipeline_id" in state_data
        assert "status" in state_data
        assert "metrics" in state_data
        assert "variables" in state_data
        assert "checkpoints" in state_data
        
        # Import state into new context
        imported_context = ExecutionContext._from_state_data(state_data)
        assert imported_context.execution_id == "test_execution"
        assert imported_context.pipeline_id == "test_pipeline"
        assert imported_context.status == ExecutionStatus.RUNNING
        assert imported_context.variable_manager.get_variable("export_var") == "export_value"
        assert imported_context.metrics.steps_completed == 1
        assert len(imported_context.checkpoints) == 1
    
    def test_event_handlers(self, execution_context):
        """Test event handler registration and notification."""
        step_events = []
        status_changes = []
        
        def step_handler(step_id, event_type):
            step_events.append((step_id, event_type))
        
        def status_handler(context):
            status_changes.append(context.status)
        
        execution_context.on_step_event(step_handler)
        execution_context.on_status_change(status_handler)
        
        # Trigger events
        execution_context.start_execution()
        execution_context.start_step("step1")
        execution_context.complete_step("step1", success=True)
        execution_context.complete_execution()
        
        # Verify events were captured
        assert len(step_events) >= 2  # At least started and completed
        assert ("step1", "started") in step_events
        assert ("step1", "completed") in step_events
        
        assert ExecutionStatus.RUNNING in status_changes
        assert ExecutionStatus.COMPLETED in status_changes
    
    def test_context_manager_interface(self):
        """Test ExecutionContext as context manager."""
        with ExecutionContext("ctx_test", "test_pipeline") as context:
            assert context.status == ExecutionStatus.RUNNING
            context.variable_manager.set_variable("ctx_var", "ctx_value")
        
        assert context.status == ExecutionStatus.COMPLETED
        assert context.variable_manager.get_variable("ctx_var") == "ctx_value"
    
    def test_context_manager_with_exception(self):
        """Test ExecutionContext as context manager with exception."""
        try:
            with ExecutionContext("ctx_fail_test", "test_pipeline") as context:
                assert context.status == ExecutionStatus.RUNNING
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert context.status == ExecutionStatus.FAILED
    
    def test_variable_scope_context(self, execution_context):
        """Test variable scope context manager."""
        execution_context.variable_manager.set_variable("global_var", "global_value")
        
        with execution_context.variable_scope() as var_ctx:
            var_ctx.set_variable("scoped_var", "scoped_value")
            assert var_ctx.get_variable("scoped_var") == "scoped_value"
            assert var_ctx.get_variable("global_var") == "global_value"
        
        # Scoped variable should be cleaned up
        assert execution_context.variable_manager.get_variable("scoped_var") is None
        assert execution_context.variable_manager.get_variable("global_var") == "global_value"


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_execution_context(self):
        """Test create_execution_context convenience function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            context = create_execution_context(
                "convenience_test",
                "test_pipeline",
                state_dir=temp_dir,
                auto_save=True
            )
            
            assert context.execution_id == "convenience_test"
            assert context.pipeline_id == "test_pipeline"
            assert context.state_manager is not None
            assert isinstance(context.state_manager, FileStateManager)
    
    def test_load_execution_context(self):
        """Test load_execution_context convenience function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save context
            original_context = create_execution_context(
                "load_test",
                "test_pipeline",
                state_dir=temp_dir
            )
            original_context.variable_manager.set_variable("load_var", "load_value")
            original_context.save_state()
            
            # Load context using convenience function
            loaded_context = load_execution_context("load_test", temp_dir)
            
            assert loaded_context is not None
            assert loaded_context.execution_id == "load_test"
            assert loaded_context.variable_manager.get_variable("load_var") == "load_value"


class TestIntegrationScenarios:
    """Integration tests for complex state management scenarios."""
    
    def test_complete_pipeline_execution(self):
        """Test complete pipeline execution with state management."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create execution context
            context = create_execution_context(
                "integration_test",
                "integration_pipeline",
                state_dir=temp_dir
            )
            
            with context:
                # Set pipeline inputs
                from src.orchestrator.execution.variables import VariableType
                context.variable_manager.set_variable(
                    "input_data", [1, 2, 3, 4, 5],
                    var_type=VariableType.INPUT
                )
                
                # Step 1: Data preprocessing
                context.start_step("preprocess")
                input_data = context.variable_manager.get_variable("input_data")
                processed_data = [x * 2 for x in input_data]
                context.variable_manager.set_variable(
                    "processed_data", processed_data,
                    source_step="preprocess"
                )
                context.complete_step("preprocess", success=True)
                
                # Create checkpoint after preprocessing
                checkpoint1 = context.create_checkpoint("after_preprocessing")
                
                # Step 2: Data analysis
                context.start_step("analyze")
                processed_data = context.variable_manager.get_variable("processed_data")
                analysis_result = {
                    "sum": sum(processed_data),
                    "mean": sum(processed_data) / len(processed_data),
                    "count": len(processed_data)
                }
                context.variable_manager.set_variable(
                    "analysis_result", analysis_result,
                    source_step="analyze"
                )
                context.complete_step("analyze", success=True)
                
                # Step 3: Generate report
                context.start_step("report")
                analysis = context.variable_manager.get_variable("analysis_result")
                report = f"Analysis complete: {analysis['count']} items, sum={analysis['sum']}, mean={analysis['mean']}"
                context.variable_manager.set_variable(
                    "final_report", report,
                    source_step="report"
                )
                context.complete_step("report", success=True)
                
                # Save final state
                context.save_state()
            
            # Verify execution completed successfully
            assert context.status == ExecutionStatus.COMPLETED
            assert context.metrics.steps_completed == 3
            assert context.metrics.steps_failed == 0
            
            # Verify final results
            assert context.variable_manager.get_variable("final_report") == "Analysis complete: 5 items, sum=30, mean=6.0"
            
            # Test state recovery
            loaded_context = load_execution_context("integration_test", temp_dir)
            assert loaded_context.variable_manager.get_variable("final_report") == "Analysis complete: 5 items, sum=30, mean=6.0"
            assert loaded_context.metrics.steps_completed == 3
    
    def test_checkpoint_recovery_scenario(self):
        """Test checkpoint-based recovery scenario."""
        context = ExecutionContext("recovery_test", "recovery_pipeline")
        
        # Execute first part
        context.start_execution()
        context.variable_manager.set_variable("stage1_data", "completed")
        context.start_step("step1")
        context.complete_step("step1", success=True)
        
        # Create checkpoint
        checkpoint = context.create_checkpoint("after_step1")
        
        # Continue execution and simulate failure
        context.variable_manager.set_variable("stage2_data", "in_progress")
        context.start_step("step2")
        # Simulate failure without completing step2
        
        # Restore from checkpoint
        result = context.restore_checkpoint(checkpoint.id)
        assert result is True
        
        # Verify state was restored
        assert context.variable_manager.get_variable("stage1_data") == "completed"
        assert context.variable_manager.get_variable("stage2_data") is None
        assert context.metrics.steps_completed == 1  # From checkpoint
        assert context.current_step_id is None  # Checkpoint was after step completion
    
    def test_nested_execution_contexts(self):
        """Test nested execution contexts for sub-pipelines."""
        parent_context = ExecutionContext("parent", "main_pipeline")
        parent_context.start_execution()
        
        # Set parent variables
        parent_context.variable_manager.set_variable("shared_config", {"threads": 4})
        
        # Create nested context for sub-pipeline
        child_context = parent_context.create_nested_context("subprocess")
        child_context.start_execution()
        
        # Child can access parent variables through variable manager
        # (This would require additional implementation for variable inheritance)
        child_context.variable_manager.set_variable("child_result", "child_success")
        
        # Complete child execution
        child_context.complete_execution(success=True)
        
        # Parent can access child results if needed
        # (Implementation depends on variable sharing strategy)
        
        # Complete parent execution
        parent_context.complete_execution(success=True)
        
        # Verify both contexts completed
        assert parent_context.status == ExecutionStatus.COMPLETED
        assert child_context.status == ExecutionStatus.COMPLETED
        assert "subprocess" in parent_context.nested_contexts


if __name__ == "__main__":
    pytest.main([__file__])