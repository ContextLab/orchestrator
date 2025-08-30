"""
Integration tests for Variable & State Management with existing runtime systems.

Tests the bridge integration between the new variable management system
and the existing PipelineExecutionState from the runtime module.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.orchestrator.execution.integration import (
    ExecutionStateBridge,
    VariableManagerAdapter
)
from src.orchestrator.execution.variables import (
    VariableManager,
    VariableScope,
    VariableType
)
from src.orchestrator.execution.state import ExecutionContext, ExecutionStatus
from src.orchestrator.runtime.execution_state import PipelineExecutionState


class TestExecutionStateBridge:
    """Tests for ExecutionStateBridge integration."""
    
    @pytest.fixture
    def execution_context(self):
        """Create execution context for testing."""
        return ExecutionContext("test_execution", "test_pipeline")
    
    @pytest.fixture
    def pipeline_state(self):
        """Create pipeline execution state for testing."""
        return PipelineExecutionState("test_pipeline")
    
    @pytest.fixture
    def bridge(self, execution_context, pipeline_state):
        """Create bridge between execution context and pipeline state."""
        return ExecutionStateBridge(execution_context, pipeline_state)
    
    def test_bridge_initialization(self, execution_context, pipeline_state):
        """Test bridge initialization."""
        bridge = ExecutionStateBridge(execution_context)
        assert bridge.execution_context == execution_context
        assert bridge.pipeline_state is None
        
        bridge_with_state = ExecutionStateBridge(execution_context, pipeline_state)
        assert bridge_with_state.pipeline_state == pipeline_state
    
    def test_set_pipeline_state(self, execution_context, pipeline_state):
        """Test setting pipeline state after initialization."""
        bridge = ExecutionStateBridge(execution_context)
        bridge.set_pipeline_state(pipeline_state)
        assert bridge.pipeline_state == pipeline_state
    
    def test_sync_to_pipeline_state(self, bridge):
        """Test syncing variables from ExecutionContext to PipelineExecutionState."""
        # Set variables in execution context
        bridge.execution_context.variable_manager.set_variable(
            "config_var", "config_value",
            var_type=VariableType.CONFIGURATION
        )
        bridge.execution_context.variable_manager.set_variable(
            "output_var", "output_value",
            var_type=VariableType.OUTPUT,
            source_step="test_step"
        )
        
        # Sync to pipeline state
        bridge.sync_to_pipeline_state()
        
        # Verify variables were synced
        context = bridge.pipeline_state.get_available_context()
        assert "config_var" in context
        assert context["config_var"] == "config_value"
        assert "test_step" in bridge.pipeline_state.global_context["results"]
        assert bridge.pipeline_state.global_context["results"]["test_step"] == "output_value"
    
    def test_sync_from_pipeline_state(self, bridge):
        """Test syncing variables from PipelineExecutionState to ExecutionContext."""
        # Set variables in pipeline state
        bridge.pipeline_state.register_variable("pipeline_var", "pipeline_value")
        bridge.pipeline_state.register_result("task1", {"result": "task_result"})
        
        # Sync from pipeline state
        bridge.sync_from_pipeline_state()
        
        # Verify variables were synced
        assert bridge.execution_context.variable_manager.get_variable("pipeline_var") == "pipeline_value"
        assert bridge.execution_context.variable_manager.get_variable("task1") == {"result": "task_result"}
        
        # Check variable metadata
        pipeline_var_meta = bridge.execution_context.variable_manager.get_variable_metadata("pipeline_var")
        assert pipeline_var_meta.var_type == VariableType.CONFIGURATION
        
        task_meta = bridge.execution_context.variable_manager.get_variable_metadata("task1")
        assert task_meta.var_type == VariableType.OUTPUT
    
    def test_register_variable(self, bridge):
        """Test registering variables in both systems."""
        bridge.register_variable(
            "bridge_var", "bridge_value",
            var_type=VariableType.INTERMEDIATE,
            source_step="bridge_step"
        )
        
        # Check new system
        assert bridge.execution_context.variable_manager.get_variable("bridge_var") == "bridge_value"
        meta = bridge.execution_context.variable_manager.get_variable_metadata("bridge_var")
        assert meta.var_type == VariableType.INTERMEDIATE
        assert meta.source_step == "bridge_step"
        
        # Check legacy system
        context = bridge.pipeline_state.get_available_context()
        assert "bridge_var" in context
        assert context["bridge_var"] == "bridge_value"
    
    def test_register_step_result(self, bridge):
        """Test registering step results in both systems."""
        bridge.register_step_result("test_step", {"result": "success", "value": 42})
        
        # Check new system
        result = bridge.execution_context.variable_manager.get_variable("test_step")
        assert result == {"result": "success", "value": 42}
        
        meta = bridge.execution_context.variable_manager.get_variable_metadata("test_step")
        assert meta.var_type == VariableType.OUTPUT
        assert meta.source_step == "test_step"
        
        # Check execution context step tracking
        assert bridge.execution_context.metrics.steps_completed == 1
        
        # Check legacy system
        assert "test_step" in bridge.pipeline_state.global_context["results"]
        assert bridge.pipeline_state.global_context["results"]["test_step"] == {"result": "success", "value": 42}
        assert "test_step" in bridge.pipeline_state.executed_tasks
    
    def test_step_lifecycle(self, bridge):
        """Test complete step lifecycle through bridge."""
        # Start step
        bridge.start_step("lifecycle_step")
        
        # Check new system
        assert bridge.execution_context.current_step_id == "lifecycle_step"
        
        # Check legacy system
        assert "lifecycle_step" in bridge.pipeline_state.pending_tasks
        
        # Complete step successfully
        bridge.register_step_result("lifecycle_step", "success")
        
        # Check final state
        assert bridge.execution_context.current_step_id is None
        assert bridge.execution_context.metrics.steps_completed == 1
        assert "lifecycle_step" not in bridge.pipeline_state.pending_tasks
        assert "lifecycle_step" in bridge.pipeline_state.executed_tasks
    
    def test_step_failure(self, bridge):
        """Test step failure handling."""
        bridge.start_step("failing_step")
        bridge.fail_step("failing_step", "Test error message")
        
        # Check new system
        assert bridge.execution_context.metrics.steps_failed == 1
        
        # Check legacy system
        assert "failing_step" in bridge.pipeline_state.failed_tasks
        assert bridge.pipeline_state.failed_tasks["failing_step"] == "Test error message"
        assert "failing_step" not in bridge.pipeline_state.pending_tasks
    
    def test_get_variable_fallback(self, bridge):
        """Test variable retrieval with fallback."""
        # Set variable only in new system
        bridge.execution_context.variable_manager.set_variable("new_var", "new_value")
        
        # Set variable only in legacy system
        bridge.pipeline_state.register_variable("legacy_var", "legacy_value")
        
        # Test retrieval
        assert bridge.get_variable("new_var") == "new_value"
        assert bridge.get_variable("legacy_var") == "legacy_value"
        assert bridge.get_variable("nonexistent", "default") == "default"
    
    def test_has_variable_check(self, bridge):
        """Test variable existence check across systems."""
        # Set variables in different systems
        bridge.execution_context.variable_manager.set_variable("new_var", "new_value")
        bridge.pipeline_state.register_variable("legacy_var", "legacy_value")
        
        # Test existence checks
        assert bridge.has_variable("new_var") is True
        assert bridge.has_variable("legacy_var") is True
        assert bridge.has_variable("nonexistent") is False
    
    def test_export_combined_state(self, bridge):
        """Test exporting combined state from both systems."""
        # Set up state in both systems
        bridge.execution_context.variable_manager.set_variable("new_var", "new_value")
        bridge.pipeline_state.register_variable("legacy_var", "legacy_value")
        bridge.execution_context.start_execution()
        
        # Export combined state
        combined_state = bridge.export_combined_state()
        
        # Verify structure
        assert "execution_context" in combined_state
        assert "pipeline_state" in combined_state
        assert "bridge_metadata" in combined_state
        
        # Verify metadata
        metadata = combined_state["bridge_metadata"]
        assert metadata["pipeline_id"] == "test_pipeline"
        assert metadata["execution_id"] == "test_execution"
        
        # Verify context data
        assert combined_state["execution_context"]["execution_id"] == "test_execution"
        assert combined_state["pipeline_state"]["pipeline_id"] == "test_pipeline"
    
    def test_bidirectional_synchronization(self, execution_context):
        """Test automatic bidirectional synchronization."""
        pipeline_state = PipelineExecutionState("test_pipeline")
        pipeline_state.register_variable("initial_var", "initial_value")
        
        # Create bridge with existing pipeline state
        bridge = ExecutionStateBridge(execution_context, pipeline_state)
        
        # Initial sync should have occurred
        assert bridge.execution_context.variable_manager.get_variable("initial_var") == "initial_value"
        
        # Test ongoing synchronization via event handlers
        bridge.execution_context.variable_manager.set_variable("new_var", "new_value")
        
        # Should be synchronized to pipeline state
        context = bridge.pipeline_state.get_available_context()
        assert "new_var" in context


class TestVariableManagerAdapter:
    """Tests for VariableManagerAdapter."""
    
    @pytest.fixture
    def variable_manager(self):
        """Create variable manager for testing."""
        return VariableManager("test_pipeline")
    
    @pytest.fixture
    def adapter(self, variable_manager):
        """Create adapter for testing."""
        return VariableManagerAdapter(variable_manager)
    
    def test_dict_like_interface(self, adapter):
        """Test dict-like interface methods."""
        # Test set/get
        adapter.set("key1", "value1")
        assert adapter.get("key1") == "value1"
        assert adapter.get("nonexistent", "default") == "default"
        
        # Test bracket notation
        adapter["key2"] = "value2"
        assert adapter["key2"] == "value2"
        
        # Test KeyError for missing key
        with pytest.raises(KeyError):
            _ = adapter["nonexistent"]
        
        # Test contains
        assert "key1" in adapter
        assert "key2" in adapter
        assert "nonexistent" not in adapter
    
    def test_iteration_methods(self, adapter):
        """Test iteration methods."""
        adapter.set("key1", "value1")
        adapter.set("key2", "value2")
        adapter.set("key3", "value3")
        
        # Test keys()
        keys = list(adapter.keys())
        assert len(keys) == 3
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" in keys
        
        # Test values()
        values = list(adapter.values())
        assert len(values) == 3
        assert "value1" in values
        assert "value2" in values
        assert "value3" in values
        
        # Test items()
        items = list(adapter.items())
        assert len(items) == 3
        assert ("key1", "value1") in items
        assert ("key2", "value2") in items
        assert ("key3", "value3") in items
    
    def test_update_method(self, adapter):
        """Test update method."""
        update_data = {
            "update1": "value1",
            "update2": "value2",
            "update3": "value3"
        }
        
        adapter.update(update_data)
        
        # Verify all values were set
        for key, value in update_data.items():
            assert adapter.get(key) == value
    
    def test_adapter_variable_manager_integration(self, variable_manager, adapter):
        """Test adapter integration with underlying variable manager."""
        # Set through adapter
        adapter.set("adapter_var", "adapter_value")
        
        # Verify in underlying manager
        assert variable_manager.get_variable("adapter_var") == "adapter_value"
        
        # Set through manager
        variable_manager.set_variable("manager_var", "manager_value")
        
        # Verify through adapter
        assert adapter.get("manager_var") == "manager_value"


class TestIntegrationScenarios:
    """Integration tests for complex bridge scenarios."""
    
    def test_pipeline_execution_with_bridge(self):
        """Test complete pipeline execution using bridge integration."""
        # Set up systems
        execution_context = ExecutionContext("integration_exec", "integration_pipeline")
        pipeline_state = PipelineExecutionState("integration_pipeline")
        bridge = ExecutionStateBridge(execution_context, pipeline_state)
        
        # Start execution
        execution_context.start_execution()
        
        # Set initial configuration through bridge
        bridge.register_variable(
            "max_retries", 3,
            var_type=VariableType.CONFIGURATION
        )
        bridge.register_variable(
            "timeout_seconds", 30,
            var_type=VariableType.CONFIGURATION
        )
        
        # Execute steps through bridge
        bridge.start_step("data_preparation")
        bridge.register_step_result("data_preparation", {"processed_items": 100})
        
        bridge.start_step("data_analysis")
        # Get previous step result
        prep_result = bridge.get_variable("data_preparation")
        analysis_result = {"analyzed_items": prep_result["processed_items"], "score": 0.85}
        bridge.register_step_result("data_analysis", analysis_result)
        
        bridge.start_step("report_generation")
        analysis_data = bridge.get_variable("data_analysis")
        report = f"Analysis complete: {analysis_data['analyzed_items']} items with score {analysis_data['score']}"
        bridge.register_step_result("report_generation", report)
        
        # Complete execution
        execution_context.complete_execution(success=True)
        
        # Verify both systems have consistent state
        assert execution_context.status == ExecutionStatus.COMPLETED
        assert execution_context.metrics.steps_completed == 3
        assert execution_context.metrics.steps_failed == 0
        
        # Check pipeline state
        assert len(pipeline_state.executed_tasks) == 3
        assert "data_preparation" in pipeline_state.executed_tasks
        assert "data_analysis" in pipeline_state.executed_tasks
        assert "report_generation" in pipeline_state.executed_tasks
        
        # Check final result consistency
        exec_report = execution_context.variable_manager.get_variable("report_generation")
        pipeline_report = pipeline_state.global_context["results"]["report_generation"]
        assert exec_report == pipeline_report
        assert "100 items with score 0.85" in exec_report
    
    def test_error_recovery_with_bridge(self):
        """Test error recovery scenario using bridge."""
        execution_context = ExecutionContext("recovery_exec", "recovery_pipeline")
        pipeline_state = PipelineExecutionState("recovery_pipeline")
        bridge = ExecutionStateBridge(execution_context, pipeline_state)
        
        execution_context.start_execution()
        
        # Execute successful steps
        bridge.start_step("step1")
        bridge.register_step_result("step1", "success1")
        
        bridge.start_step("step2")
        bridge.register_step_result("step2", "success2")
        
        # Create checkpoint
        checkpoint = execution_context.create_checkpoint("before_failure")
        
        # Simulate step failure
        bridge.start_step("failing_step")
        bridge.fail_step("failing_step", "Simulated failure")
        
        # Verify failure state
        assert execution_context.metrics.steps_failed == 1
        assert "failing_step" in pipeline_state.failed_tasks
        
        # Restore from checkpoint
        success = execution_context.restore_checkpoint(checkpoint.id)
        assert success is True
        
        # Re-sync state after restore
        bridge.sync_to_pipeline_state()
        
        # Verify restored state
        assert bridge.get_variable("step1") == "success1"
        assert bridge.get_variable("step2") == "success2"
        assert execution_context.metrics.steps_completed == 2  # From checkpoint
    
    def test_nested_context_with_bridge(self):
        """Test nested execution contexts with bridge integration."""
        parent_context = ExecutionContext("parent_exec", "main_pipeline")
        parent_pipeline = PipelineExecutionState("main_pipeline")
        parent_bridge = ExecutionStateBridge(parent_context, parent_pipeline)
        
        parent_context.start_execution()
        parent_bridge.register_variable("shared_config", {"parallel": True})
        
        # Create nested context for sub-pipeline
        child_context = parent_context.create_nested_context("sub_pipeline")
        child_pipeline = PipelineExecutionState("sub_pipeline")
        child_bridge = ExecutionStateBridge(child_context, child_pipeline)
        
        child_context.start_execution()
        
        # Child pipeline execution
        child_bridge.start_step("child_step1")
        child_bridge.register_step_result("child_step1", "child_result")
        
        child_context.complete_execution(success=True)
        
        # Parent continues execution
        parent_bridge.start_step("parent_final_step")
        parent_bridge.register_step_result("parent_final_step", "parent_result")
        
        parent_context.complete_execution(success=True)
        
        # Verify both contexts completed successfully
        assert parent_context.status == ExecutionStatus.COMPLETED
        assert child_context.status == ExecutionStatus.COMPLETED
        
        # Verify nested context relationship
        assert "sub_pipeline" in parent_context.nested_contexts
        assert child_context.parent_context == parent_context


if __name__ == "__main__":
    pytest.main([__file__])