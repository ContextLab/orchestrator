"""
Tests for comprehensive execution integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.orchestrator.execution.integration import (
    ComprehensiveExecutionManager,
    create_comprehensive_execution_manager
)
from src.orchestrator.execution.state import ExecutionContext, ExecutionStatus
from src.orchestrator.execution.variables import VariableManager, VariableScope, VariableType
from src.orchestrator.execution.progress import ProgressTracker, ProgressEventType
from src.orchestrator.execution.recovery import RecoveryManager, RecoveryStrategy


class TestComprehensiveExecutionManager:
    """Test comprehensive execution manager integration."""
    
    @pytest.fixture
    def execution_manager(self):
        """Create test execution manager."""
        return ComprehensiveExecutionManager("test_exec", "test_pipeline")
    
    def test_execution_manager_creation(self):
        """Test creating execution manager."""
        manager = ComprehensiveExecutionManager("test_exec", "test_pipeline")
        
        assert manager.execution_id == "test_exec"
        assert manager.pipeline_id == "test_pipeline"
        assert isinstance(manager.execution_context, ExecutionContext)
        assert isinstance(manager.variable_manager, VariableManager)
        assert isinstance(manager.progress_tracker, ProgressTracker)
        assert isinstance(manager.recovery_manager, RecoveryManager)
    
    def test_execution_manager_with_existing_components(self):
        """Test creating execution manager with existing components."""
        execution_context = ExecutionContext("test_exec", "test_pipeline")
        progress_tracker = ProgressTracker(execution_context)
        recovery_manager = RecoveryManager(execution_context, progress_tracker)
        
        manager = ComprehensiveExecutionManager(
            execution_id="test_exec",
            pipeline_id="test_pipeline",
            execution_context=execution_context,
            progress_tracker=progress_tracker,
            recovery_manager=recovery_manager
        )
        
        assert manager.execution_context == execution_context
        assert manager.progress_tracker == progress_tracker
        assert manager.recovery_manager == recovery_manager
    
    def test_execution_lifecycle(self, execution_manager):
        """Test complete execution lifecycle."""
        # Start execution
        execution_manager.start_execution(total_steps=3)
        
        assert execution_manager.execution_context.status == ExecutionStatus.RUNNING
        assert execution_manager.execution_context.total_steps == 3
        
        # Check progress tracking was started
        execution_progress = execution_manager.progress_tracker.get_execution_progress("test_exec")
        assert execution_progress is not None
        assert execution_progress.total_steps == 3
        
        # Complete execution
        execution_manager.complete_execution(success=True)
        
        assert execution_manager.execution_context.status == ExecutionStatus.COMPLETED
    
    def test_step_lifecycle(self, execution_manager):
        """Test step execution lifecycle."""
        execution_manager.start_execution(total_steps=1)
        
        # Start step
        execution_manager.start_step("step1", "Test Step")
        
        # Check step tracking
        step_progress = execution_manager.progress_tracker.get_step_progress("test_exec", "step1")
        assert step_progress is not None
        assert step_progress.step_name == "Test Step"
        
        # Complete step
        execution_manager.complete_step("step1", success=True)
        
        # Check completion
        step_progress = execution_manager.progress_tracker.get_step_progress("test_exec", "step1")
        assert step_progress.progress_percentage == 100.0
    
    def test_step_error_handling(self, execution_manager):
        """Test step error handling."""
        execution_manager.start_execution(total_steps=1)
        execution_manager.start_step("step1", "Test Step")
        
        # Handle error
        error = ValueError("Test error")
        recovery_plan = execution_manager.handle_step_error("step1", "Test Step", error)
        
        assert recovery_plan is not None
        assert recovery_plan.strategy in [
            RecoveryStrategy.RETRY,
            RecoveryStrategy.RETRY_WITH_BACKOFF,
            RecoveryStrategy.FAIL_FAST
        ]
        
        # Check error was recorded
        error_history = execution_manager.recovery_manager.get_error_history()
        assert len(error_history) == 1
        assert error_history[0].step_id == "step1"
    
    def test_step_progress_update(self, execution_manager):
        """Test step progress updates."""
        execution_manager.start_execution(total_steps=1)
        execution_manager.start_step("step1", "Test Step")
        
        # Update progress
        execution_manager.update_step_progress("step1", 50.0, "Half done")
        
        # Check progress tracker
        step_progress = execution_manager.progress_tracker.get_step_progress("test_exec", "step1")
        assert step_progress.progress_percentage == 50.0
        
        # Check variable was set
        progress_var = execution_manager.variable_manager.get_variable("progress.step1.percentage")
        assert progress_var == 50.0
    
    def test_checkpoint_integration(self, execution_manager):
        """Test checkpoint integration."""
        execution_manager.start_execution(total_steps=1)
        
        # Create checkpoint
        checkpoint = execution_manager.create_checkpoint("test_checkpoint")
        
        assert checkpoint is not None
        assert len(execution_manager.execution_context.checkpoints) >= 1
        
        # Restore checkpoint
        success = execution_manager.restore_checkpoint(checkpoint.id)
        assert success is True
    
    def test_execution_status(self, execution_manager):
        """Test comprehensive execution status."""
        execution_manager.start_execution(total_steps=2)
        execution_manager.start_step("step1", "Test Step 1")
        execution_manager.complete_step("step1", success=True)
        
        status = execution_manager.get_execution_status()
        
        assert status["execution_id"] == "test_exec"
        assert status["pipeline_id"] == "test_pipeline"
        assert status["status"] == "running"
        assert status["progress"]["completed_steps"] == 1
        assert status["progress"]["total_steps"] == 2
        assert "recovery" in status
        assert "metrics" in status
    
    def test_cleanup(self, execution_manager):
        """Test cleanup functionality."""
        execution_manager.start_execution(total_steps=1)
        execution_manager.start_step("step1", "Test Step")
        
        # Generate some data
        execution_manager.handle_step_error("step1", "Test Step", ValueError("Error"))
        
        # Cleanup
        execution_manager.cleanup()
        
        # Check components are cleaned up
        execution_progress = execution_manager.progress_tracker.get_execution_progress("test_exec")
        assert execution_progress is None
        
        error_history = execution_manager.recovery_manager.get_error_history()
        assert len(error_history) == 0


@pytest.mark.asyncio
class TestComprehensiveExecutionManagerAsync:
    """Test asynchronous operations with comprehensive execution manager."""
    
    @pytest.fixture
    def execution_manager(self):
        """Create test execution manager."""
        return ComprehensiveExecutionManager("test_exec", "test_pipeline")
    
    async def test_execute_step_with_recovery_success(self, execution_manager):
        """Test successful step execution with recovery."""
        execution_manager.start_execution(total_steps=1)
        
        async def step_executor():
            return "success"
        
        success = await execution_manager.execute_step_with_recovery(
            "step1", "Test Step", step_executor
        )
        
        assert success is True
        
        # Check step was completed successfully
        step_progress = execution_manager.progress_tracker.get_step_progress("test_exec", "step1")
        assert step_progress.progress_percentage == 100.0
    
    async def test_execute_step_with_recovery_failure(self, execution_manager):
        """Test step execution failure with recovery."""
        execution_manager.start_execution(total_steps=1)
        
        async def failing_executor():
            raise ValueError("Persistent error")
        
        success = await execution_manager.execute_step_with_recovery(
            "step1", "Test Step", failing_executor
        )
        
        # Should fail after exhausting recovery attempts
        assert success is False
        
        # Check error was handled
        error_history = execution_manager.recovery_manager.get_error_history()
        assert len(error_history) >= 1
    
    async def test_execute_step_with_retry_recovery(self, execution_manager):
        """Test step execution with successful retry recovery."""
        execution_manager.start_execution(total_steps=1)
        
        # Set up a custom recovery plan for network errors
        from src.orchestrator.execution.recovery import ErrorCategory, RecoveryPlan, RetryConfig
        
        retry_plan = RecoveryPlan(
            strategy=RecoveryStrategy.RETRY,
            retry_config=RetryConfig(max_attempts=3, initial_delay=0.01)
        )
        execution_manager.recovery_manager.set_recovery_plan("step1", retry_plan)
        
        attempt_count = 0
        
        async def executor_with_retry():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Network error")
            return "success"
        
        success = await execution_manager.execute_step_with_recovery(
            "step1", "Test Step", executor_with_retry
        )
        
        assert success is True
        assert attempt_count == 3
    
    async def test_concurrent_step_execution(self, execution_manager):
        """Test concurrent step execution with recovery."""
        execution_manager.start_execution(total_steps=3)
        
        async def step_executor(step_id: str, delay: float, should_fail: bool = False):
            await asyncio.sleep(delay)
            if should_fail:
                raise ValueError(f"Error in {step_id}")
            return f"Result from {step_id}"
        
        # Execute steps concurrently
        results = await asyncio.gather(
            execution_manager.execute_step_with_recovery(
                "step1", "Step 1", lambda: step_executor("step1", 0.01)
            ),
            execution_manager.execute_step_with_recovery(
                "step2", "Step 2", lambda: step_executor("step2", 0.02, should_fail=True)
            ),
            execution_manager.execute_step_with_recovery(
                "step3", "Step 3", lambda: step_executor("step3", 0.01)
            ),
            return_exceptions=True
        )
        
        # Step 1 and 3 should succeed, step 2 should fail
        assert results[0] is True
        assert results[1] is False  # Failed step
        assert results[2] is True
    
    async def test_progress_event_integration(self, execution_manager):
        """Test progress event integration."""
        events = []
        
        def event_handler(event):
            events.append(event)
        
        execution_manager.progress_tracker.add_event_handler(event_handler)
        
        execution_manager.start_execution(total_steps=1)
        
        async def step_executor():
            return "success"
        
        await execution_manager.execute_step_with_recovery(
            "step1", "Test Step", step_executor
        )
        
        # Should have received various events
        event_types = [event.event_type for event in events]
        assert ProgressEventType.EXECUTION_STARTED in event_types
        assert ProgressEventType.STEP_STARTED in event_types
        assert ProgressEventType.STEP_COMPLETED in event_types


class TestExecutionManagerFactory:
    """Test execution manager factory functions."""
    
    def test_create_comprehensive_execution_manager(self):
        """Test creating execution manager with factory."""
        manager = create_comprehensive_execution_manager("test_exec", "test_pipeline")
        
        assert isinstance(manager, ComprehensiveExecutionManager)
        assert manager.execution_id == "test_exec"
        assert manager.pipeline_id == "test_pipeline"
    
    def test_factory_creates_integrated_components(self):
        """Test that factory creates properly integrated components."""
        manager = create_comprehensive_execution_manager("test_exec", "test_pipeline")
        
        # All components should be properly connected
        assert manager.progress_tracker.execution_context == manager.execution_context
        assert manager.progress_tracker.variable_manager == manager.variable_manager
        assert manager.recovery_manager.execution_context == manager.execution_context
        assert manager.recovery_manager.progress_tracker == manager.progress_tracker


class TestIntegrationWithExistingSystems:
    """Test integration with existing runtime systems."""
    
    def test_variable_integration(self):
        """Test variable manager integration."""
        manager = ComprehensiveExecutionManager("test_exec", "test_pipeline")
        
        # Set variables through manager
        manager.variable_manager.set_variable(
            "test_var", 
            "test_value",
            scope=VariableScope.GLOBAL,
            var_type=VariableType.CONFIGURATION
        )
        
        # Check variable is accessible
        value = manager.variable_manager.get_variable("test_var")
        assert value == "test_value"
    
    def test_checkpoint_restore_integration(self):
        """Test checkpoint restore with all systems."""
        manager = ComprehensiveExecutionManager("test_exec", "test_pipeline")
        manager.start_execution(total_steps=2)
        
        # Execute first step
        manager.start_step("step1", "Step 1")
        manager.variable_manager.set_variable("step1_result", "completed")
        manager.complete_step("step1", success=True)
        
        # Create checkpoint after first step
        checkpoint = manager.create_checkpoint("after_step1")
        
        # Execute and fail second step
        manager.start_step("step2", "Step 2")
        manager.variable_manager.set_variable("step2_result", "in_progress")
        
        # Restore to checkpoint
        success = manager.restore_checkpoint(checkpoint.id)
        assert success is True
        
        # Check that variables were restored
        step1_result = manager.variable_manager.get_variable("step1_result")
        step2_result = manager.variable_manager.get_variable("step2_result")
        
        assert step1_result == "completed"
        assert step2_result is None  # Should not exist after restore
    
    def test_error_recovery_with_checkpoints(self):
        """Test error recovery using checkpoints."""
        manager = ComprehensiveExecutionManager("test_exec", "test_pipeline")
        manager.start_execution(total_steps=1)
        
        # Create initial checkpoint
        initial_checkpoint = manager.create_checkpoint("initial_state")
        
        # Set up rollback recovery plan
        from src.orchestrator.execution.recovery import RecoveryPlan
        rollback_plan = RecoveryPlan(
            strategy=RecoveryStrategy.ROLLBACK,
            target_checkpoint=initial_checkpoint.id
        )
        manager.recovery_manager.set_recovery_plan("step1", rollback_plan)
        
        # Handle error
        error = Exception("Critical error")
        recovery_plan = manager.handle_step_error("step1", "Test Step", error)
        
        assert recovery_plan.strategy == RecoveryStrategy.ROLLBACK
        assert recovery_plan.target_checkpoint == initial_checkpoint.id


if __name__ == "__main__":
    pytest.main([__file__])