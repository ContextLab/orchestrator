"""
Tests for progress tracking system.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.orchestrator.execution.progress import (
    ProgressTracker,
    ProgressEvent,
    ProgressEventType,
    StepProgress,
    ExecutionProgress,
    StepStatus,
    create_progress_tracker
)
from src.orchestrator.execution.state import ExecutionContext
from src.orchestrator.execution.variables import VariableManager


class TestStepProgress:
    """Test StepProgress functionality."""
    
    def test_step_progress_creation(self):
        """Test creating step progress."""
        step = StepProgress("step1", "Test Step")
        assert step.step_id == "step1"
        assert step.step_name == "Test Step"
        assert step.status == StepStatus.PENDING
        assert step.progress_percentage == 0.0
        assert not step.is_completed
        assert not step.is_successful
    
    def test_step_progress_lifecycle(self):
        """Test step progress lifecycle."""
        step = StepProgress("step1", "Test Step")
        
        # Start step
        step.start()
        assert step.status == StepStatus.RUNNING
        assert step.start_time is not None
        assert not step.is_completed
        
        # Complete step
        step.complete(75.0)
        assert step.status == StepStatus.COMPLETED
        assert step.progress_percentage == 75.0
        assert step.end_time is not None
        assert step.duration is not None
        assert step.is_completed
        assert step.is_successful
    
    def test_step_progress_failure(self):
        """Test step progress failure."""
        step = StepProgress("step1", "Test Step")
        step.start()
        
        step.fail("Something went wrong")
        assert step.status == StepStatus.FAILED
        assert step.error_message == "Something went wrong"
        assert step.end_time is not None
        assert step.is_completed
        assert not step.is_successful
    
    def test_step_progress_skip(self):
        """Test step progress skipping."""
        step = StepProgress("step1", "Test Step")
        
        step.skip("Condition not met")
        assert step.status == StepStatus.SKIPPED
        assert step.metadata["skip_reason"] == "Condition not met"
        assert step.is_completed
        assert not step.is_successful


class TestExecutionProgress:
    """Test ExecutionProgress functionality."""
    
    def test_execution_progress_creation(self):
        """Test creating execution progress."""
        progress = ExecutionProgress("exec1", 10)
        assert progress.execution_id == "exec1"
        assert progress.total_steps == 10
        assert progress.completed_steps == 0
        assert progress.progress_percentage == 0.0
        assert not progress.is_completed
    
    def test_execution_progress_calculation(self):
        """Test progress percentage calculation."""
        progress = ExecutionProgress("exec1", 10)
        
        # Complete some steps
        progress.completed_steps = 3
        progress.failed_steps = 1
        progress.skipped_steps = 1
        
        # Should be (3 + 1 + 1) / 10 * 100 = 50%
        assert progress.progress_percentage == 50.0
        assert not progress.is_completed
        
        # Complete all steps
        progress.completed_steps = 5
        progress.failed_steps = 3
        progress.skipped_steps = 2
        
        assert progress.progress_percentage == 100.0
        assert progress.is_completed
    
    def test_execution_progress_duration(self):
        """Test execution duration calculation."""
        progress = ExecutionProgress("exec1", 5)
        
        # No duration without start time
        assert progress.duration is None
        
        # Duration with start time only
        progress.start_time = datetime.now()
        time.sleep(0.01)  # Small delay
        duration = progress.duration
        assert duration is not None
        assert duration.total_seconds() > 0
        
        # Duration with both start and end time
        progress.end_time = datetime.now()
        final_duration = progress.duration
        assert final_duration is not None
        assert final_duration >= duration


class TestProgressTracker:
    """Test ProgressTracker functionality."""
    
    @pytest.fixture
    def execution_context(self):
        """Create test execution context."""
        return ExecutionContext("test_exec", "test_pipeline")
    
    @pytest.fixture
    def variable_manager(self):
        """Create test variable manager."""
        return VariableManager("test_pipeline")
    
    @pytest.fixture
    def progress_tracker(self, execution_context, variable_manager):
        """Create test progress tracker."""
        return ProgressTracker(execution_context, variable_manager)
    
    def test_progress_tracker_creation(self, execution_context, variable_manager):
        """Test creating progress tracker."""
        tracker = ProgressTracker(execution_context, variable_manager)
        assert tracker.execution_context == execution_context
        assert tracker.variable_manager == variable_manager
        assert len(tracker._executions) == 0
        assert len(tracker._steps) == 0
    
    def test_execution_tracking(self, progress_tracker):
        """Test execution tracking lifecycle."""
        execution_id = "test_exec"
        
        # Start execution
        progress_tracker.start_execution(execution_id, 5)
        
        execution_progress = progress_tracker.get_execution_progress(execution_id)
        assert execution_progress is not None
        assert execution_progress.total_steps == 5
        assert execution_progress.start_time is not None
        
        # Complete execution
        progress_tracker.complete_execution(execution_id, success=True)
        
        execution_progress = progress_tracker.get_execution_progress(execution_id)
        assert execution_progress.end_time is not None
    
    def test_step_tracking(self, progress_tracker):
        """Test step tracking lifecycle."""
        execution_id = "test_exec"
        step_id = "step1"
        step_name = "Test Step"
        
        # Start execution first
        progress_tracker.start_execution(execution_id, 1)
        
        # Start step
        progress_tracker.start_step(execution_id, step_id, step_name)
        
        step_progress = progress_tracker.get_step_progress(execution_id, step_id)
        assert step_progress is not None
        assert step_progress.step_id == step_id
        assert step_progress.step_name == step_name
        assert step_progress.status == StepStatus.RUNNING
        
        # Complete step
        progress_tracker.complete_step(execution_id, step_id, success=True)
        
        step_progress = progress_tracker.get_step_progress(execution_id, step_id)
        assert step_progress.status == StepStatus.COMPLETED
        assert step_progress.progress_percentage == 100.0
    
    def test_step_failure(self, progress_tracker):
        """Test step failure tracking."""
        execution_id = "test_exec"
        step_id = "step1"
        
        progress_tracker.start_execution(execution_id, 1)
        progress_tracker.start_step(execution_id, step_id, "Test Step")
        
        # Fail step
        error_message = "Test error"
        progress_tracker.complete_step(execution_id, step_id, success=False, error_message=error_message)
        
        step_progress = progress_tracker.get_step_progress(execution_id, step_id)
        assert step_progress.status == StepStatus.FAILED
        assert step_progress.error_message == error_message
    
    def test_step_skip(self, progress_tracker):
        """Test step skipping."""
        execution_id = "test_exec"
        step_id = "step1"
        
        progress_tracker.start_execution(execution_id, 1)
        progress_tracker.start_step(execution_id, step_id, "Test Step")
        
        # Skip step
        reason = "Condition not met"
        progress_tracker.skip_step(execution_id, step_id, reason)
        
        step_progress = progress_tracker.get_step_progress(execution_id, step_id)
        assert step_progress.status == StepStatus.SKIPPED
        assert step_progress.metadata["skip_reason"] == reason
    
    def test_progress_update(self, progress_tracker):
        """Test step progress updates."""
        execution_id = "test_exec"
        step_id = "step1"
        
        progress_tracker.start_execution(execution_id, 1)
        progress_tracker.start_step(execution_id, step_id, "Test Step")
        
        # Update progress
        progress_tracker.update_step_progress(execution_id, step_id, 50.0, "Half done")
        
        step_progress = progress_tracker.get_step_progress(execution_id, step_id)
        assert step_progress.progress_percentage == 50.0
        assert step_progress.metadata["last_message"] == "Half done"
    
    def test_event_handling(self, progress_tracker):
        """Test progress event handling."""
        events = []
        
        def event_handler(event: ProgressEvent):
            events.append(event)
        
        progress_tracker.add_event_handler(event_handler)
        
        # Start execution and step
        execution_id = "test_exec"
        progress_tracker.start_execution(execution_id, 1)
        progress_tracker.start_step(execution_id, "step1", "Test Step")
        progress_tracker.complete_step(execution_id, "step1", success=True)
        
        # Should have received events
        assert len(events) >= 3  # execution_started, step_started, step_completed
        
        event_types = [event.event_type for event in events]
        assert ProgressEventType.EXECUTION_STARTED in event_types
        assert ProgressEventType.STEP_STARTED in event_types
        assert ProgressEventType.STEP_COMPLETED in event_types
    
    def test_track_step_context_manager(self, progress_tracker):
        """Test step tracking context manager."""
        execution_id = "test_exec"
        step_id = "step1"
        
        progress_tracker.start_execution(execution_id, 1)
        
        # Successful execution
        with progress_tracker.track_step(execution_id, step_id, "Test Step"):
            pass
        
        step_progress = progress_tracker.get_step_progress(execution_id, step_id)
        assert step_progress.status == StepStatus.COMPLETED
        
        # Failed execution
        step_id2 = "step2"
        with pytest.raises(ValueError):
            with progress_tracker.track_step(execution_id, step_id2, "Test Step 2"):
                raise ValueError("Test error")
        
        step_progress2 = progress_tracker.get_step_progress(execution_id, step_id2)
        assert step_progress2.status == StepStatus.FAILED
        assert "Test error" in step_progress2.error_message
    
    def test_variable_integration(self, progress_tracker):
        """Test integration with variable manager."""
        execution_id = "test_exec"
        step_id = "step1"
        
        progress_tracker.start_execution(execution_id, 1)
        progress_tracker.start_step(execution_id, step_id, "Test Step")
        
        # Set progress variable
        progress_var_name = f"progress.{step_id}.percentage"
        progress_tracker.variable_manager.set_variable(progress_var_name, 75.0)
        
        # Allow some time for variable change handler to process
        time.sleep(0.01)
        
        # Check that progress was updated
        step_progress = progress_tracker.get_step_progress(execution_id, step_id)
        assert step_progress.progress_percentage == 75.0
    
    def test_cleanup(self, progress_tracker):
        """Test progress tracker cleanup."""
        execution_id = "test_exec"
        
        progress_tracker.start_execution(execution_id, 1)
        progress_tracker.start_step(execution_id, "step1", "Test Step")
        
        # Verify data exists
        assert execution_id in progress_tracker._executions
        assert execution_id in progress_tracker._steps
        
        # Cleanup specific execution
        progress_tracker.cleanup(execution_id)
        
        assert execution_id not in progress_tracker._executions
        assert execution_id not in progress_tracker._steps
    
    def test_performance_metrics(self, progress_tracker):
        """Test progress tracker performance metrics."""
        # Add some event handlers
        def handler1(event):
            time.sleep(0.001)  # Small delay
        
        def handler2(event):
            pass
        
        progress_tracker.add_event_handler(handler1)
        progress_tracker.add_event_handler(handler2)
        
        # Generate some events
        execution_id = "test_exec"
        progress_tracker.start_execution(execution_id, 1)
        progress_tracker.start_step(execution_id, "step1", "Test Step")
        progress_tracker.complete_step(execution_id, "step1", success=True)
        
        # Get performance metrics
        metrics = progress_tracker.get_performance_metrics()
        
        assert "total_events" in metrics
        assert "active_executions" in metrics
        assert "handler_performance" in metrics
        assert metrics["total_events"] > 0
        assert len(metrics["handler_performance"]) >= 2  # Two handlers


class TestProgressTrackerFactory:
    """Test progress tracker factory functions."""
    
    def test_create_progress_tracker(self):
        """Test creating progress tracker with factory."""
        execution_context = ExecutionContext("test_exec", "test_pipeline")
        variable_manager = VariableManager("test_pipeline")
        
        tracker = create_progress_tracker(execution_context, variable_manager)
        
        assert isinstance(tracker, ProgressTracker)
        assert tracker.execution_context == execution_context
        assert tracker.variable_manager == variable_manager
    
    def test_create_progress_tracker_minimal(self):
        """Test creating progress tracker with minimal parameters."""
        tracker = create_progress_tracker()
        
        assert isinstance(tracker, ProgressTracker)
        assert tracker.execution_context is None
        assert tracker.variable_manager is None


@pytest.mark.asyncio
class TestProgressTrackerAsync:
    """Test asynchronous operations with progress tracker."""
    
    async def test_concurrent_step_tracking(self):
        """Test concurrent step tracking."""
        tracker = ProgressTracker()
        execution_id = "test_exec"
        
        tracker.start_execution(execution_id, 3)
        
        async def track_step(step_id: str, delay: float):
            tracker.start_step(execution_id, step_id, f"Step {step_id}")
            await asyncio.sleep(delay)
            tracker.complete_step(execution_id, step_id, success=True)
        
        # Run steps concurrently
        await asyncio.gather(
            track_step("step1", 0.01),
            track_step("step2", 0.02),
            track_step("step3", 0.01)
        )
        
        # All steps should be completed
        all_steps = tracker.get_all_step_progress(execution_id)
        assert len(all_steps) == 3
        
        for step_id, step_progress in all_steps.items():
            assert step_progress.status == StepStatus.COMPLETED
    
    async def test_real_time_handlers(self):
        """Test real-time progress handlers."""
        tracker = ProgressTracker()
        real_time_events = []
        
        def real_time_handler(event_type: str, data: dict):
            real_time_events.append((event_type, data))
        
        tracker.add_real_time_handler(real_time_handler)
        
        execution_id = "test_exec"
        tracker.start_execution(execution_id, 1)
        tracker.start_step(execution_id, "step1", "Test Step")
        tracker.complete_step(execution_id, "step1", success=True)
        
        # Should have received real-time events
        assert len(real_time_events) >= 3
        
        event_types = [event[0] for event in real_time_events]
        assert "execution_started" in event_types
        assert "step_started" in event_types
        assert "step_completed" in event_types


if __name__ == "__main__":
    pytest.main([__file__])