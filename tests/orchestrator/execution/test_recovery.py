"""
Tests for recovery and error handling system.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

from src.orchestrator.execution.recovery import (
    RecoveryManager,
    RecoveryStrategy,
    ErrorSeverity,
    ErrorCategory,
    ErrorInfo,
    RetryConfig,
    RecoveryPlan,
    create_recovery_manager,
    network_error_handler,
    timeout_error_handler,
    critical_error_handler
)
from src.orchestrator.execution.state import ExecutionContext
from src.orchestrator.execution.progress import ProgressTracker


class TestErrorInfo:
    """Test ErrorInfo functionality."""
    
    def test_error_info_creation(self):
        """Test creating error info."""
        error = ValueError("Test error")
        error_info = ErrorInfo(
            error_id="test_001",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            message="Test error",
            exception=error
        )
        
        assert error_info.error_id == "test_001"
        assert error_info.category == ErrorCategory.VALIDATION
        assert error_info.severity == ErrorSeverity.MEDIUM
        assert error_info.message == "Test error"
        assert error_info.exception == error
        assert error_info.timestamp is not None
    
    def test_error_info_serialization(self):
        """Test error info serialization."""
        error = ValueError("Test error")
        error_info = ErrorInfo(
            error_id="test_001",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            message="Test error",
            exception=error
        )
        
        data = error_info.to_dict()
        
        assert data["error_id"] == "test_001"
        assert data["category"] == "validation"
        assert data["severity"] == "medium"
        assert data["message"] == "Test error"
        assert data["exception"] == "Test error"
        assert "timestamp" in data


class TestRetryConfig:
    """Test RetryConfig functionality."""
    
    def test_retry_config_creation(self):
        """Test creating retry config."""
        config = RetryConfig(max_attempts=5, initial_delay=2.0)
        assert config.max_attempts == 5
        assert config.initial_delay == 2.0
        assert config.exponential_backoff is True
    
    def test_should_retry_logic(self):
        """Test retry decision logic."""
        config = RetryConfig(max_attempts=3)
        
        # Network error should be retried
        network_error = ErrorInfo(
            error_id="test",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            message="Network error"
        )
        
        assert config.should_retry(network_error, 1) is True
        assert config.should_retry(network_error, 2) is True
        assert config.should_retry(network_error, 3) is False  # Max attempts reached
        
        # Validation error should not be retried by default
        validation_error = ErrorInfo(
            error_id="test",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            message="Validation error"
        )
        
        assert config.should_retry(validation_error, 1) is False
    
    def test_retry_delay_calculation(self):
        """Test retry delay calculation."""
        config = RetryConfig(
            initial_delay=1.0,
            backoff_factor=2.0,
            max_delay=10.0,
            exponential_backoff=True
        )
        
        assert config.get_delay(1) == 2.0  # 1.0 * 2^1
        assert config.get_delay(2) == 4.0  # 1.0 * 2^2
        assert config.get_delay(3) == 8.0  # 1.0 * 2^3
        assert config.get_delay(4) == 10.0  # Capped at max_delay
    
    def test_custom_retry_condition(self):
        """Test custom retry condition."""
        def custom_condition(error_info: ErrorInfo) -> bool:
            return "retry" in error_info.message.lower()
        
        config = RetryConfig(retry_condition=custom_condition)
        
        retry_error = ErrorInfo(
            error_id="test",
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.MEDIUM,
            message="Retry this error"
        )
        
        no_retry_error = ErrorInfo(
            error_id="test",
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.MEDIUM,
            message="Do not retry"
        )
        
        assert config.should_retry(retry_error, 1) is True
        assert config.should_retry(no_retry_error, 1) is False


class TestRecoveryPlan:
    """Test RecoveryPlan functionality."""
    
    def test_recovery_plan_creation(self):
        """Test creating recovery plan."""
        plan = RecoveryPlan(
            strategy=RecoveryStrategy.RETRY,
            retry_config=RetryConfig(max_attempts=3)
        )
        
        assert plan.strategy == RecoveryStrategy.RETRY
        assert plan.retry_config.max_attempts == 3
        assert plan.is_automated() is True
    
    def test_manual_recovery_plan(self):
        """Test manual recovery plan."""
        plan = RecoveryPlan(
            strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            manual_instructions="Check system logs"
        )
        
        assert plan.strategy == RecoveryStrategy.MANUAL_INTERVENTION
        assert plan.manual_instructions == "Check system logs"
        assert plan.is_automated() is False


class TestRecoveryManager:
    """Test RecoveryManager functionality."""
    
    @pytest.fixture
    def execution_context(self):
        """Create test execution context."""
        return ExecutionContext("test_exec", "test_pipeline")
    
    @pytest.fixture
    def progress_tracker(self, execution_context):
        """Create test progress tracker."""
        return ProgressTracker(execution_context)
    
    @pytest.fixture
    def recovery_manager(self, execution_context, progress_tracker):
        """Create test recovery manager."""
        return RecoveryManager(execution_context, progress_tracker)
    
    def test_recovery_manager_creation(self, execution_context, progress_tracker):
        """Test creating recovery manager."""
        manager = RecoveryManager(execution_context, progress_tracker)
        assert manager.execution_context == execution_context
        assert manager.progress_tracker == progress_tracker
        assert len(manager._error_history) == 0
    
    def test_error_classification(self, recovery_manager):
        """Test error classification."""
        # Network error
        network_error = ConnectionError("Connection failed")
        error_info = recovery_manager.classify_error(network_error)
        assert error_info.category == ErrorCategory.NETWORK
        assert error_info.severity == ErrorSeverity.MEDIUM
        
        # Validation error
        validation_error = ValueError("Invalid format")
        error_info = recovery_manager.classify_error(validation_error)
        assert error_info.category == ErrorCategory.VALIDATION
        assert error_info.severity == ErrorSeverity.LOW
        
        # System error
        system_error = OSError("System error")
        error_info = recovery_manager.classify_error(system_error)
        assert error_info.category == ErrorCategory.SYSTEM
    
    def test_error_handler_registration(self, recovery_manager):
        """Test error handler registration."""
        def custom_handler(error_info: ErrorInfo) -> RecoveryPlan:
            return RecoveryPlan(strategy=RecoveryStrategy.SKIP)
        
        recovery_manager.register_error_handler(ErrorCategory.VALIDATION, custom_handler)
        
        # Test that handler is called
        validation_error = ValueError("Test error")
        recovery_plan = recovery_manager.handle_error(validation_error, "step1")
        
        assert recovery_plan.strategy == RecoveryStrategy.SKIP
    
    def test_global_error_handler(self, recovery_manager):
        """Test global error handler."""
        def global_handler(error_info: ErrorInfo) -> RecoveryPlan:
            if error_info.severity == ErrorSeverity.CRITICAL:
                return RecoveryPlan(strategy=RecoveryStrategy.FAIL_FAST)
            return None
        
        recovery_manager.register_global_error_handler(global_handler)
        
        # Mock the severity determination to return CRITICAL
        with patch.object(recovery_manager, '_determine_error_severity', return_value=ErrorSeverity.CRITICAL):
            critical_error = Exception("Critical error")
            recovery_plan = recovery_manager.handle_error(critical_error, "step1")
            assert recovery_plan.strategy == RecoveryStrategy.FAIL_FAST
    
    def test_step_recovery_plan(self, recovery_manager):
        """Test step-specific recovery plan."""
        step_plan = RecoveryPlan(strategy=RecoveryStrategy.ROLLBACK)
        recovery_manager.set_recovery_plan("step1", step_plan)
        
        # Error in step1 should use the specific plan
        error = Exception("Test error")
        recovery_plan = recovery_manager.handle_error(error, "step1")
        
        assert recovery_plan.strategy == RecoveryStrategy.ROLLBACK
    
    def test_error_history(self, recovery_manager):
        """Test error history tracking."""
        error1 = ValueError("Error 1")
        error2 = ConnectionError("Error 2")
        
        recovery_manager.handle_error(error1, "step1")
        recovery_manager.handle_error(error2, "step2")
        
        history = recovery_manager.get_error_history()
        assert len(history) == 2
        
        # Test filtering
        validation_history = recovery_manager.get_error_history(category=ErrorCategory.VALIDATION)
        assert len(validation_history) == 1
        assert validation_history[0].message == "Error 1"
        
        step1_history = recovery_manager.get_error_history(step_id="step1")
        assert len(step1_history) == 1
        assert step1_history[0].step_id == "step1"
    
    def test_retry_count_tracking(self, recovery_manager):
        """Test retry count tracking."""
        assert recovery_manager.get_retry_count("step1") == 0
        
        # Simulate retry attempts
        recovery_manager._retry_counts["step1"] = 3
        assert recovery_manager.get_retry_count("step1") == 3
        
        recovery_manager.reset_retry_count("step1")
        assert recovery_manager.get_retry_count("step1") == 0


@pytest.mark.asyncio
class TestRecoveryExecution:
    """Test recovery execution functionality."""
    
    @pytest.fixture
    def recovery_manager(self):
        """Create test recovery manager."""
        return RecoveryManager()
    
    async def test_fail_fast_recovery(self, recovery_manager):
        """Test fail fast recovery strategy."""
        plan = RecoveryPlan(strategy=RecoveryStrategy.FAIL_FAST)
        
        async def failing_executor():
            raise ValueError("Test error")
        
        success = await recovery_manager.execute_recovery("step1", plan, failing_executor)
        assert success is False
    
    async def test_skip_recovery(self, recovery_manager):
        """Test skip recovery strategy."""
        plan = RecoveryPlan(strategy=RecoveryStrategy.SKIP)
        
        async def failing_executor():
            raise ValueError("Test error")
        
        success = await recovery_manager.execute_recovery("step1", plan, failing_executor)
        assert success is True
    
    async def test_retry_recovery_success(self, recovery_manager):
        """Test successful retry recovery."""
        attempt_count = 0
        
        async def executor():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Network error")
            return "success"
        
        plan = RecoveryPlan(
            strategy=RecoveryStrategy.RETRY,
            retry_config=RetryConfig(max_attempts=5, initial_delay=0.01)
        )
        
        success = await recovery_manager.execute_recovery("step1", plan, executor)
        assert success is True
        assert attempt_count == 3
    
    async def test_retry_recovery_failure(self, recovery_manager):
        """Test failed retry recovery."""
        async def failing_executor():
            raise ConnectionError("Persistent network error")
        
        plan = RecoveryPlan(
            strategy=RecoveryStrategy.RETRY,
            retry_config=RetryConfig(max_attempts=2, initial_delay=0.01)
        )
        
        success = await recovery_manager.execute_recovery("step1", plan, failing_executor)
        assert success is False
        assert recovery_manager.get_retry_count("step1") == 2
    
    async def test_retry_with_backoff(self, recovery_manager):
        """Test retry with exponential backoff."""
        start_time = time.time()
        
        async def failing_executor():
            raise ConnectionError("Network error")
        
        plan = RecoveryPlan(
            strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
            retry_config=RetryConfig(
                max_attempts=3,
                initial_delay=0.01,
                backoff_factor=2.0
            )
        )
        
        success = await recovery_manager.execute_recovery("step1", plan, failing_executor)
        
        elapsed = time.time() - start_time
        # Should have taken at least the sum of delays
        # First attempt: immediate, second: 0.01s, third: 0.02s
        # Total: ~0.03s minimum
        assert elapsed >= 0.025  # Allow some margin for timing
        assert success is False
    
    async def test_rollback_recovery(self, recovery_manager):
        """Test rollback recovery strategy."""
        # Create execution context with checkpoint capability
        execution_context = ExecutionContext("test_exec", "test_pipeline")
        recovery_manager.execution_context = execution_context
        
        # Create a checkpoint
        checkpoint = execution_context.create_checkpoint("before_error")
        
        plan = RecoveryPlan(
            strategy=RecoveryStrategy.ROLLBACK,
            target_checkpoint=checkpoint.id
        )
        
        success = await recovery_manager.execute_recovery("step1", plan, None)
        # Should succeed if checkpoint exists
        assert success is True
    
    async def test_manual_intervention_recovery(self, recovery_manager):
        """Test manual intervention recovery strategy."""
        plan = RecoveryPlan(
            strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            manual_instructions="Check system configuration"
        )
        
        success = await recovery_manager.execute_recovery("step1", plan, None)
        # Manual intervention should return False (requires manual action)
        assert success is False


class TestErrorHandlers:
    """Test built-in error handlers."""
    
    def test_network_error_handler(self):
        """Test network error handler."""
        network_error = ErrorInfo(
            error_id="test",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            message="Network error"
        )
        
        plan = network_error_handler(network_error)
        assert plan is not None
        assert plan.strategy == RecoveryStrategy.RETRY_WITH_BACKOFF
        assert plan.retry_config.max_attempts == 5
        
        # Non-network error should return None
        other_error = ErrorInfo(
            error_id="test",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            message="Validation error"
        )
        
        plan = network_error_handler(other_error)
        assert plan is None
    
    def test_timeout_error_handler(self):
        """Test timeout error handler."""
        timeout_error = ErrorInfo(
            error_id="test",
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            message="Timeout error"
        )
        
        plan = timeout_error_handler(timeout_error)
        assert plan is not None
        assert plan.strategy == RecoveryStrategy.RETRY
        assert plan.retry_config.max_attempts == 3
        
        # Non-timeout error should return None
        other_error = ErrorInfo(
            error_id="test",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            message="Validation error"
        )
        
        plan = timeout_error_handler(other_error)
        assert plan is None
    
    def test_critical_error_handler(self):
        """Test critical error handler."""
        critical_error = ErrorInfo(
            error_id="test",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            message="Critical system error"
        )
        
        plan = critical_error_handler(critical_error)
        assert plan is not None
        assert plan.strategy == RecoveryStrategy.FAIL_FAST
        assert "Critical system error" in plan.manual_instructions
        
        # Non-critical error should return None
        other_error = ErrorInfo(
            error_id="test",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            message="Validation error"
        )
        
        plan = critical_error_handler(other_error)
        assert plan is None


class TestRecoveryManagerIntegration:
    """Test recovery manager integration with other components."""
    
    def test_context_manager_error_handling(self):
        """Test context manager for error handling."""
        recovery_manager = RecoveryManager()
        
        with pytest.raises(ValueError):
            with recovery_manager.handle_step_errors("step1", "Test Step"):
                raise ValueError("Test error")
        
        # Error should be in history
        history = recovery_manager.get_error_history()
        assert len(history) == 1
        assert history[0].step_id == "step1"
        assert history[0].message == "Test error"
    
    def test_recovery_status(self):
        """Test recovery status reporting."""
        recovery_manager = RecoveryManager()
        
        # Add some errors
        recovery_manager.handle_error(ValueError("Error 1"), "step1")
        recovery_manager.handle_error(ConnectionError("Error 2"), "step2")
        
        status = recovery_manager.get_recovery_status()
        
        assert status["total_errors"] == 2
        assert status["error_counts_by_category"]["validation"] == 1
        assert status["error_counts_by_category"]["network"] == 1
        assert status["active_recoveries"] == 2  # Two active recovery plans
    
    def test_cleanup(self):
        """Test recovery manager cleanup."""
        recovery_manager = RecoveryManager()
        
        # Add some data
        recovery_manager.handle_error(ValueError("Error 1"), "step1")
        recovery_manager._retry_counts["step1"] = 2
        recovery_manager._active_recoveries["step1"] = RecoveryPlan(strategy=RecoveryStrategy.RETRY)
        
        # Cleanup
        recovery_manager.cleanup()
        
        assert len(recovery_manager._error_history) == 0
        assert len(recovery_manager._retry_counts) == 0
        assert len(recovery_manager._active_recoveries) == 0


class TestRecoveryFactory:
    """Test recovery manager factory functions."""
    
    def test_create_recovery_manager(self):
        """Test creating recovery manager with factory."""
        execution_context = ExecutionContext("test_exec", "test_pipeline")
        progress_tracker = ProgressTracker(execution_context)
        
        manager = create_recovery_manager(execution_context, progress_tracker)
        
        assert isinstance(manager, RecoveryManager)
        assert manager.execution_context == execution_context
        assert manager.progress_tracker == progress_tracker
    
    def test_create_recovery_manager_minimal(self):
        """Test creating recovery manager with minimal parameters."""
        manager = create_recovery_manager()
        
        assert isinstance(manager, RecoveryManager)
        assert manager.execution_context is None
        assert manager.progress_tracker is None


if __name__ == "__main__":
    pytest.main([__file__])