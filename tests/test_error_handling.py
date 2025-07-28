"""Tests for error handling and recovery functionality."""

import time

import pytest

from src.orchestrator.core.error_handler import (
    CircuitBreaker,
    ErrorCategory,
    ErrorHandler,
    ErrorSeverity,
    RecoveryManager,
    RetryStrategy)


class TestErrorHandler:
    """Test cases for ErrorHandler class."""

    def test_error_handler_creation(self):
        """Test basic error handler creation."""
        handler = ErrorHandler()

        assert handler.error_strategies is not None
        assert handler.circuit_breaker is not None
        assert ErrorCategory.RATE_LIMIT in handler.error_strategies
        assert ErrorCategory.TIMEOUT in handler.error_strategies
        assert ErrorCategory.RESOURCE_EXHAUSTION in handler.error_strategies

    @pytest.mark.asyncio
    async def test_handle_rate_limit_error(self):
        """Test handling rate limit errors."""
        handler = ErrorHandler()

        # Simulate rate limit error
        error = Exception("Rate limit exceeded")
        context = {"system_id": "test_system", "retry_count": 0}

        result = await handler.handle_error(error, context)

        assert result["action"] in ["retry", "switch_system"]
        if result["action"] == "retry":
            assert "delay" in result
            assert result["delay"] > 0

    @pytest.mark.asyncio
    async def test_handle_timeout_error(self):
        """Test handling timeout errors."""
        handler = ErrorHandler()

        error = Exception("Request timeout")
        context = {"timeout": 30, "retry_count": 0}

        result = await handler.handle_error(error, context)

        assert result["action"] in ["retry", "fail"]
        if result["action"] == "retry":
            assert result["timeout"] > context["timeout"]
            assert result["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_handle_resource_exhaustion_error(self):
        """Test handling resource exhaustion errors."""
        handler = ErrorHandler()

        error = Exception("Out of memory")
        context = {"system_id": "test_system", "memory_usage": 0.9}

        result = await handler.handle_error(error, context)

        assert result["action"] in ["retry", "switch_system", "scale_up"]
        if result["action"] == "switch_system":
            assert "reason" in result
            assert result["reason"] == "resource_exhaustion"

    @pytest.mark.asyncio
    async def test_handle_validation_error(self):
        """Test handling validation errors."""
        handler = ErrorHandler()

        error = ValueError("Invalid input format")
        context = {"input_data": "invalid_json"}

        result = await handler.handle_error(error, context)

        assert result["action"] in ["fail", "sanitize_input"]
        if result["action"] == "sanitize_input":
            assert "sanitized_input" in result

    @pytest.mark.asyncio
    async def test_handle_system_error(self):
        """Test handling system errors."""
        handler = ErrorHandler()

        error = RuntimeError("System malfunction")
        context = {"system_id": "test_system"}

        result = await handler.handle_error(error, context)

        assert result["action"] in ["restart", "switch_system", "fail"]
        if result["action"] == "restart":
            assert "restart_delay" in result

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration."""
        handler = ErrorHandler()

        # Test that circuit breaker tracking works
        context = {"system_id": "failing_system"}

        for i in range(6):  # Exceed failure threshold
            error = Exception(f"Failure {i}")
            result = await handler.handle_error(error, context)
            # Update retry count for next iteration
            context["retry_count"] = result.get("retry_count", 0)

        # After many failures, should get fail action
        final_error = Exception("Another failure")
        final_result = await handler.handle_error(final_error, context)
        assert final_result["action"] == "fail"  # Should fail after many retries

    def test_classify_error_rate_limit(self):
        """Test error classification for rate limits."""
        handler = ErrorHandler()

        errors = [
            Exception("Rate limit exceeded"),
            Exception("Too many requests"),
            Exception("API quota exceeded"),
        ]

        for error in errors:
            category = handler._classify_error(error)
            assert category == ErrorCategory.RATE_LIMIT

    def test_classify_error_timeout(self):
        """Test error classification for timeouts."""
        handler = ErrorHandler()

        errors = [
            Exception("Request timeout"),
            Exception("Connection timeout"),
            Exception("Read timeout"),
        ]

        for error in errors:
            category = handler._classify_error(error)
            assert category == ErrorCategory.TIMEOUT

    def test_classify_error_resource_exhaustion(self):
        """Test error classification for resource exhaustion."""
        handler = ErrorHandler()

        errors = [
            Exception("Out of memory"),
            Exception("Disk full"),
            Exception("CPU overload"),
        ]

        for error in errors:
            category = handler._classify_error(error)
            assert category == ErrorCategory.RESOURCE_EXHAUSTION

    def test_determine_severity_critical(self):
        """Test determining critical error severity."""
        handler = ErrorHandler()

        error = Exception("System crash")
        context = {"system_id": "critical_system", "pipeline_id": "important_pipeline"}

        severity = handler._determine_severity(error, context)

        assert severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]

    def test_determine_severity_low(self):
        """Test determining low error severity."""
        handler = ErrorHandler()

        error = Exception("Minor warning")
        context = {"system_id": "test_system", "retry_count": 0}

        severity = handler._determine_severity(error, context)

        assert severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]

    @pytest.mark.asyncio
    async def test_log_error(self):
        """Test error logging functionality."""
        handler = ErrorHandler()

        error = Exception("Test error")
        context = {"system_id": "test_system"}

        # Should not raise exception
        await handler._log_error_async(
            error, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.MEDIUM, context
        )

    def test_extract_retry_after(self):
        """Test extracting retry-after header."""
        handler = ErrorHandler()

        # Test with retry-after in exception message
        error = Exception("Rate limit exceeded. Retry after 60 seconds")
        retry_after = handler._extract_retry_after(error)

        assert retry_after == 60

        # Test without retry-after
        error = Exception("Generic error")
        retry_after = handler._extract_retry_after(error)

        assert retry_after is None


class TestCircuitBreaker:
    """Test cases for CircuitBreaker class."""

    def test_circuit_breaker_creation(self):
        """Test basic circuit breaker creation."""
        breaker = CircuitBreaker(failure_threshold=5)

        assert breaker.failure_threshold == 5
        assert breaker.timeout == 60.0
        assert breaker.failures == {}
        assert breaker.last_failure_time == {}

    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker initial state."""
        breaker = CircuitBreaker()

        # Should be closed initially
        assert breaker.is_open("test_system") is False

    def test_circuit_breaker_record_failure(self):
        """Test recording failures."""
        breaker = CircuitBreaker(failure_threshold=3)

        # Record failures
        breaker.record_failure("test_system")
        breaker.record_failure("test_system")

        assert breaker.failures["test_system"] == 2
        assert breaker.is_open("test_system") is False

        # Third failure should open circuit
        breaker.record_failure("test_system")
        assert breaker.is_open("test_system") is True

    def test_circuit_breaker_record_success(self):
        """Test recording successes."""
        breaker = CircuitBreaker(failure_threshold=3)

        # Record some failures
        breaker.record_failure("test_system")
        breaker.record_failure("test_system")

        # Record success
        breaker.record_success("test_system")

        assert breaker.failures["test_system"] == 1
        assert breaker.is_open("test_system") is False

    def test_circuit_breaker_timeout_recovery(self):
        """Test circuit breaker timeout recovery."""
        breaker = CircuitBreaker(failure_threshold=2)

        # Open circuit
        breaker.record_failure("test_system")
        breaker.record_failure("test_system")
        assert breaker.is_open("test_system") is True

        # Wait for timeout
        time.sleep(0.2)

        # Should be closed again
        assert breaker.is_open("test_system") is False

    def test_circuit_breaker_multiple_systems(self):
        """Test circuit breaker with multiple systems."""
        breaker = CircuitBreaker(failure_threshold=2)

        # Fail system1
        breaker.record_failure("system1")
        breaker.record_failure("system1")

        # Fail system2
        breaker.record_failure("system2")

        assert breaker.is_open("system1") is True
        assert breaker.is_open("system2") is False

    def test_circuit_breaker_state_transitions(self):
        """Test circuit breaker state transitions."""
        breaker = CircuitBreaker(failure_threshold=2)

        # Initial state: CLOSED
        assert breaker.is_open("test_system") is False

        # Record failures to open circuit
        breaker.record_failure("test_system")
        breaker.record_failure("test_system")

        # State: OPEN
        assert breaker.is_open("test_system") is True

        # Wait for timeout
        time.sleep(0.2)

        # State: HALF_OPEN (closed but ready to open again)
        assert breaker.is_open("test_system") is False

        # Success should keep it closed
        breaker.record_success("test_system")
        assert breaker.is_open("test_system") is False


class TestRetryStrategy:
    """Test cases for RetryStrategy class."""

    def test_retry_strategy_creation(self):
        """Test basic retry strategy creation."""
        strategy = RetryStrategy(max_retries=3, base_delay=1.0, backoff_factor=2.0)

        assert strategy.max_retries == 3
        assert strategy.base_delay == 1.0
        assert strategy.backoff_factor == 2.0

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        strategy = RetryStrategy(base_delay=1.0, backoff_factor=2.0)

        assert strategy.calculate_delay(0) == 1.0
        assert strategy.calculate_delay(1) == 2.0
        assert strategy.calculate_delay(2) == 4.0
        assert strategy.calculate_delay(3) == 8.0

    def test_linear_backoff(self):
        """Test linear backoff calculation."""
        strategy = RetryStrategy(
            base_delay=1.0, backoff_factor=1.0, strategy_type="linear"
        )

        assert strategy.calculate_delay(0) == 1.0
        assert strategy.calculate_delay(1) == 2.0
        assert strategy.calculate_delay(2) == 3.0
        assert strategy.calculate_delay(3) == 4.0

    def test_fixed_backoff(self):
        """Test fixed backoff calculation."""
        strategy = RetryStrategy(base_delay=2.0, strategy_type="fixed")

        assert strategy.calculate_delay(0) == 2.0
        assert strategy.calculate_delay(1) == 2.0
        assert strategy.calculate_delay(2) == 2.0
        assert strategy.calculate_delay(3) == 2.0

    def test_should_retry_within_limit(self):
        """Test should retry within limit."""
        strategy = RetryStrategy(max_retries=3)

        assert strategy.should_retry(0) is True
        assert strategy.should_retry(1) is True
        assert strategy.should_retry(2) is True
        assert strategy.should_retry(3) is False

    def test_should_retry_with_conditions(self):
        """Test should retry with specific conditions."""
        strategy = RetryStrategy(
            max_retries=3, retryable_errors=[ValueError, TypeError]
        )

        # Retryable errors
        assert strategy.should_retry_error(ValueError("test")) is True
        assert strategy.should_retry_error(TypeError("test")) is True

        # Non-retryable errors
        assert strategy.should_retry_error(RuntimeError("test")) is False
        assert strategy.should_retry_error(KeyError("test")) is False

    def test_jitter_application(self):
        """Test jitter application to delays."""
        strategy = RetryStrategy(base_delay=1.0, jitter=True)

        # With jitter, delays should vary
        delays = [strategy.calculate_delay(0) for _ in range(10)]

        # All delays should be around 1.0 but with some variation
        assert all(0.5 <= delay <= 1.5 for delay in delays)
        assert len(set(delays)) > 1  # Should have some variation

    @pytest.mark.asyncio
    async def test_retry_decorator(self):
        """Test retry decorator functionality."""
        strategy = RetryStrategy(max_retries=3, base_delay=0.01)

        call_count = 0

        @strategy.retry_decorator
        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = await failing_function()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_decorator_max_retries_exceeded(self):
        """Test retry decorator when max retries exceeded."""
        strategy = RetryStrategy(max_retries=2, base_delay=0.01)

        call_count = 0

        @strategy.retry_decorator
        async def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent failure")

        with pytest.raises(ValueError):
            await always_failing_function()

        assert call_count == 3  # Initial call + 2 retries


class TestRecoveryManager:
    """Test cases for RecoveryManager class."""

    def test_recovery_manager_creation(self):
        """Test basic recovery manager creation."""
        manager = RecoveryManager()

        assert manager.recovery_strategies is not None
        assert manager.recovery_history == []
        assert manager.state_manager is not None

    @pytest.mark.asyncio
    async def test_recover_from_checkpoint(self):
        """Test recovering from checkpoint."""
        manager = RecoveryManager()

        # Mock checkpoint data
        checkpoint_data = {
            "pipeline_id": "test_pipeline",
            "completed_tasks": ["task1", "task2"],
            "failed_task": "task3",
            "timestamp": "2023-01-01T00:00:00Z",
        }

        recovery_result = await manager.recover_from_checkpoint(checkpoint_data)

        assert recovery_result["success"] is True
        assert recovery_result["resumed_from"] == "task3"
        assert recovery_result["completed_tasks"] == ["task1", "task2"]

    @pytest.mark.asyncio
    async def test_recover_from_failure(self):
        """Test recovering from task failure."""
        manager = RecoveryManager()

        failure_context = {
            "pipeline_id": "test_pipeline",
            "failed_task_id": "task3",
            "error": "Network timeout",
            "attempt_count": 1,
        }

        recovery_result = await manager.recover_from_failure(failure_context)

        assert recovery_result["strategy"] in ["retry", "skip", "alternative"]
        if recovery_result["strategy"] == "retry":
            assert recovery_result["retry_delay"] > 0

    @pytest.mark.asyncio
    async def test_recover_partial_pipeline(self):
        """Test recovering partially completed pipeline."""
        manager = RecoveryManager()

        pipeline_state = {
            "pipeline_id": "test_pipeline",
            "total_tasks": 5,
            "completed_tasks": ["task1", "task2", "task3"],
            "failed_tasks": ["task4"],
            "pending_tasks": ["task5"],
        }

        recovery_result = await manager.recover_partial_pipeline(pipeline_state)

        assert recovery_result["success"] is True
        assert recovery_result["recovery_point"] == "task4"
        assert recovery_result["remaining_tasks"] == ["task4", "task5"]

    @pytest.mark.asyncio
    async def test_rollback_transaction(self):
        """Test transaction rollback functionality."""
        manager = RecoveryManager()

        transaction_data = {
            "transaction_id": "txn_123",
            "operations": [
                {"type": "create", "resource": "task1", "data": {"name": "Task 1"}},
                {
                    "type": "update",
                    "resource": "task2",
                    "data": {"status": "completed"},
                },
                {"type": "delete", "resource": "task3"},
            ],
        }

        rollback_result = await manager.rollback_transaction(transaction_data)

        assert rollback_result["success"] is True
        assert rollback_result["operations_rolled_back"] == 3

    def test_analyze_failure_pattern(self):
        """Test failure pattern analysis."""
        manager = RecoveryManager()

        # Add failure history
        failures = [
            {"error": "Network timeout", "timestamp": "2023-01-01T00:00:00Z"},
            {"error": "Network timeout", "timestamp": "2023-01-01T00:05:00Z"},
            {"error": "Network timeout", "timestamp": "2023-01-01T00:10:00Z"},
            {"error": "Memory error", "timestamp": "2023-01-01T00:15:00Z"},
        ]

        for failure in failures:
            manager.record_failure(failure)

        pattern = manager.analyze_failure_pattern()

        assert pattern["dominant_error"] == "Network timeout"
        assert pattern["error_frequency"]["Network timeout"] == 3
        assert pattern["error_frequency"]["Memory error"] == 1
        assert pattern["recommendation"] in ["check_network", "increase_timeout"]

    @pytest.mark.asyncio
    async def test_automatic_recovery(self):
        """Test automatic recovery functionality."""
        manager = RecoveryManager()

        # Enable automatic recovery
        manager.enable_automatic_recovery()

        # Simulate failure
        failure_data = {
            "pipeline_id": "test_pipeline",
            "error": "Temporary network error",
            "severity": "medium",
            "recoverable": True,
        }

        recovery_result = await manager.attempt_automatic_recovery(failure_data)

        assert recovery_result["attempted"] is True
        assert recovery_result["strategy"] in [
            "retry",
            "alternative_path",
            "resource_reallocation",
        ]

    def test_recovery_strategy_selection(self):
        """Test recovery strategy selection."""
        manager = RecoveryManager()

        # Test different error types
        network_error = {"error": "Network timeout", "retryable": True}
        memory_error = {"error": "Out of memory", "retryable": False}
        validation_error = {"error": "Invalid input", "retryable": False}

        assert manager.select_recovery_strategy(network_error) == "retry_with_backoff"
        assert manager.select_recovery_strategy(memory_error) == "resource_reallocation"
        assert (
            manager.select_recovery_strategy(validation_error) == "input_sanitization"
        )

    def test_recovery_metrics(self):
        """Test recovery metrics collection."""
        manager = RecoveryManager()

        # Simulate recovery attempts
        manager.record_recovery_attempt("retry", success=True, duration=1.5)
        manager.record_recovery_attempt("retry", success=False, duration=2.0)
        manager.record_recovery_attempt("rollback", success=True, duration=0.5)

        metrics = manager.get_recovery_metrics()

        assert metrics["total_attempts"] == 3
        assert metrics["success_rate"] == 2 / 3
        assert metrics["average_duration"] == (1.5 + 2.0 + 0.5) / 3
        assert metrics["strategy_success_rates"]["retry"] == 0.5
        assert metrics["strategy_success_rates"]["rollback"] == 1.0


class TestRetryStrategyAdvanced:
    """Advanced test cases for RetryStrategy."""

    def test_should_retry_with_error_info_object(self):
        """Test should_retry with ErrorInfo object."""
        from src.orchestrator.core.error_handler import (
            ErrorCategory,
            ErrorInfo,
            ErrorSeverity,
            RetryStrategy)

        strategy = RetryStrategy(max_retries=3)

        # Test with recoverable error
        recoverable_error = ErrorInfo(
            error_type="TestError",
            message="Test message",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            recoverable=True)

        assert strategy.should_retry(recoverable_error, 1) is True
        assert strategy.should_retry(recoverable_error, 3) is False

        # Test with non-recoverable error
        non_recoverable_error = ErrorInfo(
            error_type="TestError",
            message="Test message",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            recoverable=False)

        assert strategy.should_retry(non_recoverable_error, 1) is False

    def test_should_retry_with_attempt_only(self):
        """Test should_retry with attempt number only."""
        from src.orchestrator.core.error_handler import RetryStrategy

        strategy = RetryStrategy(max_retries=3)

        # When called with just attempt number
        assert strategy.should_retry(1) is True
        assert strategy.should_retry(3) is False
        assert strategy.should_retry(5) is False

    def test_should_retry_error_method(self):
        """Test should_retry_error method."""
        from src.orchestrator.core.error_handler import RetryStrategy

        # Test with no specific retryable errors (should retry all)
        strategy = RetryStrategy(max_retries=3)
        assert strategy.should_retry_error(Exception("test")) is True
        assert strategy.should_retry_error(ValueError("test")) is True

        # Test with specific retryable errors
        strategy_specific = RetryStrategy(
            max_retries=3, retryable_errors=[ValueError, TypeError]
        )
        assert strategy_specific.should_retry_error(ValueError("test")) is True
        assert strategy_specific.should_retry_error(TypeError("test")) is True
        assert strategy_specific.should_retry_error(RuntimeError("test")) is False


class TestExponentialBackoffRetry:
    """Test cases for ExponentialBackoffRetry class."""

    def test_should_retry_validation_errors(self):
        """Test should_retry with validation errors."""
        from src.orchestrator.core.error_handler import (
            ErrorCategory,
            ErrorInfo,
            ErrorSeverity,
            ExponentialBackoffRetry)

        strategy = ExponentialBackoffRetry(max_retries=3)

        validation_error = ErrorInfo(
            error_type="ValidationError",
            message="Invalid input",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            recoverable=True)

        # Validation errors should not be retried even if recoverable
        assert strategy.should_retry(validation_error, 1) is False

    def test_should_retry_permission_errors(self):
        """Test should_retry with permission errors."""
        from src.orchestrator.core.error_handler import (
            ErrorCategory,
            ErrorInfo,
            ErrorSeverity,
            ExponentialBackoffRetry)

        strategy = ExponentialBackoffRetry(max_retries=3)

        permission_error = ErrorInfo(
            error_type="PermissionError",
            message="Access denied",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.PERMISSION,
            recoverable=True)

        # Permission errors should not be retried
        assert strategy.should_retry(permission_error, 1) is False

    def test_should_retry_configuration_errors(self):
        """Test should_retry with configuration errors."""
        from src.orchestrator.core.error_handler import (
            ErrorCategory,
            ErrorInfo,
            ErrorSeverity,
            ExponentialBackoffRetry)

        strategy = ExponentialBackoffRetry(max_retries=3)

        config_error = ErrorInfo(
            error_type="ConfigError",
            message="Invalid configuration",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION,
            recoverable=True)

        # Configuration errors should not be retried
        assert strategy.should_retry(config_error, 1) is False

    def test_should_retry_max_retries_exceeded(self):
        """Test should_retry when max retries exceeded."""
        from src.orchestrator.core.error_handler import (
            ErrorCategory,
            ErrorInfo,
            ErrorSeverity,
            ExponentialBackoffRetry)

        strategy = ExponentialBackoffRetry(max_retries=2)

        error = ErrorInfo(
            error_type="NetworkError",
            message="Connection failed",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            recoverable=True)

        # Should not retry when max retries exceeded
        assert strategy.should_retry(error, 2) is False
        assert strategy.should_retry(error, 3) is False

    def test_should_retry_critical_non_recoverable(self):
        """Test should_retry with critical non-recoverable errors."""
        from src.orchestrator.core.error_handler import (
            ErrorCategory,
            ErrorInfo,
            ErrorSeverity,
            RetryStrategy)

        strategy = RetryStrategy(max_retries=3)

        critical_error = ErrorInfo(
            error_type="CriticalError",
            message="System failure",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SYSTEM,
            recoverable=False)

        # Critical non-recoverable errors should not be retried
        assert strategy.should_retry(critical_error, 1) is False

    def test_should_retry_critical_recoverable(self):
        """Test should_retry with critical recoverable errors."""
        from src.orchestrator.core.error_handler import (
            ErrorCategory,
            ErrorInfo,
            ErrorSeverity,
            RetryStrategy)

        strategy = RetryStrategy(max_retries=3)

        critical_recoverable_error = ErrorInfo(
            error_type="CriticalError",
            message="System failure",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SYSTEM,
            recoverable=True)

        # Critical recoverable errors should be retried
        assert strategy.should_retry(critical_recoverable_error, 1) is True
        assert strategy.should_retry(critical_recoverable_error, 3) is False


class TestErrorHandlerAdvanced:
    """Advanced test cases for ErrorHandler."""

    @pytest.mark.asyncio
    async def test_error_handler_circuit_breaker_integration(self):
        """Test error handler integration with circuit breaker."""
        from src.orchestrator.core.error_handler import ErrorHandler

        handler = ErrorHandler()

        # Test that circuit breaker is used
        assert handler.circuit_breaker is not None

        # Simulate multiple failures to trigger circuit breaker
        error = Exception("Test error")
        context = {"retry_count": 5}  # High retry count

        result = await handler.handle_error(error, context)

        # Should have some recovery action
        assert "action" in result

    @pytest.mark.asyncio
    async def test_error_handler_fallback_strategies(self):
        """Test error handler fallback strategies."""
        from src.orchestrator.core.error_handler import ErrorHandler

        handler = ErrorHandler()

        # Test with unknown error category
        unknown_error = Exception("Unknown error type")
        context = {"error_category": "unknown", "retry_count": 0}

        result = await handler.handle_error(unknown_error, context)

        # Should have fallback handling
        assert "action" in result
        assert result["action"] in ["retry", "fail", "switch_system"]
