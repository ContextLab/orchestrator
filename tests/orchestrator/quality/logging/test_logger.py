"""
Comprehensive tests for the quality control logging framework.

This test suite validates all aspects of the structured logging system
including context management, performance tracking, quality event logging,
and integration with external monitoring systems.
"""

import asyncio
import json
import logging
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import pytest

from src.orchestrator.quality.logging.logger import (
    StructuredLogger,
    LogLevel,
    LogCategory,
    LogContext,
    QualityEvent,
    get_logger,
    flush_all_loggers,
    get_all_quality_metrics
)


class TestLogContext:
    """Test LogContext dataclass functionality."""
    
    def test_log_context_creation(self):
        """Test creating LogContext with all fields."""
        context = LogContext(
            timestamp="2023-01-01T00:00:00Z",
            level="INFO",
            category=LogCategory.VALIDATION,
            component="test_component",
            operation="test_operation",
            execution_id="exec-123",
            pipeline_id="pipe-456",
            session_id="sess-789",
            quality_score=0.95,
            rule_violations=2,
            tags=["test", "validation"]
        )
        
        assert context.timestamp == "2023-01-01T00:00:00Z"
        assert context.level == "INFO"
        assert context.category == LogCategory.VALIDATION
        assert context.component == "test_component"
        assert context.execution_id == "exec-123"
        assert context.quality_score == 0.95

    def test_log_context_to_dict(self):
        """Test converting LogContext to dictionary."""
        context = LogContext(
            timestamp="2023-01-01T00:00:00Z",
            level="INFO",
            category=LogCategory.VALIDATION,
            component="test_component",
            operation="test_operation",
            metadata={"key": "value"}
        )
        
        result = context.to_dict()
        
        assert result["timestamp"] == "2023-01-01T00:00:00Z"
        assert result["category"] == "validation"
        assert result["metadata"] == {"key": "value"}
        # None values should be excluded
        assert "execution_id" not in result

    def test_log_context_removes_none_values(self):
        """Test that None values are removed from dictionary representation."""
        context = LogContext(
            timestamp="2023-01-01T00:00:00Z",
            level="INFO",
            category=LogCategory.EXECUTION,
            component="test",
            operation="test",
            execution_id=None,
            pipeline_id=None
        )
        
        result = context.to_dict()
        
        assert "execution_id" not in result
        assert "pipeline_id" not in result
        assert len(result) == 5  # Only non-None fields


class TestQualityEvent:
    """Test QualityEvent dataclass functionality."""
    
    def test_quality_event_creation(self):
        """Test creating QualityEvent with all fields."""
        event = QualityEvent(
            event_type="validation_completed",
            severity="WARNING",
            source_component="validator",
            quality_score=0.75,
            rule_violations=[{"rule": "test", "message": "violation"}],
            recommendations=["Fix validation issue"],
            remediation_actions=["Update code"]
        )
        
        assert event.event_type == "validation_completed"
        assert event.severity == "WARNING"
        assert event.quality_score == 0.75
        assert len(event.rule_violations) == 1
        assert event.recommendations[0] == "Fix validation issue"


class TestStructuredLogger:
    """Test StructuredLogger functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = StructuredLogger(
            "test_logger",
            level=LogLevel.DEBUG,
            enable_structured=True,
            buffer_size=0  # Disable buffering for tests
        )

    def test_logger_creation(self):
        """Test creating a structured logger."""
        assert self.logger.name == "test_logger"
        assert self.logger.level == LogLevel.DEBUG
        assert self.logger.enable_structured is True

    def test_logger_context_manager(self):
        """Test context manager functionality."""
        with self.logger.context(execution_id="exec-123", pipeline_id="pipe-456"):
            context = self.logger._get_current_context()
            assert context["execution_id"] == "exec-123"
            assert context["pipeline_id"] == "pipe-456"
        
        # Context should be cleaned up after exiting
        context = self.logger._get_current_context()
        assert "execution_id" not in context
        assert "pipeline_id" not in context

    def test_nested_context_managers(self):
        """Test nested context managers."""
        with self.logger.context(execution_id="exec-123"):
            with self.logger.context(step_id="step-789"):
                context = self.logger._get_current_context()
                assert context["execution_id"] == "exec-123"
                assert context["step_id"] == "step-789"
            
            # Inner context should be cleaned up
            context = self.logger._get_current_context()
            assert context["execution_id"] == "exec-123"
            assert "step_id" not in context

    def test_operation_timer(self):
        """Test operation timing context manager."""
        with patch.object(self.logger, '_emit_log') as mock_emit:
            with self.logger.operation_timer("test_operation"):
                time.sleep(0.01)  # Small delay for timing
        
        # Verify debug log was emitted with duration
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][0] == LogLevel.DEBUG
        assert "duration_ms" in call_args[0][1].to_dict()
        assert call_args[0][1].to_dict()["duration_ms"] > 0

    def test_should_log_level_filtering(self):
        """Test log level filtering."""
        # Set logger to INFO level
        self.logger.level = LogLevel.INFO
        
        assert self.logger._should_log(LogLevel.DEBUG) is False
        assert self.logger._should_log(LogLevel.INFO) is True
        assert self.logger._should_log(LogLevel.WARNING) is True
        assert self.logger._should_log(LogLevel.ERROR) is True

    def test_validation_context_setting(self):
        """Test setting validation context."""
        validation_context = {"session_id": "sess-123", "pipeline_id": "pipe-456"}
        self.logger.set_validation_context(validation_context)
        
        assert self.logger._validation_context == validation_context

    def test_quality_event_logging(self):
        """Test logging quality events."""
        event = QualityEvent(
            event_type="test_event",
            severity="INFO",
            source_component="test_component",
            quality_score=0.85
        )
        
        with patch.object(self.logger, '_emit_log') as mock_emit:
            self.logger.log_quality_event(event)
        
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][0] == LogLevel.INFO
        assert "quality_event_test_event" in call_args[0][1].operation

    def test_validation_result_logging(self):
        """Test logging validation results."""
        validation_result = {
            "severity": "PASS",
            "quality_score": 0.95,
            "violations": [],
            "output_path": "/test/path"
        }
        
        with patch.object(self.logger, '_emit_log') as mock_emit:
            self.logger.log_validation_result(validation_result)
        
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][0] == LogLevel.INFO
        assert call_args[0][1].quality_level == "PASS"
        assert call_args[0][1].validation_score == 0.95

    def test_standard_logging_methods(self):
        """Test standard logging methods (info, warning, error, etc.)."""
        with patch.object(self.logger, '_emit_log') as mock_emit:
            self.logger.info("Test info message")
            self.logger.warning("Test warning message")
            self.logger.error("Test error message")
            self.logger.debug("Test debug message")
            self.logger.critical("Test critical message")
        
        assert mock_emit.call_count == 5
        
        # Check that correct levels were used
        calls = mock_emit.call_args_list
        assert calls[0][0][0] == LogLevel.INFO
        assert calls[1][0][0] == LogLevel.WARNING
        assert calls[2][0][0] == LogLevel.ERROR
        assert calls[3][0][0] == LogLevel.DEBUG
        assert calls[4][0][0] == LogLevel.CRITICAL

    def test_error_logging_with_exception(self):
        """Test logging errors with exception details."""
        test_exception = ValueError("Test exception")
        
        with patch.object(self.logger, '_emit_log') as mock_emit:
            self.logger.error("Test error", exception=test_exception)
        
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        context = call_args[0][1]
        
        assert context.error_details is not None
        assert context.error_details["exception_type"] == "ValueError"
        assert context.error_details["exception_message"] == "Test exception"
        assert "traceback" in context.error_details

    def test_audit_logging(self):
        """Test audit trail logging."""
        with patch.object(self.logger, '_emit_log') as mock_emit:
            self.logger.audit("User performed action", user_action="file_upload")
        
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][0] == LogLevel.AUDIT
        assert call_args[0][1].category == LogCategory.AUDIT

    def test_quality_metrics_collection(self):
        """Test quality metrics collection."""
        # Log some quality events
        event1 = QualityEvent("event1", "INFO", "test", quality_score=0.8)
        event2 = QualityEvent("event2", "WARNING", "test", quality_score=0.6)
        
        with patch.object(self.logger, '_emit_log'):
            self.logger.log_quality_event(event1)
            self.logger.log_quality_event(event2)
        
        metrics = self.logger.get_quality_metrics()
        
        assert metrics["total_quality_events"] == 2
        assert metrics["quality_events_by_type"]["event1"] == 1
        assert metrics["quality_events_by_type"]["event2"] == 1
        assert metrics["average_quality_score"] == 0.7

    def test_flush_functionality(self):
        """Test manual flushing of logs."""
        # This test mainly ensures flush doesn't crash
        self.logger.flush()
        assert True  # If we get here, flush worked


class TestLoggerBuffering:
    """Test logger buffering functionality."""
    
    def test_buffered_logging(self):
        """Test buffered logging functionality."""
        logger = StructuredLogger(
            "buffered_logger",
            buffer_size=10,
            flush_interval=1.0
        )
        
        with patch.object(logger._logger, 'log') as mock_log:
            # Log messages that should be buffered
            for i in range(5):
                logger.info(f"Message {i}")
            
            # Should not have logged yet (buffered)
            mock_log.assert_not_called()
            
            # Force flush
            logger.flush()
            
            # Now should have logged all messages
            assert mock_log.call_count == 5

    def test_buffer_overflow_handling(self):
        """Test buffer overflow handling."""
        logger = StructuredLogger(
            "overflow_logger",
            buffer_size=2,
            flush_interval=10.0  # Long interval to prevent automatic flushing
        )
        
        with patch.object(logger._logger, 'log') as mock_log:
            # Fill buffer to capacity
            logger.info("Message 1")
            logger.info("Message 2")
            
            # This should trigger flush
            logger.info("Message 3")
            
            # Should have flushed when buffer hit capacity
            assert mock_log.call_count >= 2


class TestLoggerRegistry:
    """Test global logger registry functionality."""
    
    def test_get_logger_creates_new(self):
        """Test that get_logger creates new loggers."""
        logger1 = get_logger("test_registry_1")
        logger2 = get_logger("test_registry_2")
        
        assert logger1.name == "test_registry_1"
        assert logger2.name == "test_registry_2"
        assert logger1 is not logger2

    def test_get_logger_reuses_existing(self):
        """Test that get_logger reuses existing loggers."""
        logger1 = get_logger("test_registry_reuse")
        logger2 = get_logger("test_registry_reuse")
        
        assert logger1 is logger2

    def test_flush_all_loggers(self):
        """Test flushing all registered loggers."""
        logger1 = get_logger("flush_test_1")
        logger2 = get_logger("flush_test_2")
        
        with patch.object(logger1, 'flush') as mock_flush1, \
             patch.object(logger2, 'flush') as mock_flush2:
            flush_all_loggers()
            
            mock_flush1.assert_called_once()
            mock_flush2.assert_called_once()

    def test_get_all_quality_metrics(self):
        """Test getting quality metrics from all loggers."""
        logger1 = get_logger("metrics_test_1")
        logger2 = get_logger("metrics_test_2")
        
        # Mock quality metrics
        with patch.object(logger1, 'get_quality_metrics', return_value={"total": 5}), \
             patch.object(logger2, 'get_quality_metrics', return_value={"total": 3}):
            
            all_metrics = get_all_quality_metrics()
            
            assert "metrics_test_1" in all_metrics
            assert "metrics_test_2" in all_metrics
            assert all_metrics["metrics_test_1"]["total"] == 5
            assert all_metrics["metrics_test_2"]["total"] == 3


class TestLoggerThreadSafety:
    """Test thread safety of logging operations."""
    
    def test_concurrent_logging(self):
        """Test concurrent logging from multiple threads."""
        logger = get_logger("thread_safety_test")
        messages_logged = []
        
        def log_messages(thread_id):
            for i in range(10):
                with patch.object(logger._logger, 'log') as mock_log:
                    logger.info(f"Thread {thread_id} message {i}")
                    messages_logged.append(f"Thread {thread_id} message {i}")
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_messages, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have logged messages from all threads
        assert len(messages_logged) == 50

    def test_concurrent_context_management(self):
        """Test concurrent context management."""
        logger = get_logger("context_thread_test")
        results = {}
        
        def set_context_and_check(thread_id):
            execution_id = f"exec-{thread_id}"
            with logger.context(execution_id=execution_id):
                context = logger._get_current_context()
                results[thread_id] = context.get("execution_id")
        
        threads = []
        for i in range(10):
            thread = threading.Thread(target=set_context_and_check, args=(i,))
            threads.append(thread)
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Each thread should have had its own context
        for i in range(10):
            assert results[i] == f"exec-{i}"


class TestLoggerPerformance:
    """Test logger performance characteristics."""
    
    def test_logging_performance(self):
        """Test that logging operations complete within reasonable time."""
        logger = get_logger("performance_test", buffer_size=0)
        
        start_time = time.time()
        
        with patch.object(logger._logger, 'log'):
            for i in range(1000):
                logger.info(f"Performance test message {i}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete 1000 log operations in reasonable time (< 1 second)
        assert duration < 1.0

    def test_context_overhead(self):
        """Test that context management doesn't add significant overhead."""
        logger = get_logger("context_performance_test")
        
        # Test without context
        start_time = time.time()
        with patch.object(logger, '_emit_log'):
            for i in range(100):
                logger.info("No context message")
        no_context_time = time.time() - start_time
        
        # Test with context
        start_time = time.time()
        with patch.object(logger, '_emit_log'):
            with logger.context(execution_id="exec-123", pipeline_id="pipe-456"):
                for i in range(100):
                    logger.info("Context message")
        context_time = time.time() - start_time
        
        # Context overhead should be minimal (less than 50% increase)
        assert context_time < no_context_time * 1.5


class TestLoggerIntegration:
    """Test integration with standard logging system."""
    
    def test_standard_logger_integration(self):
        """Test that structured logger integrates with standard logging."""
        logger = get_logger("integration_test")
        
        # The underlying logger should be a standard Python logger
        assert isinstance(logger._logger, logging.Logger)
        assert logger._logger.name == "orchestrator.quality.integration_test"

    def test_log_level_mapping(self):
        """Test that custom log levels map correctly to standard levels."""
        logger = StructuredLogger("level_test", LogLevel.WARNING)
        
        assert logger._logger.level == LogLevel.WARNING.value
        assert logger._should_log(LogLevel.ERROR) is True
        assert logger._should_log(LogLevel.DEBUG) is False


if __name__ == "__main__":
    pytest.main([__file__])