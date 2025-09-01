"""
Comprehensive test suite for quality control logging framework.

This test package provides complete coverage for the orchestrator quality
control logging system including:

- Core structured logging functionality
- Specialized log handlers and formatters  
- External monitoring system integration
- Quality event tracking and alerting
- Integration with validation system components
- Performance and thread safety testing

Test Modules:
- test_logger.py: Tests for core StructuredLogger and related classes
- test_handlers.py: Tests for specialized handlers and formatters
- test_monitoring.py: Tests for external monitoring integration
- test_integration.py: Tests for validation system integration

Usage:
    Run all logging tests:
    pytest tests/orchestrator/quality/logging/
    
    Run specific test module:
    pytest tests/orchestrator/quality/logging/test_logger.py
    
    Run with coverage:
    pytest --cov=src/orchestrator/quality/logging tests/orchestrator/quality/logging/
"""

# Test fixtures and utilities that can be shared across test modules
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from src.orchestrator.quality.logging.logger import StructuredLogger, LogLevel, LogCategory, QualityEvent
from src.orchestrator.quality.validation.validator import ValidationResult, ValidationSeverity
from src.orchestrator.quality.validation.rules import RuleViolation, RuleSeverity, ValidationContext
from src.orchestrator.execution.state import ExecutionContext, ExecutionStatus


def create_mock_logger(name: str = "test_logger") -> Mock:
    """Create a mock StructuredLogger for testing."""
    mock_logger = Mock(spec=StructuredLogger)
    mock_logger.name = name
    mock_logger.level = LogLevel.INFO
    mock_logger.context = Mock()
    mock_logger.operation_timer = Mock()
    return mock_logger


def create_test_quality_event(
    event_type: str = "test_event",
    severity: str = "INFO",
    quality_score: float = 0.85
) -> QualityEvent:
    """Create a QualityEvent for testing."""
    return QualityEvent(
        event_type=event_type,
        severity=severity,
        source_component="test_component",
        quality_score=quality_score,
        rule_violations=[{"rule": "test_rule", "message": "Test violation"}],
        recommendations=["Fix the issue"],
        remediation_actions=["Update configuration"]
    )


def create_test_validation_result(
    session_id: str = "test_session",
    severity: ValidationSeverity = ValidationSeverity.PASS,
    quality_score: float = 0.95
) -> ValidationResult:
    """Create a ValidationResult for testing."""
    return ValidationResult(
        session_id=session_id,
        pipeline_id="pipe-123",
        execution_id="exec-456",
        step_id="step-789",
        output_path="/test/output",
        severity=severity,
        quality_score=quality_score,
        violations=[],
        performance_metrics={"duration_ms": 100.0},
        recommendations=[],
        execution_time=0.1
    )


def create_test_rule_violation(
    rule_name: str = "test_rule",
    severity: RuleSeverity = RuleSeverity.WARNING
) -> RuleViolation:
    """Create a RuleViolation for testing."""
    return RuleViolation(
        rule_name=rule_name,
        severity=severity,
        message="Test violation message",
        context={"file": "test.txt", "line": 42}
    )


def create_test_execution_context(
    execution_id: str = "exec-123",
    pipeline_id: str = "pipe-456"
) -> Mock:
    """Create a mock ExecutionContext for testing."""
    context = Mock(spec=ExecutionContext)
    context.execution_id = execution_id
    context.pipeline_id = pipeline_id
    context.status = ExecutionStatus.RUNNING
    return context


def create_test_validation_context(
    execution_id: str = "exec-123",
    pipeline_id: str = "pipe-456",
    output_path: str = "/test/output"
) -> ValidationContext:
    """Create a ValidationContext for testing."""
    return ValidationContext(
        execution_id=execution_id,
        pipeline_id=pipeline_id,
        output_path=output_path,
        metadata={"test": "metadata"}
    )


class TestLogCapture:
    """Utility class for capturing log output in tests."""
    
    def __init__(self):
        self.logs = []
    
    def emit(self, record):
        """Capture log record."""
        self.logs.append(record)
    
    def get_logs(self, level=None):
        """Get captured logs, optionally filtered by level."""
        if level is None:
            return self.logs
        return [log for log in self.logs if log.levelno == level]
    
    def clear(self):
        """Clear captured logs."""
        self.logs.clear()
    
    def has_message(self, message_part: str) -> bool:
        """Check if any log contains the specified message part."""
        return any(message_part in str(log.getMessage()) for log in self.logs)


class TempDirectoryManager:
    """Context manager for temporary directories in tests."""
    
    def __init__(self):
        self.temp_dir = None
    
    def __enter__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        return self.temp_dir
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)


def wait_for_async_operations(timeout: float = 1.0):
    """Wait for async operations to complete in tests."""
    import asyncio
    loop = asyncio.get_event_loop()
    if loop.is_running():
        time.sleep(timeout)
    else:
        loop.run_until_complete(asyncio.sleep(timeout))


# Test configuration constants
TEST_LOG_LEVEL = LogLevel.DEBUG
TEST_BUFFER_SIZE = 10
TEST_FLUSH_INTERVAL = 0.1
TEST_TIMEOUT = 5.0

# Mock data for consistent testing
MOCK_PROMETHEUS_METRICS = {
    "orchestrator_quality_score": {
        "value": 0.85,
        "labels": {"component": "validation", "pipeline": "test"},
        "help": "Quality score for validation"
    },
    "orchestrator_violations_total": {
        "value": 3,
        "labels": {"severity": "warning"},
        "help": "Total number of violations"
    }
}

MOCK_WEBHOOK_PAYLOAD = {
    "type": "alert",
    "timestamp": "2023-01-01T00:00:00Z",
    "source": "orchestrator_quality",
    "data": {
        "rule_name": "quality_threshold",
        "severity": "WARNING",
        "message": "Quality score below threshold"
    }
}

MOCK_ELASTICSEARCH_DOCUMENT = {
    "@timestamp": "2023-01-01T00:00:00Z",
    "type": "metrics",
    "source": "orchestrator_quality",
    "metrics": {
        "quality_score": 0.75,
        "violations": 2,
        "execution_time": 1.5
    }
}


# Export commonly used test utilities
__all__ = [
    "create_mock_logger",
    "create_test_quality_event", 
    "create_test_validation_result",
    "create_test_rule_violation",
    "create_test_execution_context",
    "create_test_validation_context",
    "TestLogCapture",
    "TempDirectoryManager",
    "wait_for_async_operations",
    "TEST_LOG_LEVEL",
    "TEST_BUFFER_SIZE", 
    "TEST_FLUSH_INTERVAL",
    "TEST_TIMEOUT",
    "MOCK_PROMETHEUS_METRICS",
    "MOCK_WEBHOOK_PAYLOAD",
    "MOCK_ELASTICSEARCH_DOCUMENT"
]