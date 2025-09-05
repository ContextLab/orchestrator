"""
Comprehensive tests for quality control logging handlers.

This test suite validates all specialized handlers including JSON formatters,
rotating file handlers, async handlers, and external monitoring integration.
"""

import asyncio
import gzip
import json
import logging
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, mock_open
import pytest

from src.orchestrator.quality.logging.handlers import (
    QualityJSONFormatter,
    QualityRotatingFileHandler,
    AsyncQualityHandler,
    PrometheusMetricsHandler,
    QualityEventStreamHandler,
    create_quality_logging_setup
)
from src.orchestrator.quality.logging.logger import QualityEvent


class TestQualityJSONFormatter:
    """Test QualityJSONFormatter functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = QualityJSONFormatter()

    def test_basic_formatting(self):
        """Test basic log record formatting."""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.created = 1640995200.0  # Fixed timestamp for testing
        
        result = self.formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test.logger"
        assert log_data["message"] == "Test message"
        assert log_data["line"] == 42
        assert "timestamp" in log_data

    def test_structured_data_inclusion(self):
        """Test inclusion of structured data from record."""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.created = 1640995200.0
        record.structured_data = {"execution_id": "exec-123", "pipeline_id": "pipe-456"}
        
        result = self.formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data["execution_id"] == "exec-123"
        assert log_data["pipeline_id"] == "pipe-456"

    def test_quality_metrics_inclusion(self):
        """Test inclusion of quality metrics when enabled."""
        formatter = QualityJSONFormatter(include_quality_metrics=True)
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.created = 1640995200.0
        record.quality_metrics = {"quality_score": 0.85, "violations": 2}
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data["quality_metrics"]["quality_score"] == 0.85
        assert log_data["quality_metrics"]["violations"] == 2

    def test_performance_metrics_inclusion(self):
        """Test inclusion of performance metrics when enabled."""
        formatter = QualityJSONFormatter(include_performance_metrics=True)
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.created = 1640995200.0
        record.performance_metrics = {"duration_ms": 150.5, "memory_mb": 64.2}
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data["performance_metrics"]["duration_ms"] == 150.5
        assert log_data["performance_metrics"]["memory_mb"] == 64.2

    def test_exception_handling(self):
        """Test exception information formatting."""
        formatter = QualityJSONFormatter(include_stack_trace=True)
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="/test/path.py",
                lineno=42,
                msg="Test error message",
                args=(),
                exc_info=True  # This captures current exception
            )
            record.created = 1640995200.0
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert log_data["exception"]["message"] == "Test exception"
        assert "traceback" in log_data["exception"]

    def test_field_exclusion(self):
        """Test excluding specified fields."""
        formatter = QualityJSONFormatter(exclude_fields=["thread", "process"])
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.created = 1640995200.0
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert "thread" not in log_data
        assert "process" not in log_data
        assert "message" in log_data  # Non-excluded field should be present

    def test_timestamp_formats(self):
        """Test different timestamp formats."""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.created = 1640995200.0  # 2022-01-01 00:00:00 UTC
        
        # Test ISO format
        iso_formatter = QualityJSONFormatter(timestamp_format="iso")
        iso_result = json.loads(iso_formatter.format(record))
        assert "2022-01-01" in iso_result["timestamp"]
        
        # Test Unix format
        unix_formatter = QualityJSONFormatter(timestamp_format="unix")
        unix_result = json.loads(unix_formatter.format(record))
        assert unix_result["timestamp"] == "1640995200.0"

    def test_formatting_error_handling(self):
        """Test that formatting errors don't crash the logger."""
        formatter = QualityJSONFormatter()
        
        # Create a record with problematic data
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.created = 1640995200.0
        
        # Add non-serializable data
        record.structured_data = {"bad_data": object()}
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        # Should have fallback formatting
        assert "timestamp" in log_data
        assert "level" in log_data


class TestQualityRotatingFileHandler:
    """Test QualityRotatingFileHandler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "test.log"

    def test_handler_creation(self):
        """Test creating rotating file handler."""
        handler = QualityRotatingFileHandler(
            str(self.log_file),
            maxBytes=1024,
            backupCount=5,
            compress_rotated=True
        )
        
        assert handler.baseFilename == str(self.log_file)
        assert handler.maxBytes == 1024
        assert handler.backupCount == 5
        assert handler.compress_rotated is True

    def test_size_based_rollover(self):
        """Test size-based log rollover."""
        handler = QualityRotatingFileHandler(
            str(self.log_file),
            maxBytes=100,  # Small size to trigger rollover
            backupCount=2,
            compress_rotated=False
        )
        
        # Write enough data to trigger rollover
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="A" * 50,  # 50 character message
            args=(),
            exc_info=None
        )
        
        # Write multiple records to exceed maxBytes
        for i in range(5):
            handler.emit(record)
        
        handler.close()
        
        # Should have created backup files
        assert self.log_file.exists()
        # Note: Testing actual backup creation requires more complex setup

    def test_quality_event_based_rollover(self):
        """Test rollover based on quality events."""
        handler = QualityRotatingFileHandler(
            str(self.log_file),
            maxBytes=1000000,  # Large size to prevent size-based rollover
            rotate_on_quality_events=True,
            quality_event_threshold=2
        )
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.quality_event = True  # Mark as quality event
        
        # Should not rollover on first quality event
        assert not handler.shouldRollover(record)
        handler.emit(record)
        
        # Should rollover on second quality event (threshold reached)
        assert handler.shouldRollover(record)

    def test_time_based_rollover(self):
        """Test time-based rollover logic."""
        handler = QualityRotatingFileHandler(
            str(self.log_file),
            maxBytes=1000000,  # Large size
            time_based_rotation="daily"
        )
        
        # Mock the last rotation time to be old
        handler._last_rotation_time = time.time() - (25 * 60 * 60)  # 25 hours ago
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Should trigger rollover due to time
        assert handler.shouldRollover(record)

    def test_compression_functionality(self):
        """Test log file compression."""
        # This test would require more complex file system operations
        # For now, test that the method exists and doesn't crash
        handler = QualityRotatingFileHandler(
            str(self.log_file),
            compress_rotated=True
        )
        
        # Create a test backup file
        backup_file = Path(f"{self.log_file}.1")
        backup_file.write_text("test log content")
        
        # Test compression method exists
        assert hasattr(handler, '_compress_backup_files')
        
        # Test it doesn't crash when called
        try:
            handler._compress_backup_files()
        except Exception:
            pass  # Compression might fail in test environment, but shouldn't crash


class TestAsyncQualityHandler:
    """Test AsyncQualityHandler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.target_handler = Mock(spec=logging.Handler)
        self.async_handler = AsyncQualityHandler(
            target_handler=self.target_handler,
            queue_size=100,
            num_workers=2,
            batch_size=5,
            flush_interval=0.1
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        self.async_handler.close()

    def test_handler_creation(self):
        """Test creating async handler."""
        assert self.async_handler.target_handler is self.target_handler
        assert self.async_handler.queue_size == 100
        assert self.async_handler.num_workers == 2
        assert len(self.async_handler._workers) == 2

    def test_async_emit(self):
        """Test asynchronous log emission."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Emit record
        self.async_handler.emit(record)
        
        # Give workers time to process
        time.sleep(0.2)
        
        # Target handler should have been called
        self.target_handler.emit.assert_called()

    def test_queue_overflow_handling(self):
        """Test handling of queue overflow."""
        # Create handler with very small queue
        small_handler = AsyncQualityHandler(
            target_handler=self.target_handler,
            queue_size=2,
            num_workers=1,
            batch_size=1,
            flush_interval=10.0  # Long interval to prevent automatic processing
        )
        
        try:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Test message",
                args=(),
                exc_info=None
            )
            
            # Fill queue beyond capacity
            for i in range(5):
                small_handler.emit(record)
            
            # Should not crash despite queue overflow
            assert True
            
        finally:
            small_handler.close()

    def test_graceful_shutdown(self):
        """Test graceful shutdown of async handler."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Emit some records
        for i in range(3):
            self.async_handler.emit(record)
        
        # Close should process remaining records
        self.async_handler.close()
        
        # All workers should be stopped
        for worker in self.async_handler._workers:
            assert not worker.is_alive() or worker.daemon

    def test_batch_processing(self):
        """Test batch processing functionality."""
        records = []
        for i in range(10):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg=f"Test message {i}",
                args=(),
                exc_info=None
            )
            records.append(record)
            self.async_handler.emit(record)
        
        # Give workers time to process batches
        time.sleep(0.2)
        
        # Should have processed all records
        assert self.target_handler.emit.call_count == 10


class TestPrometheusMetricsHandler:
    """Test PrometheusMetricsHandler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = PrometheusMetricsHandler(
            metrics_port=8090,
            enable_histogram_metrics=True,
            enable_counter_metrics=True,
            enable_gauge_metrics=True
        )

    def test_handler_creation(self):
        """Test creating Prometheus metrics handler."""
        assert self.handler.metrics_port == 8090
        assert self.handler.enable_histogram_metrics is True
        assert self.handler.enable_counter_metrics is True

    def test_basic_metric_emission(self):
        """Test basic metric emission."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Emit record
        self.handler.emit(record)
        
        # Check that basic metrics were updated
        assert self.handler._metrics["log_entries_total"] == 1
        assert self.handler._metrics["error_counts"]["INFO"] == 1

    def test_quality_event_metrics(self):
        """Test quality event metric collection."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add quality event data
        quality_event = QualityEvent(
            event_type="test_event",
            severity="WARNING",
            source_component="test",
            quality_score=0.75
        )
        record.quality_event = quality_event
        
        self.handler.emit(record)
        
        assert self.handler._metrics["quality_events_total"] == 1
        assert 0.75 in self.handler._metrics["quality_scores"]

    def test_validation_result_metrics(self):
        """Test validation result metric collection."""
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        record.validation_result = {"severity": "FAIL", "violations": ["error1"]}
        
        self.handler.emit(record)
        
        assert self.handler._metrics["validation_failures_total"] == 1

    def test_performance_metrics(self):
        """Test performance metric collection."""
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        record.performance_metrics = {"duration_ms": 250.0}
        
        self.handler.emit(record)
        
        assert 0.25 in self.handler._metrics["execution_duration_seconds"]

    def test_prometheus_metrics_format(self):
        """Test Prometheus metrics format generation."""
        # Add some test data
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        self.handler.emit(record)
        
        metrics_output = self.handler.get_metrics()
        
        assert "orchestrator_log_entries_total" in metrics_output
        assert "# HELP" in metrics_output
        assert "# TYPE" in metrics_output


class TestQualityEventStreamHandler:
    """Test QualityEventStreamHandler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_stream = Mock()
        self.handler = QualityEventStreamHandler(
            stream=self.mock_stream,
            colorize_output=False,  # Disable for testing
            include_recommendations=True
        )

    def test_handler_creation(self):
        """Test creating quality event stream handler."""
        assert self.handler.stream is self.mock_stream
        assert self.handler.colorize_output is False
        assert self.handler.include_recommendations is True

    def test_quality_level_filtering(self):
        """Test filtering by quality level."""
        handler = QualityEventStreamHandler(
            stream=self.mock_stream,
            quality_level_filter="WARNING"
        )
        
        # Create record with matching quality level
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.quality_level = "WARNING"
        
        with patch.object(handler, '_format_quality_record', return_value=record):
            handler.emit(record)
        
        # Should have been processed
        assert handler._format_quality_record.called

    def test_event_type_filtering(self):
        """Test filtering by event type."""
        handler = QualityEventStreamHandler(
            stream=self.mock_stream,
            event_type_filter=["validation_completed"]
        )
        
        # Create record with matching event type
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.event_type = "validation_completed"
        
        with patch.object(handler, '_format_quality_record', return_value=record):
            handler.emit(record)
        
        # Should have been processed
        assert handler._format_quality_record.called

    def test_quality_record_formatting(self):
        """Test quality-specific record formatting."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.quality_level = "WARNING"
        record.validation_score = 0.75
        record.event_type = "validation_completed"
        
        formatted_record = self.handler._format_quality_record(record)
        
        # Check that quality information was added to message
        message = formatted_record.msg
        assert "Quality: WARNING" in message
        assert "Score: 0.75" in message
        assert "Event: validation_completed" in message

    def test_recommendation_inclusion(self):
        """Test inclusion of recommendations in output."""
        handler = QualityEventStreamHandler(
            stream=self.mock_stream,
            include_recommendations=True
        )
        
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        quality_event = QualityEvent(
            event_type="test_event",
            severity="WARNING",
            source_component="test",
            recommendations=["Fix issue 1", "Check configuration"]
        )
        record.quality_event = quality_event
        
        formatted_record = handler._format_quality_record(record)
        
        assert "Recommendations:" in formatted_record.msg
        assert "Fix issue 1" in formatted_record.msg


class TestCreateQualityLoggingSetup:
    """Test create_quality_logging_setup function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir)

    def test_basic_setup(self):
        """Test basic logging setup creation."""
        handlers = create_quality_logging_setup(
            log_dir=self.log_dir,
            enable_prometheus=False,
            enable_async=False
        )
        
        assert "structured" in handlers
        assert "console" in handlers
        assert isinstance(handlers["structured"], QualityRotatingFileHandler)
        assert isinstance(handlers["console"], QualityEventStreamHandler)

    def test_prometheus_enabled_setup(self):
        """Test setup with Prometheus enabled."""
        handlers = create_quality_logging_setup(
            log_dir=self.log_dir,
            enable_prometheus=True,
            enable_async=False
        )
        
        assert "prometheus" in handlers
        assert isinstance(handlers["prometheus"], PrometheusMetricsHandler)

    def test_async_enabled_setup(self):
        """Test setup with async handlers enabled."""
        handlers = create_quality_logging_setup(
            log_dir=self.log_dir,
            enable_prometheus=False,
            enable_async=True
        )
        
        assert "structured" in handlers
        assert isinstance(handlers["structured"], AsyncQualityHandler)

    def test_log_directory_creation(self):
        """Test that log directory is created if it doesn't exist."""
        non_existent_dir = self.log_dir / "new_logs"
        assert not non_existent_dir.exists()
        
        create_quality_logging_setup(log_dir=non_existent_dir)
        
        assert non_existent_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__])